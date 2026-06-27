"""
Parser manual de stream MJPEG.

Mas confiable que cv2.VideoCapture porque:
  1. No acumula buffer de frames viejos
  2. Reconexion automatica ante fallos
  3. Frame skipping natural (siempre toma el frame mas reciente)
  4. Multiplataforma (no depende de backends de OpenCV)
  5. Timeout por chunk para detectar streams congelados
  6. Socket timeout para detectar conexiones half-open (ESP32 freeze)
"""

import socket
import cv2
import numpy as np
import requests
from config import ESP32_STREAM, MJPEG_TIMEOUT, MJPEG_CHUNK_SIZE


# Timeout por chunk (segundos). Si no llega ningun chunk en este
# tiempo, el stream se considera congelado y se reconecta.
CHUNK_TIMEOUT = 5


def mjpeg_generator(url: str = ESP32_STREAM):
    """
    Generador infinito de frames desde un stream MJPEG.

    Args:
        url: URL del stream MJPEG del ESP32.

    Yields:
        frame (np.ndarray | None): Frame decodificado, o None si hay error.
    """
    import time

    MAX_BUFFER_SIZE = 1024 * 1024  # 1MB max para evitar OOM con stream corrupto
    buf = b""
    session = requests.Session()
    last_frame_time = time.time()

    while True:
        resp = None
        try:
            resp = session.get(url, stream=True, timeout=MJPEG_TIMEOUT)
            resp.raise_for_status()
            last_frame_time = time.time()  # reset al conectar

            # CRITICAL FIX: setear socket timeout para detectar half-open connections
            # Si el ESP32 deja de enviar datos sin cerrar TCP, iter_content() bloquea
            # para siempre. Con socket timeout, lanza socket.timeout y permite
            # la reconexion automatica.
            try:
                sock = resp.raw._fp.fp.raw._sock
                sock.settimeout(CHUNK_TIMEOUT)
            except (AttributeError, TypeError):
                # Fallback: si no podemos acceder al socket (urllib3 version nueva),
                # configurar el timeout a nivel de respuesta
                pass

            for chunk in resp.iter_content(MJPEG_CHUNK_SIZE):
                if not chunk:
                    continue

                buf += chunk

                # B3 fix: si el buffer crece demasiado (stream corrupto sin
                # marker de fin JPEG), truncarlo pero buscando el proximo
                # inicio de JPEG valido (\xff\xd8)
                if len(buf) > MAX_BUFFER_SIZE:
                    # Buscar el ultimo inicio de JPEG en el buffer reciente
                    recent = buf[-MAX_BUFFER_SIZE:]
                    cut_pos = recent.rfind(b'\xff\xd8')
                    if cut_pos > 0:
                        buf = recent[cut_pos:]
                    else:
                        buf = recent
                    print(f"[MJPEG] Buffer truncado, proximo JPEG en pos={cut_pos}")

                # Buscar el JPEG mas RECIENTE en el buffer
                # FF D8 = inicio, FF D9 = fin de JPEG
                start = buf.rfind(b'\xff\xd8')
                end = buf.rfind(b'\xff\xd9', start)

                if start != -1 and end != -1 and end > start:
                    jpg_bytes = buf[start:end + 2]
                    buf = buf[end + 2:]
                    last_frame_time = time.time()

                    frame = cv2.imdecode(
                        np.frombuffer(jpg_bytes, np.uint8),
                        cv2.IMREAD_COLOR
                    )
                    if frame is not None:
                        # Detectar frame negro (cámara cubierta / warmup / falla)
                        if frame.mean() < 5.0:
                            # Frame visualmente negro — yield None para que
                            # MJPEGThread's null counter incremente y active
                            # la reconexion forzada durante warmup de camara
                            yield None
                            continue
                        yield frame
                    else:
                        yield None

                # Deteccion de stream congelado: si no llega un JPEG
                # completo en CHUNK_TIMEOUT segundos, cortar y reconectar
                elif time.time() - last_frame_time > CHUNK_TIMEOUT:
                    print(f"[MJPEG] Stream congelado (sin JPEG "
                          f"completo en {CHUNK_TIMEOUT}s), reconectando...")
                    buf = b""
                    break  # salir del for, reconectar en el while

        except (requests.exceptions.RequestException,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                socket.timeout) as e:
            print(f"[MJPEG] Error de conexion: {e}")
            yield None
        except Exception as e:
            print(f"[MJPEG] Error inesperado: {e}")
            yield None
        finally:
            # Cerrar respuesta y sesion para liberar la conexion del ESP32
            if resp is not None:
                try:
                    resp.close()
                except Exception:
                    pass
            try:
                session.close()
            except Exception:
                pass
            # Recrear sesion limpia para el siguiente intento
            session = requests.Session()
            buf = b""
            time.sleep(2)  # backoff antes de reintentar



