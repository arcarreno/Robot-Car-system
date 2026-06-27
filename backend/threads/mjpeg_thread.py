"""
Thread 2: Consume el stream MJPEG del ESP32 y pone los frames en una cola.

Usa el parser manual mjpeg_generator().
La cola tiene maxsize=2 para descartar frames viejos si el procesamiento va lento.
"""

import threading
import queue
import time
from core.mjpeg_parser import mjpeg_generator


# Si no llega ningun frame en este tiempo, log de warning
FRAME_STALE_TIMEOUT = 8
# Log de "stream vivo" cada N segundos
STREAM_ALIVE_LOG_INTERVAL = 30


class MJPEGThread(threading.Thread):
    """
    Hilo dedicado a bajar frames del ESP32.

    Atributos:
        frame_queue: cola de entrada para process_thread.
        running: flag para detener el hilo.
        fps_actual: tasa de frames que llegan del ESP32.
    """

    def __init__(self, frame_queue: queue.Queue):
        super().__init__(name="MJPEGThread", daemon=True)
        self.frame_queue = frame_queue
        self.running = True
        self.fps_actual = 0.0
        self._frame_count = 0
        self._last_time = time.time()
        self._consecutive_nulls = 0  # contador de frames nulos consecutivos

    def stop(self):
        """Detiene el hilo."""
        self.running = False

    def run(self):
        print("[MJPEGThread] Iniciando captura de stream...")
        fps_timer = time.time()
        last_frame_received = time.time()
        last_alive_log = time.time()

        while self.running:
            try:
                for frame in mjpeg_generator():
                    if not self.running:
                        break

                    if frame is not None:
                        # Reset counter de nulos
                        self._consecutive_nulls = 0

                        # Contar fps
                        self._frame_count += 1
                        now = time.time()
                        if now - fps_timer >= 1.0:
                            self.fps_actual = self._frame_count / (now - fps_timer)
                            self._frame_count = 0
                            fps_timer = now

                        last_frame_received = now

                        # Poner en cola (si esta llena, descarta el frame mas viejo)
                        try:
                            self.frame_queue.put_nowait(frame)
                        except queue.Full:
                            # Cola llena -> descartar frame viejo y poner el nuevo
                            try:
                                self.frame_queue.get_nowait()
                                self.frame_queue.put_nowait(frame)
                            except queue.Empty:
                                pass
                    else:
                        # Frame nulo = error de conexion
                        self._consecutive_nulls += 1
                        if self._consecutive_nulls >= 10:
                            print(f"[MJPEGThread] {self._consecutive_nulls} frames nulos "
                                  f"consecutivos — forzando reconexion")
                            self._consecutive_nulls = 0
                            break  # salir del for → reconectar
                        time.sleep(1)

                    # Deteccion de stream congelado
                    elapsed = time.time() - last_frame_received
                    if elapsed > FRAME_STALE_TIMEOUT:
                        print(f"[MJPEGThread] WARNING: sin frames nuevos "
                              f"en {elapsed:.0f}s — stream posiblemente congelado")

                    # Log periodico de stream vivo (cada 30s)
                    now_alive = time.time()
                    if now_alive - last_alive_log >= STREAM_ALIVE_LOG_INTERVAL:
                        print(f"[MJPEGThread] Stream vivo — "
                              f"FPS={self.fps_actual:.1f}, "
                              f"total_frames={self._frame_count}, "
                              f"queue_size={self.frame_queue.qsize()}")
                        last_alive_log = now_alive

            except Exception as e:
                print(f"[MJPEGThread] Generador crash: {e}, reiniciando en 3s...")
                time.sleep(3)

        print("[MJPEGThread] Detenido.")
