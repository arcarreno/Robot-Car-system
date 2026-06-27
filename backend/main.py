"""
FastAPI Backend para Robot Car Control.

Endpoints:
  GET /health      -> health check
  GET /status      -> estado actual del backend + ESP32
  POST /comando    -> enviar comando manual al ESP32
  WS /ws           -> WebSocket: frames anotados + estado en tiempo real

Inicia automaticamente los threads T2 (MJPEG) y T3 (Procesamiento).
"""

import asyncio
import json
import queue
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware

from config import BACKEND_HOST, BACKEND_PORT, ESP32_IP, QUEUE_MAXSIZE
from clients.esp32_client import ESP32Client
from threads.mjpeg_thread import MJPEGThread
from threads.process_thread import ProcessThread
from models.frame_result import FrameResult

# =============================================================================
# Global state (inicializado en lifespan)
# =============================================================================
frame_queue: queue.Queue = None
result_queue: queue.Queue = None
mjpeg_thread: MJPEGThread = None
process_thread: ProcessThread = None
esp32: ESP32Client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa y limpia recursos al iniciar/detener la app."""
    global frame_queue, result_queue, mjpeg_thread, process_thread, esp32

    print("[Backend] Iniciando...")

    # Inicializar componentes
    esp32 = ESP32Client()
    # Safety: enviar stop al ESP32 al iniciar para asegurar que los
    # motores estén apagados (evita que el robot se mueva solo si
    # quedó un comando residual de una sesión anterior).
    esp32.send_command("stop")
    esp32.set_speed(150)  # velocidad por defecto
    frame_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
    result_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)

    # Iniciar threads
    mjpeg_thread = MJPEGThread(frame_queue)
    process_thread = ProcessThread(frame_queue, result_queue, esp32)

    # Primero el stream, luego el procesamiento
    # (esperar a que el MJPEG se conecte para no saturar el ESP32)
    mjpeg_thread.start()
    await asyncio.sleep(3)  # dar tiempo al ESP32 para establecer el stream
    process_thread.start()

    print(f"[Backend] Escuchando en {BACKEND_HOST}:{BACKEND_PORT}")
    print(f"[Backend] ESP32 en {ESP32_IP}")

    yield  # La app corre aqui

    # Cleanup
    print("[Backend] Deteniendo...")
    mjpeg_thread.stop()
    process_thread.stop()
    mjpeg_thread.join(timeout=3)
    process_thread.join(timeout=3)
    esp32.close()
    print("[Backend] Detenido.")


# =============================================================================
# App
# =============================================================================
app = FastAPI(
    title="Robot Car Backend",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS para el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://192.168.4.2:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Endpoints REST
# =============================================================================

@app.get("/health")
async def health():
    """Health check simple."""
    return {
        "status": "ok",
        "mjpeg_alive": mjpeg_thread.is_alive() if mjpeg_thread else False,
        "process_alive": process_thread.is_alive() if process_thread else False,
        "esp32_ping": esp32.ping() if esp32 else False,
    }


@app.get("/status")
async def status():
    """Estado actual del sistema."""
    result: FrameResult = None
    if process_thread:
        result = process_thread.get_last_result()

    sm = process_thread.get_state_machine() if process_thread else None

    return {
        "state": sm.current_state if sm else "UNKNOWN",
        "last_command": sm.last_command if sm else None,
        "mjpeg_fps": round(mjpeg_thread.fps_actual, 1) if mjpeg_thread else 0,
        "process_fps": round(result.fps if result else 0, 1),
        "esp32_connected": esp32.ping() if esp32 else False,
    }


@app.post("/comando")
async def enviar_comando(comando: str = Query(...)):
    """
    Enviar comando manual al ESP32.

    Ejemplo: POST /comando?comando=go
    """
    if not esp32:
        return {"error": "Backend no inicializado"}

    # Notificar a la maquina de estados
    if process_thread:
        sm = process_thread.get_state_machine()
        if comando in ("go", "back", "left", "right", "stop"):
            sm.on_keyboard_input()
            sm.set_active_command(comando)

    ok = esp32.send_command(comando)
    return {"comando": comando, "enviado": ok}


# =============================================================================
# WebSocket
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket de frames anotados + estado en tiempo real.

    El frontend se conecta y recibe:
    {
        "state": "COLOR_CHECK",
        "detected_color": "green",
        "command_sent": "go",
        "colors": [{"name": "green", "area": 15000}],
        "fps": 12.5,
        "frame_b64": "...",
    }
    """
    await websocket.accept()
    print(f"[WS] Cliente conectado")

    try:
        while True:
            # Recibir mensajes del frontend (ej: comando manual, toggle teclado)
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=0.05
                )
                if len(raw) > 4096:
                    print(f"[WS] Mensaje demasiado grande ({len(raw)} bytes), ignorado")
                    continue
                data = json.loads(raw)
                await _handle_ws_message(websocket, data)
            except json.JSONDecodeError:
                print(f"[WS] JSON invalido recibido, ignorado")
                continue
            except asyncio.TimeoutError:
                pass

            # Enviar resultado mas reciente (si hay). El frame_b64 puede
            # ser None cuando el process_thread lo throttlea, pero la
            # metadata de color/estado siempre viene -> el frontend
            # actualiza el Tablero a maxima frecuencia.
            sent_something = False
            if process_thread:
                try:
                    result = result_queue.get_nowait()
                    payload = result.to_dict()
                    await websocket.send_json(payload)
                    sent_something = True
                except queue.Empty:
                    pass

            # Si no habia resultado, mandar heartbeat ligero (solo estado)
            # para que el frontend sepa que el backend esta vivo.
            if not sent_something:
                try:
                    hb_state = (
                        process_thread.get_state_machine().current_state
                        if process_thread else "idle"
                    )
                    await websocket.send_json({
                        "type": "heartbeat",
                        "state": hb_state,
                        "timestamp": time.time(),
                    })
                except Exception as e:
                    # B6 fix: logear la excepción y cerrar el WS si el
                    # heartbeat falla (el cliente parece conectado pero
                    # no recibe datos -> necesita reconectarse).
                    print(f"[WS] Heartbeat falló: {e}")
                    break
                # Sleep mas largo cuando no hay nada: ahorra CPU
                await asyncio.sleep(0.1)
            else:
                # Hay resultados, no dormir mucho para no bloquear el stream
                await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print(f"[WS] Cliente desconectado")
    except Exception as e:
        print(f"[WS] Error: {e}")


async def _handle_ws_message(websocket: WebSocket, data: dict):
    """Procesa mensajes del frontend via WS."""
    action = data.get("action")
    ALLOWED_COMMANDS = {"go", "back", "left", "right", "stop", "ledon", "ledoff"}

    if action == "comando":
        comando = data.get("comando", "")
        if comando not in ALLOWED_COMMANDS:
            print(f"[WS] Comando invalido ignorado: '{comando}'")
            return
        if esp32 and process_thread:
            sm = process_thread.get_state_machine()
            sm.on_keyboard_input()
            sm.set_active_command(comando)
            esp32.send_command(comando)

    elif action == "tecla_up":
        comando = data.get("comando", "")
        if esp32 and process_thread:
            esp32.send_command("stop")
            sm = process_thread.get_state_machine()
            sm.on_keyboard_release(comando)
            sm.set_active_command("stop")

    elif action == "toggle_teclado":
        # El frontend activo/desactivo modo teclado
        pass

    elif action == "iniciar_ruta":
        if process_thread:
            sm = process_thread.get_state_machine()
            distancia = data.get("distancia", 5)
            # Mapeo bilingüe: low/baja, medium/media, high/alta
            velocidad_raw = data.get("velocidad", "media")
            velocidad_map = {"low": "baja", "medium": "media", "high": "alta"}
            velocidad = velocidad_map.get(velocidad_raw, velocidad_raw)
            giro_ms = data.get("giro_ms", 1600)

            # Set ESP32 speed BEFORE starting route so the motors
            # spin at the correct velocity during turns and advance.
            speed_pwm_map = {"low": 100, "baja": 100, "medium": 150, "media": 150, "high": 200, "alta": 200}
            esp32_speed = speed_pwm_map.get(velocidad, 150)
            if esp32:
                esp32.set_speed(esp32_speed)
                process_thread.set_base_speed(esp32_speed)

            sm.start_route(distancia, velocidad, giro_ms)
            # A4 fix: eliminado esp32.send_command("go") duplicado.
            # El state machine ya envia "go" via evaluate() en el
            # ProcessThread. Enviar "go" aca creaba un race condition
            # donde el ESP32 recibia "go" antes de que evaluate() pudiera
            # evaluar si hay un rojo en el primer frame.
            await websocket.send_json({
                "type": "route_status",
                "status": "started",
                "route_phase": "out",
                "route_progress": 0.0,
            })
            print(f"[WS] Ruta iniciada: {distancia}m, {velocidad}")

    elif action == "detener_ruta":
        if process_thread:
            sm = process_thread.get_state_machine()
            sm.stop_route()
            if esp32:
                esp32.reset_circuit_breaker()  # forzar recuperacion si estaba abierto
                esp32.send_command("stop")
            await websocket.send_json({
                "type": "route_status",
                "status": "stopped",
            })
            print("[WS] Ruta detenida por el usuario")

    elif action == "iniciar_continua":
        if process_thread:
            sm = process_thread.get_state_machine()
            sm.start_continuous()
            if esp32:
                speed = data.get("speed", 100)  # default 100 (baja pero sobre dead-zone)
                esp32.set_speed(speed)
                process_thread.set_base_speed(speed)
                # NO enviar "go" directamente: process_thread lo hara
                # en el proximo frame via _handle_continuous, respetando
                # la pipeline de comandos y el circuit breaker.
            await websocket.send_json({
                "type": "continuous_status",
                "status": "started",
            })
            print("[WS] Ruta continua iniciada")

    elif action == "detener_continua":
        if process_thread:
            sm = process_thread.get_state_machine()
            sm.stop_continuous()
            if esp32:
                esp32.reset_circuit_breaker()  # forzar recuperacion si estaba abierto
                esp32.send_command("stop")
            await websocket.send_json({
                "type": "continuous_status",
                "status": "stopped",
            })
            print("[WS] Ruta continua detenida por el usuario")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        reload=False,
    )
