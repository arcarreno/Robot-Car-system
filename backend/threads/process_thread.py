"""
Thread 3: Procesa frames con OpenCV y produce resultados.

Toma frames de frame_queue (T2), ejecuta:
  - YOLO + ColorDetector (cada frame)
  - StateMachine evaluacion
  - Anotacion de frame

Pone los resultados en result_queue para el WebSocket (T1).
"""

import threading
import queue
import time
import numpy as np
from typing import Optional
from config import (
    WATCHDOG_TIMEOUT, YOLO_MODEL_PATH, YOLO_CONF_THRESHOLD, YOLO_ENABLED,
    YOLO_SKIP_FRAMES, FRAME_SKIP,
    DEPTH_ENABLED, DEPTH_MODEL_PATH, DEPTH_SKIP_FRAMES, DEPTH_DEVICE,
    DEPTH_BILATERAL_D, DEPTH_BILATERAL_SIGMA_COLOR, DEPTH_BILATERAL_SIGMA_SPACE,
    MIDAS_CALIBRATION_K, MIDAS_CALIBRATION_P, MIDAS_CALIBRATION_OFFSET,
    DEPTH_ZONE_DANGER, DEPTH_ZONE_CAUTION, DEPTH_ZONE_PRE_CAUTION,
    DEPTH_CLOSE_RAW_THRESHOLD, DEPTH_CLOSE_RATIO_THRESHOLD,
    OBSTACLE_STOP_THRESHOLD, OBSTACLE_RESUME_FACTOR, OBSTACLE_TURN_TIME,
    OBSTACLE_SLOWDOWN_DISTANCE, OBSTACLE_SPEED_REDUCTION, OBSTACLE_FREE_DISTANCE,
)
from core.color_detector import ColorDetector
from core.yolo_detector import YoloDetector
from core.state_machine import StateMachine
from core.depth_estimator import DepthEstimator
from clients.esp32_client import ESP32Client
from models.frame_result import FrameResult
from utils.anotador import annotate_frame, frame_to_b64


class ProcessThread(threading.Thread):
    """
    Hilo de procesamiento OpenCV.

    Atributos:
        frame_queue: cola de entrada (frames del MJPEGThread).
        result_queue: cola de salida (resultados para el WebSocket).
        running: flag para detener.
        esp32: cliente ESP32.
        state_machine: maquina de estados.
    """

    def __init__(self,
                 frame_queue: queue.Queue,
                 result_queue: queue.Queue,
                 esp32_client: ESP32Client):
        super().__init__(name="ProcessThread", daemon=True)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.running = True
        self.esp32 = esp32_client
        self.sm = StateMachine(
            obstacle_turn_time=OBSTACLE_TURN_TIME,
            obstacle_threshold=OBSTACLE_STOP_THRESHOLD,
            obstacle_resume_factor=OBSTACLE_RESUME_FACTOR,
        )

        self._fps = 0.0
        self._last_fps_time = time.time()
        self._fps_counter = 0
        self._last_result: Optional[FrameResult] = None
        self._last_command: Optional[str] = None

        # Throttle para set_speed: evitar flooding HTTP al ESP32
        self._last_speed_set_time: float = 0.0
        self._last_speed_value: int = 255
        self._base_speed: int = 255

        # Watchdog: detecta si la camara deja de enviar frames
        self._last_frame_time = time.time()
        self._watchdog_active = True  # se desactiva en pruebas si es necesario

        self._command_refresh_counter = 0  # para refresh periodico de comandos

        # FASE 4 (B3): contador de frames para skip de YOLO. Cuando llega a
        # YOLO_SKIP_FRAMES, corre YOLO y resetea; entremedio, usa el ultimo
        # resultado cacheado. Asi: YOLO_SKIP_FRAMES=1 -> corre cada 2do frame.
        self._yolo_frame_counter = 0
        self._yolo_enabled_runtime = YOLO_ENABLED and YOLO_SKIP_FRAMES >= 0

        # Throttle del frame JPEG: la metadata de color/estado se manda a
        # maxima frecuencia (15+ FPS), pero el frame anotado (JPEG base64)
        # pesa ~10-20KB y se manda cada FRAME_SKIP+1 frames para no
        # saturar el WS. Resultado: el color en el Tablero de Estados se
        # ve instantaneo, el video solo cuando hay frame nuevo.
        self._frame_skip_counter = 0

        # Desaceleracion gradual para amarillo
        self._yellow_decel_active = False
        self._yellow_decel_start = 0.0
        self._yellow_decel_duration = 1.0  # segundos para desacelerar

        # Timer de giro por flecha (45° = 1000ms)
        self._arrow_turn_active = False
        self._arrow_turn_start = 0.0
        self._arrow_turn_duration = 1.0  # segundos para giro de 45°
        self._arrow_turn_command = None  # "left" o "right"

        # Command queue para no bloquear el pipeline con HTTP
        self._command_queue = queue.Queue(maxsize=5)

        # Inicializar YOLO si esta habilitado
        self._yolo = None
        if YOLO_ENABLED:
            self._yolo = YoloDetector(
                model_path=YOLO_MODEL_PATH,
                conf_threshold=YOLO_CONF_THRESHOLD,
            )

        self._color_detector = ColorDetector(yolo_detector=self._yolo)

        # --- Depth Estimation (MiDaS Small) ---
        self._depth_estimator: Optional[DepthEstimator] = None
        self._depth_frame_counter = 0
        self._cached_obstacle_distance: Optional[float] = None
        self._cached_obstacle_direction: Optional[str] = None
        self._cached_obstacle_zone: str = "clear"
        self._cached_obstacle_confidence: float = 0.0
        self._cached_obstacle_ttc: Optional[float] = None
        self._cached_obstacle_approach_speed: float = 0.0
        if DEPTH_ENABLED:
            self._depth_estimator = DepthEstimator(
                model_path=DEPTH_MODEL_PATH,
                device=DEPTH_DEVICE,
                calibration_k=MIDAS_CALIBRATION_K,
                calibration_p=MIDAS_CALIBRATION_P,
                calibration_offset=MIDAS_CALIBRATION_OFFSET,
                zone_danger=DEPTH_ZONE_DANGER,
                zone_caution=DEPTH_ZONE_CAUTION,
                zone_pre_caution=DEPTH_ZONE_PRE_CAUTION,
                bilateral_d=DEPTH_BILATERAL_D,
                bilateral_sigma_color=DEPTH_BILATERAL_SIGMA_COLOR,
                bilateral_sigma_space=DEPTH_BILATERAL_SIGMA_SPACE,
                close_raw_threshold=DEPTH_CLOSE_RAW_THRESHOLD,
                close_ratio_threshold=DEPTH_CLOSE_RATIO_THRESHOLD,
                free_distance=OBSTACLE_FREE_DISTANCE,
            )

        # Hilo sender de comandos (daemon)
        self._command_sender_thread = threading.Thread(
            target=self._command_sender, name="CommandSender", daemon=True
        )
        self._command_sender_thread.start()

    def stop(self):
        """Detiene el hilo."""
        self.running = False

    def run(self):
        print("[ProcessThread] Iniciando procesamiento OpenCV...")

        while self.running:
            try:
                # Tomar frame de la cola (timeout para poder verificar running)
                try:
                    frame = self.frame_queue.get(timeout=0.5)
                except queue.Empty:
                    # Watchdog: si no llegan frames, detener el robot.
                    # B4 fix: durante ROUTE usar timeout mas tolerante (5s)
                    # para no interrumpir ruta por un parpadeo de cámara,
                    # pero SÍ actuar si la cámara muere por completo.
                    if self._watchdog_active:
                        elapsed = time.time() - self._last_frame_time
                        route_timeout = 5.0 if self.sm.current_state == "ROUTE" else WATCHDOG_TIMEOUT
                        if elapsed > route_timeout:
                            self.sm.reset()
                            self.esp32.reset_circuit_breaker()  # forzar recuperacion
                            if self._depth_estimator is not None:
                                self._depth_estimator.reset_temporal()
                            self._last_command = None
                            self._enqueue_command("stop")
                    continue

                if frame is None:
                    continue

                # Watchdog: actualizar timestamp del ultimo frame
                self._last_frame_time = time.time()

                # FPS tracking
                self._fps_counter += 1
                now = time.time()
                if now - self._last_fps_time >= 1.0:
                    self._fps = self._fps_counter / (now - self._last_fps_time)
                    self._fps_counter = 0
                    self._last_fps_time = now

                # --- Deteccion de colores (cada frame) ---
                # FASE 4 (B3): decidir si este frame corre YOLO o usa el cache.
                # Estrategia: si YOLO_SKIP_FRAMES >= 0 y habilitado, correr YOLO
                # solo cuando el counter llega a N; sino, en TODOS los frames.
                run_yolo_this_frame = True
                if self._yolo_enabled_runtime and self._yolo is not None and YOLO_SKIP_FRAMES > 0:
                    run_yolo_this_frame = (self._yolo_frame_counter == 0)
                color_result = self._color_detector.detect(
                    frame, run_yolo=run_yolo_this_frame,
                )
                # Avanzar counter SOLO si YOLO esta activo (si no hay modelo, no
                # tiene sentido contar)
                if self._yolo_enabled_runtime and self._yolo is not None and YOLO_SKIP_FRAMES > 0:
                    self._yolo_frame_counter = (self._yolo_frame_counter + 1) % (YOLO_SKIP_FRAMES + 1)
                detected_color = color_result["detected"]
                detected_arrow = color_result["arrow"]
                color_overlay = color_result["overlay"]

                # DEBUG: log cada 60 frames
                self._debug_frame_count = getattr(self, '_debug_frame_count', 0) + 1
                if self._debug_frame_count % 60 == 0:
                    yolo_det = self._color_detector._yolo._last_result if self._color_detector._yolo else None
                    sem = yolo_det.get("semaforo") if yolo_det else None
                    print(f"[Frame {self._debug_frame_count}] semaforo={sem is not None} "
                          f"color={detected_color} arrow={detected_arrow}")

                # --- Depth Estimation (cada N frames, v2 — con analyze) ---
                if (self._depth_estimator is not None
                        and self._depth_estimator.available):
                    run_depth = (self._depth_frame_counter == 0)
                    if run_depth:
                        depth_map = self._depth_estimator.estimate(frame)
                        if depth_map is not None:
                            result = self._depth_estimator.analyze(depth_map)
                            self._cached_obstacle_distance = result.distance
                            self._cached_obstacle_direction = result.direction
                            self._cached_obstacle_zone = result.zone
                            self._cached_obstacle_confidence = result.confidence
                            self._cached_obstacle_ttc = result.ttc
                            self._cached_obstacle_approach_speed = result.approach_speed
                        else:
                            # Depth falló: limpiar cache para no mantener datos viejos
                            self._cached_obstacle_distance = None
                            self._cached_obstacle_direction = None
                            self._cached_obstacle_zone = "clear"
                            self._cached_obstacle_confidence = 0.0
                            self._cached_obstacle_ttc = None
                            self._cached_obstacle_approach_speed = 0.0
                    self._depth_frame_counter = (
                        (self._depth_frame_counter + 1) % (DEPTH_SKIP_FRAMES + 1)
                    )

                # --- Evaluar maquina de estados (v2 — con zonas y confianza) ---
                command = self.sm.evaluate(
                    detected_color, detected_arrow,
                    obstacle_distance=self._cached_obstacle_distance,
                    obstacle_direction=self._cached_obstacle_direction,
                    obstacle_zone=self._cached_obstacle_zone,
                    obstacle_confidence=self._cached_obstacle_confidence,
                    obstacle_ttc=self._cached_obstacle_ttc,
                    obstacle_approach_speed=self._cached_obstacle_approach_speed,
                )

                # --- Desaceleracion gradual para amarillo ---
                # Solo activar si el stop viene del amarillo, no de obstáculo
                if (detected_color == "yellow" and command == "stop"
                        and not self.sm.obstacle_active):
                    if not self._yellow_decel_active:
                        self._yellow_decel_active = True
                        self._yellow_decel_start = time.time()
                        print("[ProcessThread] Amarillo detectado, desacelerando...")

                    elapsed = time.time() - self._yellow_decel_start
                    if elapsed < self._yellow_decel_duration:
                        # Enviar velocidad decreciente
                        speed_factor = 1.0 - (elapsed / self._yellow_decel_duration)
                        speed_value = int(255 * speed_factor)
                        self.esp32.set_speed(speed_value)
                        # No enviar stop aun
                    else:
                        # Desaceleracion completa, enviar stop final
                        self._yellow_decel_active = False
                        self._yellow_decel_start = 0.0
                        self._drain_command_queue()
                        self._enqueue_command("stop")
                        self._last_command = "stop"
                        print("[ProcessThread] Desaceleracion completa, STOP")
                else:
                    # No es amarillo: cancelar desaceleracion si hay ROJO, VERDE,
                    # o si el color desaparece (None = camara fallo / parpadeo).
                    # A6 fix: antes solo cancelaba con rojo/verde, permitiendo
                    # que la desaceleracion continuara con None (frame sin
                    # deteccion), lo que dejaba el robot moviendose 1s despues
                    # de que el amarillo desaparecia.
                    if self._yellow_decel_active:
                        if detected_color in ("red", "green") or detected_color is None:
                            self._yellow_decel_active = False
                            self._yellow_decel_start = 0.0
                            if detected_color is not None:
                                print(f"[ProcessThread] Desaceleracion cancelada ({detected_color})")

                    # --- Timer de giro por flecha (45° = 1000ms) ---
                    if self._arrow_turn_active:
                        elapsed = time.time() - self._arrow_turn_start
                        if elapsed >= self._arrow_turn_duration:
                            # Timer expirado: reanudar movimiento
                            self._arrow_turn_active = False
                            self._arrow_turn_start = 0.0
                            self._arrow_turn_command = None
                            command = "go"
                            print("[ProcessThread] Giro flecha completado, reanudando")
                        elif self.sm.obstacle_active:
                            # Obstáculo detectado: cancelar timer de flecha
                            # y dejar que la evitación de obstáculo actúe
                            self._arrow_turn_active = False
                            self._arrow_turn_start = 0.0
                            self._arrow_turn_command = None
                            print("[ProcessThread] Timer flecha cancelado (obstáculo activo)")
                        elif command in ("left", "right"):
                            # Timer activo: suprimir nuevo comando de flecha
                            # (mantener el giro en curso)
                            command = self._arrow_turn_command

                    # Si hay un comando de flecha nuevo (fuera de timer), iniciar timer
                    if (command in ("left", "right")
                        and not self._arrow_turn_active
                        and not self.sm.obstacle_active):
                        self._arrow_turn_active = True
                        self._arrow_turn_start = time.time()
                        self._arrow_turn_command = command
                        print(f"[ProcessThread] Giro flecha: {command} (1000ms)")

                    # --- Speed reduction al acercarse al obstaculo ---
                    if (command == "go"
                            and self._cached_obstacle_distance is not None
                            and self._cached_obstacle_distance < OBSTACLE_SLOWDOWN_DISTANCE):
                        # Reducir proporcionalmente: a 0.8m → 50% speed, a 1.5m → ~100%
                        ratio = self._cached_obstacle_distance / OBSTACLE_SLOWDOWN_DISTANCE
                        speed_factor = max(OBSTACLE_SPEED_REDUCTION, ratio)
                        speed_value = int(255 * speed_factor)
                        now = time.time()
                        if (abs(speed_value - self._last_speed_value) > 26
                                or (now - self._last_speed_set_time) > 0.2):
                            self.esp32.set_speed(speed_value)
                            self._last_speed_set_time = now
                            self._last_speed_value = speed_value
                    elif command == "go":
                        # Restaurar velocidad base (user-selected o default 255)
                        now = time.time()
                        if self._last_speed_value != self._base_speed and (
                                abs(self._base_speed - self._last_speed_value) > 26
                                or (now - self._last_speed_set_time) > 0.2):
                            self.esp32.set_speed(self._base_speed)
                            self._last_speed_set_time = now
                            self._last_speed_value = self._base_speed

                    # --- Enviar comando al ESP32 con safety features ---
                    # Defense-in-depth: no enviar "go" si evitacion activa
                    if (command == "go" and self.sm.obstacle_active):
                        command = None  # suprimir — ESP32 mantiene su accion actual

                    if command is not None:
                        is_new = command != self._last_command
                        is_stop = command == "stop"  # solo stop es emergencia

                        if is_stop:
                            # STOP: emergencia, drain queue y enviar siempre
                            self._drain_command_queue()
                            self._enqueue_command(command)
                            self._last_command = command
                            self._command_refresh_counter = 0

                        elif is_new:
                            # Comando nuevo: enviar normalmente
                            self._enqueue_command(command)
                            self._last_command = command
                            self._command_refresh_counter = 0

                        else:
                            # Mismo comando sostenido: refresh periodico (~cada 3 frames)
                            # B2 fix: antes esto estaba en la rama `else` externa que solo
                            # se alcanzaba con `command is None` (inalcanzable). Ahora vive
                            # dentro del `if command is not None`, cuando el comando se
                            # repite (turn/go). Necesario para que el ESP32 mantenga accion
                            # continua: sin refresh, el robot se detiene a los pocos frames.
                            self._command_refresh_counter += 1
                            if self._command_refresh_counter >= 3:
                                self._command_refresh_counter = 0
                                self._enqueue_command(command)

                # --- Anotar frame ---
                annotated = annotate_frame(
                    frame=frame,
                    color_overlay=color_overlay,
                )

                # Throttle del JPEG: solo codear cada FRAME_SKIP+1 frames.
                # cv2.imencode + base64 cuesta ~3-5ms que se ahorran en
                # frames intermedios. La metadata (color/estado) se manda
                # igual a maxima frecuencia.
                send_full_frame = (
                    FRAME_SKIP <= 0
                    or self._frame_skip_counter == 0
                )
                if send_full_frame:
                    frame_payload = frame_to_b64(annotated)
                else:
                    frame_payload = None
                self._frame_skip_counter = (self._frame_skip_counter + 1) % (FRAME_SKIP + 1)

                # --- Armar resultado (incluye datos de ruta si aplica) ---
                obstacle_detected = (
                    self._cached_obstacle_distance is not None
                    and self._cached_obstacle_distance < OBSTACLE_STOP_THRESHOLD
                )
                result = FrameResult(
                    state=self.sm.current_state,
                    detected_color=detected_color,
                    detected_arrow=detected_arrow,
                    command_sent=command,
                    colors=[
                        {"name": c["name"], "area": c["area"]}
                        for c in color_result["colors"]
                    ],
                    fps=self._fps,
                    frame_b64=frame_payload,
                    route_progress=self.sm.route_progress,
                    route_phase=self.sm.route_phase,
                    obstacle_distance=self._cached_obstacle_distance,
                    obstacle_detected=obstacle_detected,
                    obstacle_direction=self._cached_obstacle_direction,
                    obstacle_ttc=self._cached_obstacle_ttc,
                    obstacle_approach_speed=self._cached_obstacle_approach_speed,
                )
                self._last_result = result

                # --- Poner resultado en cola (no bloqueante) ---
                try:
                    self.result_queue.put_nowait(result)
                except queue.Full:
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(result)
                    except queue.Empty:
                        pass

            except Exception as e:
                print(f"[ProcessThread] Error en pipeline: {e}")
                continue

        print("[ProcessThread] Detenido.")

    def _command_sender(self):
        """Hilo daemon que drena la cola de comandos y los envia al ESP32."""
        while self.running:
            try:
                cmd = self._command_queue.get(timeout=0.5)
                if cmd:
                    self.esp32.send_command(cmd)
            except queue.Empty:
                continue

    def _drain_command_queue(self):
        """Vacia la cola de comandos (para priorizar STOP)."""
        while not self._command_queue.empty():
            try:
                self._command_queue.get_nowait()
            except queue.Empty:
                break

    def _enqueue_command(self, command: str):
        """Pone comando en la cola, descartando el mas viejo si esta llena.
        Coalescing: si el ultimo en la cola es el mismo comando, no duplicar."""
        # Coalescing: peek at the last item in the underlying deque (O(1))
        try:
            q = self._command_queue.queue
            if q and q[-1] == command:
                return
        except AttributeError:
            pass
        try:
            self._command_queue.put_nowait(command)
        except queue.Full:
            try:
                self._command_queue.get_nowait()
                self._command_queue.put_nowait(command)
            except queue.Empty:
                pass

    def get_last_result(self) -> Optional[FrameResult]:
        """Obtiene el ultimo resultado sin consumir la cola."""
        return self._last_result

    def get_state_machine(self) -> StateMachine:
        return self.sm

    def set_base_speed(self, speed: int):
        """Actualiza la velocidad base del robot (usada al restaurar tras obstaculo)."""
        self._base_speed = max(0, min(255, speed))
