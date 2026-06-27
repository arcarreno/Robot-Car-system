"""
Maquina de estados del robot.

Estados:
   IDLE           -> esperando, sin procesamiento activo
   COLOR_CHECK    -> evaluando colores de semaforo en cada frame
   MANUAL         -> control manual por teclado (override)
   ROUTE          -> ruta autonoma con vision de semaforos (ida y vuelta)
   CONTINUOUS     -> avanza siempre, reacciona a semaforos y flechas

Transiciones:
   IDLE ──(iniciar monitoreo)──> COLOR_CHECK
   COLOR_CHECK ──(rojo detectado)──> STOP via ESP32 ──> COLOR_CHECK
   CUALQUIERA ──(tecla presionada)──> MANUAL (*excepto ROUTE/CONTINUOUS)
   MANUAL ──(3s sin tecla)──> IDLE
   IDLE/COLOR_CHECK ──(iniciar_ruta)──> ROUTE
   ROUTE ──(detener_ruta)──> IDLE
   ROUTE ──(ruta_completada)──> IDLE
   IDLE/COLOR_CHECK ──(iniciar_continua)──> CONTINUOUS
   CONTINUOUS ──(detener_continua)──> IDLE

Evasion de obstaculos (v5 — escaneo):
   Zonas: danger (<0.4m), caution (<0.8m), pre_caution (<1.2m)
   Confirmacion: 2 frames consecutivos para activar
   Escaneo: robot escanea ambiente en vez de retroceder
     SCAN_180: escanea 180° (izq + der) en 2 segundos
     SCAN_360: escanea 360° en 4 segundos si no hay salida
   Giro adaptativo: gira hasta que el camino este despejado (no timer fijo)
   Histeresis: reanuda a threshold * 1.3 (evita oscilacion)
   IMPORTANTE: robot NUNCA retrocede — solo escanea y gira
"""

import threading
import time
from enum import Enum
from typing import Optional
from config import (
    OBSTACLE_BACKUP_TIME, OBSTACLE_COLLISION_DISTANCE, OBSTACLE_COLLISION_BACKUP_TIME,
    OBSTACLE_TTC_STOP, OBSTACLE_TTC_SLOWDOWN,
    OBSTACLE_TTC_BACKUP_MIN, OBSTACLE_TTC_BACKUP_MAX, OBSTACLE_TTC_BACKUP_SCALE,
    OBSTACLE_SCAN_180_TIME, OBSTACLE_SCAN_360_TIME, OBSTACLE_FREE_DISTANCE,
)

# Calibracion: ms por metro a velocidad base
MS_PER_METER = 1666

# Limites de giro (v2)
OBSTACLE_MAX_TURN_TIME = 3.0     # tope maximo de giro (seguridad)
OBSTACLE_MIN_TURN_TIME = 1.0     # tiempo minimo de giro (evita oscilacion)


class RobotState(Enum):
    IDLE = "IDLE"
    COLOR_CHECK = "COLOR_CHECK"
    MANUAL = "MANUAL"
    ROUTE = "ROUTE"
    CONTINUOUS = "CONTINUOUS"


class StateMachine:
    """
    Maquina de estados del robot.

    No contiene logica de hardware — solo decide QUE hacer.
    La ejecucion de comandos la hace ESP32Client.
    """

    def __init__(self, obstacle_turn_time: float = 1.0,
                 obstacle_threshold: float = 0.5,
                 obstacle_resume_factor: float = 1.5):
        self._obstacle_turn_time = obstacle_turn_time
        self._lock = threading.RLock()
        self.state = RobotState.IDLE
        self._last_manual_time = time.time()
        self._manual_timeout = 3.0  # segundos
        self._last_command: Optional[str] = None
        self._current_detected_color: Optional[str] = None
        self._current_detected_arrow: Optional[str] = None
        # Control manual con deteccion de colores
        self._active_command: Optional[str] = None  # ultimo comando manual enviado
        self._manual_paused = False  # True si el semaforo detuvo el avance manual

        # --- Ruta autonoma (ROUTE state) ---
        self._route_phase: Optional[str] = None       # "out" | "turn" | "back" | "done"
        self._route_target_time_s: float = 0.0        # tiempo total de movimiento (ida o vuelta)
        self._route_moving_time_s: float = 0.0        # tiempo acumulado de movimiento
        self._route_paused: bool = False              # pausado por semaforo
        self._route_last_time: float = 0.0            # ultimo time() de evaluate
        self._route_turn_time_s: float = 1.6          # duracion del giro 180°
        self._route_turn_start: float = 0.0           # cuando empezo el giro
        self._route_progress: float = 0.0             # 0.0 - 1.0
        self._route_completed: bool = False           # bloquea movimiento tras ruta

        # --- Ruta continua (CONTINUOUS state) ---
        self._continuous_paused: bool = False         # pausado por semaforo
        self._continuous_last_arrow: Optional[str] = None  # ultima flecha detectada

        # --- Deteccion de obstaculos (v4 — escaneo) ---
        self._obstacle_active: bool = False           # True si estamos esquivando
        self._obstacle_timer: float = 0.0             # timestamp de cuando empezo la fase
        self._obstacle_direction: Optional[str] = None  # "left" o "right"
        self._obstacle_threshold: float = obstacle_threshold
        self._obstacle_resume_factor: float = obstacle_resume_factor
        # v4: Fases de escaneo (reemplaza BACKUP)
        # idle → stop → scan_180 → scan_360 → turn
        self._obstacle_phase: str = "idle"     # "idle" | "stop" | "scan_180" | "scan_360" | "turn"
        self._obstacle_collision: bool = False  # True si colision confirmada (distance < 0.2m)
        self._obstacle_backup_time: float = OBSTACLE_BACKUP_TIME  # duracion del backup (OBSOLETO)
        self._obstacle_phase_sent: bool = False  # True si ya enviamos comando en esta fase
        self._obstacle_cooldown_until: float = 0.0  # timestamp hasta el cual no re-detectar

        # v4: Escaneo de ambiente
        self._obstacle_scan_results: list = []     # resultados del escaneo [(direction, distance)]
        self._obstacle_scan_phase_sent: bool = False  # True si ya enviamos comando en sub-fase de escaneo

        # v2: Confirmacion multi-frame
        self._obstacle_zone: str = "clear"            # "clear" | "pre_caution" | "caution" | "danger"

    def reset(self):
        """Vuelve a IDLE."""
        with self._lock:
            self.state = RobotState.IDLE
            self._last_command = None
            self._current_detected_color = None
            self._current_detected_arrow = None
            self._active_command = None
            self._manual_paused = False
            self._route_completed = False
            self._continuous_paused = False
            self._continuous_last_arrow = None
            self._obstacle_active = False
            self._obstacle_timer = 0.0
            self._obstacle_direction = None
            self._obstacle_zone = "clear"
            self._obstacle_phase = "idle"
            self._obstacle_collision = False
            self._obstacle_phase_sent = False
            self._reset_route()

    def set_active_command(self, command: Optional[str]):
        """Actualiza el comando manual activo (go/left/right/stop/back)."""
        with self._lock:
            self._active_command = command
            # Si el usuario cambio a un comando que no es de avance, resumir pausa
            if command not in ("go", "left", "right"):
                self._manual_paused = False

    def on_keyboard_input(self):
        """
        Cambia a MANUAL cuando el usuario presiona una tecla.
        No afecta al estado ROUTE/CONTINUOUS mientras esta en ejecucion.
        Si la ruta ya se completo (phase="done"), la desbloquea.
        Debe llamarse desde el WebSocket o endpoint REST.
        """
        with self._lock:
            if self.state == RobotState.ROUTE:
                if self._route_phase == "done":
                    # Ruta completada, salir a MANUAL
                    self._reset_route()
                    self.state = RobotState.MANUAL
                    self._last_manual_time = time.time()
                    return "go"
                return None  # no interrumpir ruta en ejecucion
            if self.state == RobotState.CONTINUOUS:
                return None  # no interrumpir ruta continua
            self._route_completed = False
            self.state = RobotState.MANUAL
            self._last_manual_time = time.time()
        return "go"  # comando por defecto al entrar a manual

    def on_keyboard_release(self, key: str):
        """
        Cuando se suelta una tecla en modo MANUAL.
        """
        with self._lock:
            self._last_manual_time = time.time()
        return "stop"

    def _check_manual_timeout(self) -> bool:
        """Si paso el timeout sin teclas, salir de MANUAL."""
        with self._lock:
            if self.state == RobotState.MANUAL:
                if time.time() - self._last_manual_time > self._manual_timeout:
                    self.state = RobotState.IDLE
                    return True
            return False

    # =========================================================================
    # Metodos de ruta autonoma
    # =========================================================================

    def start_route(self, distance_m: float, speed: str = "media",
                    turn_ms: int = 1600):
        """
        Inicia ruta autonoma con vision de semaforos.

        Args:
            distance_m: distancia en metros (ida = vuelta, debe ser > 0)
            speed: "baja" | "low" | "media" | "medium" | "alta" | "high"
            turn_ms: milisegundos para giro de 180°
        """
        with self._lock:
            if distance_m <= 0:
                print(f"[Route] Distancia invalida: {distance_m}m")
                return

            # Mapeo bilingüe para compatibilidad frontend
            speed_map = {
                "baja": 1.0, "low": 1.0,
                "media": 1.5, "medium": 1.5,
                "alta": 2.0, "high": 2.0,
            }
            speed_factor = speed_map.get(speed, 1.5)

            self._route_target_time_s = (
                distance_m * (MS_PER_METER / 1000.0) / speed_factor
            )
            self._route_moving_time_s = 0.0
            self._route_paused = False
            self._route_phase = "out"
            self._route_last_time = time.time()
            # Scale turn time with speed: faster = needs less time, slower = needs more
            turn_speed_multiplier = {1.0: 1.2, 1.5: 1.0, 2.0: 0.8}.get(speed_factor, 1.0)
            self._route_turn_time_s = (turn_ms / 1000.0) * turn_speed_multiplier
            self._route_progress = 0.0
            self._route_completed = False  # nueva ruta, desbloquear
            self._clear_obstacle_state()  # limpiar estado de obstaculo previo
            self.state = RobotState.ROUTE
            self._last_command = "go"
            print(f"[Route] Iniciada: {distance_m}m, velocidad={speed}, "
                  f"target={self._route_target_time_s:.1f}s de movimiento, "
                  f"giro={self._route_turn_time_s:.1f}s")

    def stop_route(self):
        """Detiene la ruta en curso y vuelve a IDLE."""
        with self._lock:
            if self.state != RobotState.ROUTE:
                return
            self._reset_route()
            self._clear_obstacle_state()
            self.state = RobotState.IDLE
            self._last_command = "stop"
            print("[Route] Detenida por el usuario")

    # =========================================================================
    # Metodos de ruta continua
    # =========================================================================

    def start_continuous(self):
        """Inicia ruta continua: avanza siempre, reacciona a semaforos y flechas."""
        with self._lock:
            self._continuous_paused = False
            self._continuous_last_arrow = None
            self._route_completed = False
            self._clear_obstacle_state()  # limpiar estado de obstaculo previo
            self.state = RobotState.CONTINUOUS
            self._last_command = "go"
            print("[Continuous] Iniciada")

    def stop_continuous(self):
        """Detiene la ruta continua y vuelve a IDLE."""
        with self._lock:
            if self.state != RobotState.CONTINUOUS:
                return
            self._continuous_paused = False
            self._continuous_last_arrow = None
            self._clear_obstacle_state()
            self.state = RobotState.IDLE
            self._last_command = "stop"
            print("[Continuous] Detenida por el usuario")

    def _handle_continuous(self,
                           detected_color: Optional[str],
                           detected_arrow: Optional[str]) -> Optional[str]:
        """
        Logica de ruta continua con vision de semaforos y flechas.

        Comportamiento:
          - Siempre intenta avanzar (go)
          - Rojo/Amarillo -> STOP (pausa)
          - Verde sin flecha -> GO (avanzar)
          - Verde + flecha izquierda -> LEFT
          - Verde + flecha derecha -> RIGHT
          - Sin color -> GO (siempre avanza)
        """
        # Prioridad 1: ROJO o AMARILLO -> STOP + pausa
        if detected_color in ("red", "yellow"):
            if not self._continuous_paused:
                self._continuous_paused = True
                self._last_command = "stop"
                print(f"[Continuous] Pausada por {detected_color.upper()}")
            return "stop"

        # Si estaba pausado y ahora esta despejado -> reanudar
        if self._continuous_paused:
            self._continuous_paused = False
            self._last_command = "go"
            print("[Continuous] Reanudada")
            # No retornar aqui: dejar que evaluar la flecha

        # Prioridad 2: FLECHA -> girar (funciona con o sin semaforo verde)
        if detected_arrow == "left":
            self._continuous_last_arrow = "left"
            self._last_command = "left"
            return "left"
        elif detected_arrow == "right":
            self._continuous_last_arrow = "right"
            self._last_command = "right"
            return "right"

        # Prioridad 3: VERDE sin flecha -> avanzar
        if detected_color == "green":
            self._continuous_last_arrow = None
            self._last_command = "go"
            return "go"

        # Sin color detectado: AVANZAR (el robot SIEMPRE se mueve en
        # CONTINUOUS salvo que un semáforo o obstáculo lo detenga).
        self._last_command = "go"
        return "go"

    def _reset_route(self):
        """Limpia todo el estado de ruta."""
        self._route_phase = None
        self._route_target_time_s = 0.0
        self._route_moving_time_s = 0.0
        self._route_paused = False
        self._route_last_time = 0.0
        self._route_turn_time_s = 1.6
        self._route_turn_start = 0.0
        self._route_progress = 0.0
        self._route_completed = False

    def _clear_obstacle_state(self):
        """Limpia estado de obstáculo (llamar al cambiar de modo)."""
        self._obstacle_active = False
        self._obstacle_timer = 0.0
        self._obstacle_direction = None
        self._obstacle_zone = "clear"
        self._obstacle_phase = "idle"
        self._obstacle_collision = False
        self._obstacle_phase_sent = False
        self._obstacle_scan_results = []
        self._obstacle_scan_phase_sent = False

    def _handle_route(self,
                      detected_color: Optional[str]) -> Optional[str]:
        """
        Logica de ruta autonoma con vision de semaforos.

        Fases:
          "out"  -> avanzando ida (respeta semaforos)
          "turn" -> girando 180° (ignora semaforos, necesario para completar giro)
          "back" -> avanzando vuelta (respeta semaforos)
          "done" -> ruta completada -> IDLE
        """
        now = time.time()
        phase = self._route_phase

        # --- Fase "done": ruta completada, robot detenido ---
        if phase == "done":
            self._route_completed = True
            self._route_progress = 1.0
            self._last_command = "stop"
            return "stop"

        # --- Fase "turn": girando 180° (no chequea semaforos) ---
        if phase == "turn":
            elapsed = now - self._route_turn_start
            if elapsed < self._route_turn_time_s:
                self._route_progress = 0.45  # 45% fijo durante el giro
                self._last_command = "right"
                return "right"
            else:
                # Giro completo -> empezar vuelta
                self._route_phase = "back"
                self._route_moving_time_s = 0.0
                self._route_last_time = now
                self._last_command = "stop"
                self._route_progress = 0.45
                print("[Route] Giro completado, iniciando regreso")
                return "stop"

        # --- Fase "out" o "back": avanzando ---
        is_out = (phase == "out")
        target_time = self._route_target_time_s

        # Prioridad 1: ROJO o AMARILLO -> STOP + pausa
        if detected_color in ("red", "yellow"):
            if not self._route_paused:
                self._route_paused = True
                self._last_command = "stop"
                print(f"[Route] {'Ida' if is_out else 'Vuelta'} pausada por "
                      f"{detected_color.upper()}")
            return "stop"

        # Si estaba pausado y ahora está despejado -> reanudar
        if self._route_paused:
            self._route_paused = False
            self._route_last_time = now  # reiniciar reloj para evitar catch-up
            self._route_progress = self._calc_route_progress(is_out)
            self._last_command = "go"
            print(f"[Route] {'Ida' if is_out else 'Vuelta'} reanudada")
            return "go"

        # Avance normal: acumular tiempo de movimiento
        dt = now - self._route_last_time
        self._route_last_time = now
        self._route_moving_time_s += dt
        self._route_progress = self._calc_route_progress(is_out)

        if self._route_moving_time_s >= target_time:
            if is_out:
                # Ida completada -> empezar giro
                self._route_phase = "turn"
                self._route_turn_start = now
                self._route_progress = 0.45
                self._last_command = "stop"
                print(f"[Route] Ida completada ({self._route_moving_time_s:.1f}s)")
                return "stop"
            else:
                # Vuelta completada -> done
                self._route_phase = "done"
                self._route_progress = 1.0
                self._last_command = "stop"
                print(f"[Route] Vuelta completada ({self._route_moving_time_s:.1f}s)")
                return "stop"

        self._last_command = "go"
        return "go"

    def _calc_route_progress(self, is_out: bool) -> float:
        """
        Calcula progreso de ruta 0.0-1.0:
          out:  0% -> 45%
          turn: 45% (fijo)
          back: 45% -> 90%
          done: 100%
        """
        if self._route_target_time_s <= 0:
            return 0.0
        fraction = min(self._route_moving_time_s / self._route_target_time_s, 1.0)
        if is_out:
            return fraction * 0.45
        else:
            return 0.45 + fraction * 0.45

    # =========================================================================
    # Evaluate (v2 — polished obstacle avoidance)
    # =========================================================================

    def evaluate(self,
                 detected_color: Optional[str],
                 detected_arrow: Optional[str] = None,
                 obstacle_distance: Optional[float] = None,
                 obstacle_direction: Optional[str] = None,
                 obstacle_zone: Optional[str] = None,
                 obstacle_confidence: Optional[float] = None,
                 obstacle_ttc: Optional[float] = None,
                 obstacle_approach_speed: float = 0.0) -> Optional[str]:
        """
        Evalua el estado actual y decide que comando enviar.

        v3: Color y obstaculos se evaluan INDEPENDIENTEMENTE.
        El comando mas urgente gane:
          stop por semaforo > stop por obstaculo > girando > color > go

        Args:
            detected_color: "red" | "yellow" | "green" | None
            detected_arrow: "left" | "right" | None
            obstacle_distance: metros al obstaculo mas cercano (None = sin datos)
            obstacle_direction: "left" | "right" — lado con mas espacio
            obstacle_zone: "clear" | "pre_caution" | "caution" | "danger"
            obstacle_confidence: 0.0-1.0, cuantos frames confirmaron

        Returns:
            str | None: comando a enviar ("go", "stop", "left", "right", None)
        """

        with self._lock:
            # Verificar timeout de modo manual
            if self._check_manual_timeout():
                return "stop"  # timeout → detener y volver a IDLE

            self._current_detected_color = detected_color
            self._current_detected_arrow = detected_arrow

            # =================================================================
            # MANUAL: detectar colores solo si el robot avanza
            # =================================================================
            if self.state == RobotState.MANUAL:
                if self._active_command in ("go", "left", "right"):
                    if detected_color == "red" or detected_color == "yellow":
                        self._manual_paused = True
                        self._last_command = "stop"
                        return "stop"
                    if detected_color == "green" and self._manual_paused:
                        self._manual_paused = False
                        self._last_command = self._active_command
                        return self._active_command
                return None

            # =================================================================
            # COLOR CHECK / IDLE: solo observar
            # =================================================================
            if self.state in (RobotState.COLOR_CHECK, RobotState.IDLE):
                if self.state == RobotState.IDLE:
                    self.state = RobotState.COLOR_CHECK
                return None

            # =================================================================
            # ROUTE / CONTINUOUS: evaluar obstaculo y color INDEPENDIENTEMENTE
            # =================================================================

            # --- Paso 1: Evaluar obstaculo ---
            obstacle_cmd = None
            if self.state in (RobotState.ROUTE, RobotState.CONTINUOUS):
                obstacle_cmd = self._evaluate_obstacle(
                    obstacle_distance, obstacle_direction,
                    obstacle_zone, obstacle_confidence,
                    obstacle_ttc, obstacle_approach_speed)

            # --- Paso 2: Evaluar color (SIEMPRE, no depende del obstaculo) ---
            color_cmd = None
            if self.state == RobotState.ROUTE:
                color_cmd = self._handle_route(detected_color)
            elif self.state == RobotState.CONTINUOUS:
                color_cmd = self._handle_continuous(detected_color, detected_arrow)

            # --- Paso 3: Merge — prioridad mas urgente gana ---
            # stop por semaforo > stop/back por obstaculo > turning > color > go
            result = None
            if color_cmd == "stop":
                result = "stop"
            elif self._obstacle_active and obstacle_cmd is None:
                # CRITICAL FIX: Durante backup/turn, _evaluate_obstacle retorna None
                # ("ESP32 ya esta ejecutando"). NO podemos dejar que "go" del
                # route/continuous handler pase y sobreescriba la evitacion.
                # El ESP32 mantiene su ultimo comando (back/left/right).
                result = None
            elif obstacle_cmd == "stop":
                result = "stop"
            elif obstacle_cmd == "back":
                result = "back"
            elif obstacle_cmd in ("left", "right"):
                result = obstacle_cmd
            elif color_cmd:
                result = color_cmd
            else:
                result = "go"

            self._last_command = result
            return result

    def _evaluate_obstacle(self,
                           obstacle_distance, obstacle_direction,
                           obstacle_zone, obstacle_confidence,
                           obstacle_ttc=None, obstacle_approach_speed=0.0):
        """
        Evalua obstaculo y retorna comando (stop/left/right) o None.

        v5: Escaneo de ambiente (reemplaza BACKUP):
          idle → stop → scan_180 → scan_360 → turn → resume

          Cuando se detecta un obstáculo:
          1. STOP: parada inicial (1 frame)
          2. SCAN_180: escanea 180° (izq + der) en 2 segundos
             - Si hay espacio libre (> 0.3m) → TURN
             - Si no hay → SCAN_360
          3. SCAN_360: escanea 360° en 4 segundos
          4. TURN: gira hasta que el camino esté despejado

          El robot NUNCA retrocede — solo escanea y gira.
        """

        # --- Fase STOP: parada inicial (1 frame) ---
        if self._obstacle_phase == "stop":
            # Transicion a SCAN_180
            self._obstacle_phase = "scan_180"
            self._obstacle_timer = time.time()
            self._obstacle_scan_results = []
            self._obstacle_scan_phase_sent = False
            print(f"[Obstacle] STOP → SCAN_180 (distancia={obstacle_distance:.2f}m)")
            return "stop"

        # --- Fase SCAN_180: escanear 180° (2 segundos) ---
        if self._obstacle_phase == "scan_180":
            elapsed = time.time() - self._obstacle_timer

            # Sub-fase 1: Giro izquierda 90° (0 - 0.7s)
            if elapsed < 0.7:
                if not self._obstacle_phase_sent:
                    self._obstacle_phase_sent = True
                    print(f"[Obstacle] SCAN_180: girando izquierda")
                return "left"

            # Sub-fase 2: Parada y análisis izquierda (0.7 - 1.0s)
            elif elapsed < 1.0:
                if not self._obstacle_scan_phase_sent:
                    self._obstacle_scan_phase_sent = True
                    # Guardar medición izquierda
                    self._obstacle_scan_results.append({
                        "direction": "left",
                        "distance": obstacle_distance if obstacle_distance else float("inf"),
                    })
                    print(f"[Obstacle] SCAN_180: izquierda = {obstacle_distance:.2f}m")
                return "stop"

            # Sub-fase 3: Giro derecha 180° (1.0 - 1.7s)
            elif elapsed < 1.7:
                if not self._obstacle_phase_sent:
                    self._obstacle_phase_sent = False  # reset para nueva sub-fase
                    print(f"[Obstacle] SCAN_180: girando derecha")
                return "right"

            # Sub-fase 4: Parada y análisis derecha (1.7 - 2.0s)
            elif elapsed < 2.0:
                if not self._obstacle_scan_phase_sent:
                    self._obstacle_scan_phase_sent = True
                    # Guardar medición derecha
                    self._obstacle_scan_results.append({
                        "direction": "right",
                        "distance": obstacle_distance if obstacle_distance else float("inf"),
                    })
                    print(f"[Obstacle] SCAN_180: derecha = {obstacle_distance:.2f}m")
                return "stop"

            # Análisis final: decidir si hay salida en 180°
            else:
                best_dir = self._analyze_scan_results()
                if best_dir:
                    # Hay espacio libre → girar a esa dirección
                    self._obstacle_direction = best_dir
                    self._obstacle_phase = "turn"
                    self._obstacle_timer = time.time()
                    self._obstacle_phase_sent = False
                    print(f"[Obstacle] SCAN_180 completo: mejor dirección = {best_dir}")
                    return best_dir
                else:
                    # No hay espacio libre → escanear 360°
                    self._obstacle_phase = "scan_360"
                    self._obstacle_timer = time.time()
                    self._obstacle_scan_results = []
                    self._obstacle_scan_phase_sent = False
                    print(f"[Obstacle] SCAN_180: sin salida libre, iniciando SCAN_360")
                    return "stop"

        # --- Fase SCAN_360: escanear 360° (4 segundos) ---
        if self._obstacle_phase == "scan_360":
            elapsed = time.time() - self._obstacle_timer

            # Giro continuo con paradas cada 1 segundo (4 puntos: 0°, 90°, 180°, 270°)
            if elapsed < 4.0:
                # Determinar en qué segundo estamos
                current_second = int(elapsed)
                sub_elapsed = elapsed - current_second

                # Primer frame de cada segundo: guardar medición
                if sub_elapsed < 0.1 and not self._obstacle_scan_phase_sent:
                    self._obstacle_scan_phase_sent = True
                    self._obstacle_scan_results.append({
                        "direction": f"angle_{current_second * 90}",
                        "distance": obstacle_distance if obstacle_distance else float("inf"),
                    })
                    print(f"[Obstacle] SCAN_360: ángulo {current_second * 90}° = {obstacle_distance:.2f}m")

                # Entre 0.1s y 0.8s de cada segundo: girar
                if sub_elapsed >= 0.1:
                    self._obstacle_scan_phase_sent = False  # reset para próximo segundo

                return "right"
            else:
                # Análisis final: encontrar mejor dirección
                best_dir = self._find_best_direction_from_scan()
                self._obstacle_direction = best_dir or "left"
                self._obstacle_phase = "turn"
                self._obstacle_timer = time.time()
                self._obstacle_phase_sent = False
                print(f"[Obstacle] SCAN_360 completo: mejor dirección = {self._obstacle_direction}")
                return self._obstacle_direction

        # --- Fase TURN: girar hasta que el camino esté despejado ---
        if self._obstacle_phase == "turn":
            elapsed = time.time() - self._obstacle_timer

            # Safety: tope maximo de giro
            if elapsed >= OBSTACLE_MAX_TURN_TIME:
                self._reset_obstacle()
                print("[Obstacle] Giro timeout, reanudando por seguridad")
                return "stop"

            # Verificar si el camino esta despejado
            if (elapsed >= OBSTACLE_MIN_TURN_TIME
                    and obstacle_distance is not None
                    and obstacle_distance > self._obstacle_threshold * self._obstacle_resume_factor):
                self._reset_obstacle()
                print(f"[Obstacle] Despejado ({obstacle_distance:.2f}m > "
                      f"{self._obstacle_threshold * self._obstacle_resume_factor:.2f}m), "
                      f"reanudando")
                return None

            # Enviar giro solo en el primer frame — ESP32 mantiene el comando
            if not self._obstacle_phase_sent:
                self._obstacle_phase_sent = True
                return self._obstacle_direction or "left"
            return None  # ESP32 ya esta girando

        # --- Fase IDLE: detectar nuevo obstáculo ---
        # Cooldown post-evitación: no re-detectar inmediatamente
        if time.time() < self._obstacle_cooldown_until:
            return None

        is_confident = (
            obstacle_confidence is not None
            and obstacle_confidence >= 1.0
        )
        # Doble umbral: distancia OR TTC
        is_close = (
            obstacle_distance is not None
            and obstacle_distance < self._obstacle_threshold
        )
        is_ttc_urgent = (
            obstacle_ttc is not None
            and obstacle_ttc < OBSTACLE_TTC_STOP
        )

        if is_confident and (is_close or is_ttc_urgent):
            # Reset cooldown — estamos en una nueva secuencia de evitación
            self._obstacle_cooldown_until = 0.0

            # Detectar colisión (muy cerca)
            is_collision = (
                obstacle_distance is not None
                and obstacle_distance < OBSTACLE_COLLISION_DISTANCE
            )

            direction = obstacle_direction or "left"
            self._obstacle_active = True
            self._obstacle_direction = direction
            self._obstacle_collision = is_collision

            print(f"[Obstacle] Detectado a {obstacle_distance:.2f}m "
                  f"(zona={obstacle_zone}, conf={obstacle_confidence:.1f}), "
                  f"STOP → SCAN_180")

            # Primero STOP, luego el next frame entrará a SCAN_180
            self._obstacle_phase = "stop"
            self._obstacle_timer = time.time()
            self._obstacle_scan_results = []
            self._last_command = "stop"
            return "stop"

        return None

    def _analyze_scan_results(self) -> Optional[str]:
        """
        Analiza los resultados del escaneo 180° y retorna la mejor dirección.

        Returns:
            "left" | "right" | None (si no hay espacio libre)
        """
        if not self._obstacle_scan_results:
            return None

        # Buscar direcciones con espacio libre (> OBSTACLE_FREE_DISTANCE)
        free_dirs = [
            r for r in self._obstacle_scan_results
            if r["distance"] > OBSTACLE_FREE_DISTANCE
        ]

        if not free_dirs:
            return None  # bloqueado por todos lados

        # Retornar la dirección con más distancia (más espacio)
        best = max(free_dirs, key=lambda x: x["distance"])
        return best["direction"]

    def _find_best_direction_from_scan(self) -> Optional[str]:
        """
        Encuentra la mejor dirección a partir de los resultados del escaneo 360°.

        Returns:
            "left" | "right" | None (si no hay espacio libre)
        """
        if not self._obstacle_scan_results:
            return None

        # Buscar direcciones con espacio libre (> OBSTACLE_FREE_DISTANCE)
        free_dirs = [
            r for r in self._obstacle_scan_results
            if r["distance"] > OBSTACLE_FREE_DISTANCE
        ]

        if not free_dirs:
            return None  # bloqueado por todos lados

        # Retornar la dirección con más distancia (más espacio)
        best = max(free_dirs, key=lambda x: x["distance"])

        # Mapear ángulos a direcciones de giro
        angle_str = best["direction"]
        if angle_str == "angle_0":
            return "left"  # 0° = adelante, pero ya hay obstáculo → girar izq
        elif angle_str == "angle_90":
            return "right"  # 90° derecha
        elif angle_str == "angle_180":
            return "right"  # 180° = atrás, girar derecha para volver
        elif angle_str == "angle_270":
            return "left"  # 270° = izquierda
        else:
            return best["direction"]

    def _reset_obstacle(self):
        """Limpia todo el estado de obstáculo."""
        self._obstacle_active = False
        self._obstacle_timer = 0.0
        self._obstacle_direction = None
        self._obstacle_phase = "idle"
        self._obstacle_collision = False
        self._obstacle_zone = "clear"
        self._obstacle_phase_sent = False
        self._obstacle_scan_results = []
        self._obstacle_scan_phase_sent = False
        # Cooldown: 1.0s después de evitación para evitar re-detección inmediata
        self._obstacle_cooldown_until = time.time() + 1.0

    @property
    def current_state(self) -> str:
        with self._lock:
            return self.state.value

    @property
    def last_command(self) -> Optional[str]:
        with self._lock:
            return self._last_command

    @property
    def route_progress(self) -> Optional[float]:
        with self._lock:
            if self.state != RobotState.ROUTE:
                return None
            return self._route_progress

    @property
    def route_phase(self) -> Optional[str]:
        with self._lock:
            if self.state != RobotState.ROUTE:
                return None
            return self._route_phase

    @property
    def obstacle_active(self) -> bool:
        """True si el robot esta esquivando un obstaculo."""
        with self._lock:
            return self._obstacle_active

    @property
    def obstacle_direction(self) -> Optional[str]:
        """Direccion del giro de obstaculo: 'left' o 'right'."""
        with self._lock:
            return self._obstacle_direction
