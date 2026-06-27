"""Tests para la maquina de estados del robot."""
import sys
import os
import unittest
from unittest.mock import patch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.state_machine import StateMachine, RobotState


class TestStateMachineBasics(unittest.TestCase):
    """Tests basicos de la maquina de estados."""

    def test_initial_state_is_idle(self):
        sm = StateMachine()
        self.assertEqual(sm.current_state, "IDLE")

    def test_reset_returns_to_idle(self):
        sm = StateMachine()
        sm.start_route(5.0)
        sm.reset()
        self.assertEqual(sm.current_state, "IDLE")

    def test_last_command_initially_none(self):
        sm = StateMachine()
        self.assertIsNone(sm.last_command)


class TestColorCheck(unittest.TestCase):
    """Tests del estado COLOR_CHECK — solo observa, no mueve."""

    def test_idle_transitions_to_color_check_on_first_frame(self):
        sm = StateMachine()
        result = sm.evaluate(None)
        self.assertEqual(sm.current_state, "COLOR_CHECK")
        # COLOR_CHECK no envia comandos — el robot no se mueve solo
        self.assertIsNone(result)

    def test_red_returns_none(self):
        sm = StateMachine()
        sm.evaluate(None)
        result = sm.evaluate("red")
        self.assertIsNone(result)

    def test_green_returns_none(self):
        sm = StateMachine()
        sm.evaluate(None)
        result = sm.evaluate("green")
        self.assertIsNone(result)

    def test_yellow_returns_none(self):
        sm = StateMachine()
        sm.evaluate(None)
        result = sm.evaluate("yellow")
        self.assertIsNone(result)

    def test_no_color_returns_none(self):
        sm = StateMachine()
        sm.evaluate(None)
        result = sm.evaluate(None)
        self.assertIsNone(result)

    def test_green_with_left_arrow_returns_none(self):
        sm = StateMachine()
        sm.evaluate(None)
        result = sm.evaluate("green", "left")
        self.assertIsNone(result)

    def test_green_with_right_arrow_returns_none(self):
        sm = StateMachine()
        sm.evaluate(None)
        result = sm.evaluate("green", "right")
        self.assertIsNone(result)

    def test_state_stays_color_check(self):
        """COLOR_CHECK nunca cambia a otro estado automaticamente."""
        sm = StateMachine()
        sm.evaluate(None)
        for color in ["red", "green", "yellow", None]:
            sm.evaluate(color)
        self.assertEqual(sm.current_state, "COLOR_CHECK")


class TestManual(unittest.TestCase):
    """Tests del estado MANUAL."""

    def test_keyboard_input_enters_manual(self):
        sm = StateMachine()
        sm.evaluate(None)
        sm.on_keyboard_input()
        self.assertEqual(sm.current_state, "MANUAL")

    def test_manual_returns_none_for_non_advance_command(self):
        sm = StateMachine()
        sm.on_keyboard_input()
        sm.set_active_command("stop")
        result = sm.evaluate("green")
        self.assertIsNone(result)

    def test_manual_returns_stop_on_red(self):
        sm = StateMachine()
        sm.on_keyboard_input()
        sm.set_active_command("go")
        result = sm.evaluate("red")
        self.assertEqual(result, "stop")

    def test_manual_timeout_returns_to_idle(self):
        import time
        sm = StateMachine()
        sm.on_keyboard_input()
        sm._last_manual_time = time.time() - 4.0
        sm.evaluate(None)
        self.assertEqual(sm.current_state, "IDLE")

    def test_manual_pause_on_red_then_resume_on_green(self):
        """E3: MANUAL pausa en rojo y reanuda en verde."""
        sm = StateMachine()
        sm.on_keyboard_input()
        sm.set_active_command("go")

        # Robot avanzando, detecta rojo -> pausa
        result = sm.evaluate("red")
        self.assertEqual(result, "stop")
        self.assertTrue(sm._manual_paused)

        # Ahora verde -> reanuda
        result = sm.evaluate("green")
        self.assertEqual(result, "go")
        self.assertFalse(sm._manual_paused)

    def test_manual_pause_on_yellow(self):
        """E3: MANUAL pausa en amarillo."""
        sm = StateMachine()
        sm.on_keyboard_input()
        sm.set_active_command("go")

        result = sm.evaluate("yellow")
        self.assertEqual(result, "stop")
        self.assertTrue(sm._manual_paused)

    def test_manual_no_pause_for_stop_command(self):
        """E3: MANUAL no pausa si el comando activo es stop."""
        sm = StateMachine()
        sm.on_keyboard_input()
        sm.set_active_command("stop")

        result = sm.evaluate("red")
        self.assertIsNone(result)
        self.assertFalse(sm._manual_paused)

    def test_manual_resume_resets_paused_on_non_advance(self):
        """E3: Cambiar a comando no-avance desactiva la pausa."""
        sm = StateMachine()
        sm.on_keyboard_input()
        sm.set_active_command("go")
        sm.evaluate("red")  # pausa
        self.assertTrue(sm._manual_paused)

        sm.set_active_command("stop")  # cambiar a stop
        self.assertFalse(sm._manual_paused)


class TestRoute(unittest.TestCase):
    """Tests del estado ROUTE."""

    def test_start_route(self):
        sm = StateMachine()
        sm.start_route(5.0)
        self.assertEqual(sm.current_state, "ROUTE")

    def test_route_returns_go(self):
        sm = StateMachine()
        sm.start_route(5.0)
        result = sm.evaluate("green")
        self.assertEqual(result, "go")

    def test_route_stops_on_red(self):
        sm = StateMachine()
        sm.start_route(5.0)
        result = sm.evaluate("red")
        self.assertEqual(result, "stop")

    def test_stop_route(self):
        sm = StateMachine()
        sm.start_route(5.0)
        sm.stop_route()
        self.assertEqual(sm.current_state, "IDLE")

    def test_keyboard_input_does_not_interrupt_route(self):
        sm = StateMachine()
        sm.start_route(5.0)
        result = sm.on_keyboard_input()
        self.assertIsNone(result)
        self.assertEqual(sm.current_state, "ROUTE")

    def test_route_pause_on_red_then_resume(self):
        """E1: ROUTE pausa en rojo y reanuda cuando desaparece."""
        sm = StateMachine()
        sm.start_route(5.0)

        # Avanzando
        result = sm.evaluate("green")
        self.assertEqual(result, "go")

        # Rojo -> pausa
        result = sm.evaluate("red")
        self.assertEqual(result, "stop")
        self.assertTrue(sm._route_paused)

        # Verde -> reanuda
        result = sm.evaluate("green")
        self.assertEqual(result, "go")
        self.assertFalse(sm._route_paused)

    @patch('core.state_machine.time')
    def test_route_lifecycle_out_turn_back_done(self, mock_time):
        """E1: Lifecycle completo: out → turn → back → done."""
        # Setup: tiempo inicial
        mock_time.time.return_value = 1000.0

        sm = StateMachine()
        # 100m a velocidad baja (factor 1.0): target_time = 100 * (1666/1000) / 1.0 = 166.6s
        sm.start_route(100.0, "baja", turn_ms=2000)
        self.assertEqual(sm._route_phase, "out")
        self.assertEqual(sm.current_state, "ROUTE")

        # Simular avance por 166.7s (ida completa, un poco más por float precision)
        mock_time.time.return_value = 1166.7
        result = sm.evaluate("green")
        # Ida completada -> debería haber cambiado a "turn"
        self.assertEqual(sm._route_phase, "turn")
        self.assertEqual(result, "stop")

        # Simular giro por 2.4s (turn_ms=2000 * 1.2 multiplier for speed_factor=1.0)
        mock_time.time.return_value = 1169.2
        result = sm.evaluate(None)
        # Giro completado -> debería haber cambiado a "back"
        self.assertEqual(sm._route_phase, "back")
        self.assertEqual(result, "stop")

        # Simular vuelta por 166.7s más
        mock_time.time.return_value = 1335.9
        result = sm.evaluate("green")
        # Vuelta completada -> debería haber cambiado a "done"
        self.assertEqual(sm._route_phase, "done")
        self.assertEqual(result, "stop")
        self.assertEqual(sm.route_progress, 1.0)

    def test_route_progress_calculation(self):
        """E1: Progreso se calcula correctamente."""
        sm = StateMachine()
        sm.start_route(5.0)
        self.assertEqual(sm._route_progress, 0.0)

        # Avanzar un poco
        sm._route_moving_time_s = 2.0
        sm._route_target_time_s = 5.0
        progress = sm._calc_route_progress(is_out=True)
        self.assertAlmostEqual(progress, 0.45 * (2.0 / 5.0), places=2)

    def test_route_keyboard_input_after_done(self):
        """E1: Tecla después de ruta completada sale de ROUTE."""
        sm = StateMachine()
        sm.start_route(5.0)
        sm._route_phase = "done"
        sm._route_completed = True

        result = sm.on_keyboard_input()
        self.assertEqual(result, "go")
        self.assertEqual(sm.current_state, "MANUAL")


class TestContinuous(unittest.TestCase):
    """Tests del estado CONTINUOUS."""

    def test_start_continuous(self):
        sm = StateMachine()
        sm.start_continuous()
        self.assertEqual(sm.current_state, "CONTINUOUS")

    def test_continuous_returns_go_on_green(self):
        sm = StateMachine()
        sm.start_continuous()
        result = sm.evaluate("green")
        self.assertEqual(result, "go")

    def test_continuous_stops_on_red(self):
        sm = StateMachine()
        sm.start_continuous()
        result = sm.evaluate("red")
        self.assertEqual(result, "stop")

    def test_continuous_stops_on_yellow(self):
        sm = StateMachine()
        sm.start_continuous()
        result = sm.evaluate("yellow")
        self.assertEqual(result, "stop")

    def test_continuous_turns_left_on_arrow(self):
        sm = StateMachine()
        sm.start_continuous()
        result = sm.evaluate("green", "left")
        self.assertEqual(result, "left")

    def test_continuous_turns_right_on_arrow(self):
        sm = StateMachine()
        sm.start_continuous()
        result = sm.evaluate("green", "right")
        self.assertEqual(result, "right")

    def test_continuous_advances_on_no_color(self):
        """CONTINUOUS retorna 'go' cuando no hay color — SIEMPRE avanza."""
        sm = StateMachine()
        sm.start_continuous()
        result = sm.evaluate(None)
        self.assertEqual(result, "go")

    def test_stop_continuous(self):
        sm = StateMachine()
        sm.start_continuous()
        sm.stop_continuous()
        self.assertEqual(sm.current_state, "IDLE")

    def test_keyboard_input_does_not_interrupt_continuous(self):
        sm = StateMachine()
        sm.start_continuous()
        result = sm.on_keyboard_input()
        self.assertIsNone(result)
        self.assertEqual(sm.current_state, "CONTINUOUS")

    def test_continuous_resumes_after_red(self):
        sm = StateMachine()
        sm.start_continuous()
        sm.evaluate("red")
        result = sm.evaluate("green")
        self.assertEqual(result, "go")
        self.assertFalse(sm._continuous_paused)

    def test_continuous_arrow_change_during_green(self):
        """E2: Flecha cambia de izquierda a derecha durante verde."""
        sm = StateMachine()
        sm.start_continuous()

        # Verde + izquierda
        result = sm.evaluate("green", "left")
        self.assertEqual(result, "left")
        self.assertEqual(sm._continuous_last_arrow, "left")

        # Verde + derecha (cambio)
        result = sm.evaluate("green", "right")
        self.assertEqual(result, "right")
        self.assertEqual(sm._continuous_last_arrow, "right")

    def test_continuous_green_without_arrow_after_arrow(self):
        """E2: Verde sin flecha después de tener flecha -> avanzar."""
        sm = StateMachine()
        sm.start_continuous()

        # Verde + izquierda
        sm.evaluate("green", "left")
        self.assertEqual(sm._continuous_last_arrow, "left")

        # Verde sin flecha
        result = sm.evaluate("green")
        self.assertEqual(result, "go")
        self.assertIsNone(sm._continuous_last_arrow)

    def test_continuous_pause_on_yellow_then_resume(self):
        """E2: CONTINUOUS pausa en amarillo y reanuda."""
        sm = StateMachine()
        sm.start_continuous()

        # Amarillo -> pausa
        result = sm.evaluate("yellow")
        self.assertEqual(result, "stop")
        self.assertTrue(sm._continuous_paused)

        # Verde -> reanuda
        result = sm.evaluate("green")
        self.assertEqual(result, "go")
        self.assertFalse(sm._continuous_paused)

    def test_continuous_multiple_reds_keep_paused(self):
        """E2: Múltiples frames rojos mantienen la pausa."""
        sm = StateMachine()
        sm.start_continuous()

        sm.evaluate("red")
        self.assertTrue(sm._continuous_paused)

        sm.evaluate("red")
        self.assertTrue(sm._continuous_paused)

        sm.evaluate("red")
        self.assertTrue(sm._continuous_paused)


class TestObstacleAvoidance(unittest.TestCase):
    """Tests de deteccion y esquivamiento de obstaculos."""

    def test_obstacle_ignored_in_manual(self):
        """Obstaculo cercano en MANUAL → ignorado (usuario tiene control total)."""
        sm = StateMachine()
        sm.on_keyboard_input()  # entrar a MANUAL
        sm.set_active_command("go")

        # Simular evaluate con obstaculo cercano — NO debe activar evasion
        result = sm.evaluate(
            detected_color=None,
            detected_arrow=None,
            obstacle_distance=0.3,  # < 0.5 threshold
            obstacle_direction="left",
            obstacle_zone="danger",
            obstacle_confidence=1.0,
        )
        # En MANUAL sin color detectado, evaluate retorna None (no interviene)
        # Lo importante es que NO se activó la evasión de obstáculos
        self.assertIsNone(result)
        self.assertFalse(sm.obstacle_active)

    def test_obstacle_stop_in_continuous(self):
        """Obstaculo cercano en CONTINUOUS → stop (con confianza confirmada)."""
        sm = StateMachine()
        sm.start_continuous()

        result = sm.evaluate(
            detected_color="green",
            detected_arrow=None,
            obstacle_distance=0.2,
            obstacle_direction="right",
            obstacle_zone="danger",
            obstacle_confidence=1.0,
        )
        self.assertEqual(result, "stop")
        self.assertTrue(sm.obstacle_active)

    def test_obstacle_stop_in_route(self):
        """Obstaculo cercano en ROUTE → stop (con confianza confirmada)."""
        sm = StateMachine()
        sm.start_route(5.0)

        result = sm.evaluate(
            detected_color=None,
            obstacle_distance=0.4,
            obstacle_direction="left",
            obstacle_zone="danger",
            obstacle_confidence=1.0,
        )
        self.assertEqual(result, "stop")
        self.assertTrue(sm.obstacle_active)

    def test_obstacle_no_effect_in_color_check(self):
        """Obstaculo en COLOR_CHECK NO afecta (solo observa)."""
        sm = StateMachine()
        sm.evaluate(None)  # entrar a COLOR_CHECK

        result = sm.evaluate(
            detected_color=None,
            obstacle_distance=0.1,
            obstacle_direction="left",
            obstacle_zone="danger",
            obstacle_confidence=1.0,
        )
        # COLOR_CHECK no envia comandos
        self.assertIsNone(result)
        self.assertFalse(sm.obstacle_active)

    def test_obstacle_ignored_without_confidence(self):
        """Obstaculo sin confianza suficiente NO activa evasión."""
        sm = StateMachine()
        sm.start_continuous()

        result = sm.evaluate(
            detected_color=None,
            obstacle_distance=0.2,
            obstacle_direction="right",
            obstacle_zone="danger",
            obstacle_confidence=0.0,  # sin confirmacion
        )
        # No debe activar evasión
        self.assertNotEqual(result, "stop")
        self.assertFalse(sm.obstacle_active)

    def test_obstacle_turns_after_stop(self):
        """Despues del stop, el robot escanea 180° en vez de retroceder."""
        sm = StateMachine()
        sm.start_continuous()

        # Frame 1: obstaculo detectado → stop (fase IDLE → STOP)
        sm.evaluate(
            detected_color=None,
            obstacle_distance=0.3,
            obstacle_direction="left",
            obstacle_zone="danger",
            obstacle_confidence=1.0,
        )
        self.assertTrue(sm.obstacle_active)
        self.assertEqual(sm._obstacle_phase, "stop")

        # Frame 2: fase stop → transicion a scan_180 (retorna "stop")
        import time
        result = sm.evaluate(
            detected_color=None,
            obstacle_distance=0.3,
            obstacle_direction="left",
            obstacle_zone="danger",
            obstacle_confidence=1.0,
        )
        self.assertEqual(result, "stop")
        self.assertEqual(sm._obstacle_phase, "scan_180")

        # Frame 3: fase scan_180, elapsed < 0.7s → gira izquierda
        sm._obstacle_timer = time.time()  # simular que acaba de empezar scan_180
        result = sm.evaluate(
            detected_color=None,
            obstacle_distance=0.3,
            obstacle_direction="left",
            obstacle_zone="danger",
            obstacle_confidence=1.0,
        )
        self.assertEqual(result, "left")  # escanea izquierda primero

        # Frame 4: simular que paso tiempo de escaneo 180° (> 2 segundos)
        sm._obstacle_timer = time.time() - 2.5  # 2.5 segundos atras (> 2s scan_180)
        sm._obstacle_scan_results = [
            {"direction": "left", "distance": 1.5},  # espacio libre a la izquierda
            {"direction": "right", "distance": 0.2},  # bloqueado a la derecha
        ]
        result = sm.evaluate(
            detected_color=None,
            obstacle_distance=0.3,
            obstacle_direction="left",
            obstacle_zone="danger",
            obstacle_confidence=1.0,
        )
        self.assertEqual(result, "left")  # gira a la dirección con espacio libre

    def test_obstacle_scan_360_when_blocked(self):
        """Cuando no hay espacio en 180°, escanea 360°."""
        sm = StateMachine()
        sm.start_continuous()

        # Detectar obstaculo → stop
        sm.evaluate(
            detected_color=None,
            obstacle_distance=0.3,
            obstacle_direction="left",
            obstacle_zone="danger",
            obstacle_confidence=1.0,
        )

        # Frame 2: transicion a scan_180
        sm.evaluate(
            detected_color=None,
            obstacle_distance=0.3,
            obstacle_direction="left",
            obstacle_zone="danger",
            obstacle_confidence=1.0,
        )

        # Simular escaneo 180° sin espacio libre
        import time
        sm._obstacle_phase = "scan_180"
        sm._obstacle_timer = time.time() - 2.5
        sm._obstacle_scan_results = [
            {"direction": "left", "distance": 0.2},  # bloqueado
            {"direction": "right", "distance": 0.2},  # bloqueado
        ]

        result = sm.evaluate(
            detected_color=None,
            obstacle_distance=0.3,
            obstacle_direction="left",
            obstacle_zone="danger",
            obstacle_confidence=1.0,
        )
        # Debería cambiar a scan_360
        self.assertEqual(sm._obstacle_phase, "scan_360")

    def test_obstacle_resume_after_giro(self):
        """Despues del giro (tiempo max o despejado), reanuda el modo anterior."""
        sm = StateMachine()
        sm.start_continuous()

        # Frame 1: detectar obstaculo → stop (fase IDLE → STOP)
        sm.evaluate(
            detected_color=None,
            obstacle_distance=0.3,
            obstacle_direction="left",
            obstacle_zone="danger",
            obstacle_confidence=1.0,
        )
        self.assertTrue(sm.obstacle_active)
        self.assertEqual(sm._obstacle_phase, "stop")

        # Frame 2: fase stop → transicion a scan_180 (retorna "stop")
        import time
        result = sm.evaluate(
            detected_color=None,
            obstacle_distance=0.3,
            obstacle_direction="left",
            obstacle_zone="danger",
            obstacle_confidence=1.0,
        )
        self.assertEqual(result, "stop")
        self.assertEqual(sm._obstacle_phase, "scan_180")

        # Frame 3: scan_180, gira izquierda (elapsed=0 < 0.7)
        sm._obstacle_timer = time.time()
        result = sm.evaluate(
            detected_color=None,
            obstacle_distance=0.3,
            obstacle_direction="left",
            obstacle_zone="danger",
            obstacle_confidence=1.0,
        )
        self.assertEqual(result, "left")

        # Simular que paso el tiempo de escaneo (> 2s)
        sm._obstacle_timer = time.time() - 2.5
        sm._obstacle_scan_results = [
            {"direction": "left", "distance": 1.5},
        ]
        result = sm.evaluate(
            detected_color=None,
            obstacle_distance=0.3,
            obstacle_direction="left",
            obstacle_zone="danger",
            obstacle_confidence=1.0,
        )
        # Ahora esta en fase TURN, girando
        self.assertEqual(result, "left")

        # Simular que paso el tiempo del giro (> max turn time)
        sm._obstacle_timer = time.time() - 4.0  # 4 segundos atras (> OBSTACLE_MAX_TURN_TIME)

        # Evaluate: el giro ya completo por timeout
        result = sm.evaluate(
            detected_color=None,
            obstacle_distance=None,
        )
        self.assertFalse(sm.obstacle_active)

    def test_obstacle_resume_when_clear(self):
        """Cuando el obstaculo desaparece (distancia > threshold*factor), reanuda."""
        sm = StateMachine()
        sm.start_continuous()

        # Frame 1: detectar obstaculo → stop
        sm.evaluate(
            detected_color=None,
            obstacle_distance=0.3,
            obstacle_direction="left",
            obstacle_zone="danger",
            obstacle_confidence=1.0,
        )
        self.assertTrue(sm.obstacle_active)

        # Frame 2: fase stop → transicion a scan_180 (retorna "stop")
        import time
        result = sm.evaluate(
            detected_color=None,
            obstacle_distance=0.3,
            obstacle_direction="left",
            obstacle_zone="danger",
            obstacle_confidence=1.0,
        )
        self.assertEqual(result, "stop")
        self.assertEqual(sm._obstacle_phase, "scan_180")

        # Frame 3: scan_180, gira izquierda
        sm._obstacle_timer = time.time()
        result = sm.evaluate(
            detected_color=None,
            obstacle_distance=0.3,
            obstacle_direction="left",
            obstacle_zone="danger",
            obstacle_confidence=1.0,
        )
        self.assertEqual(result, "left")

        # Simular que paso el tiempo de escaneo (> 2s)
        sm._obstacle_timer = time.time() - 2.5
        sm._obstacle_scan_results = [
            {"direction": "left", "distance": 1.5},
        ]
        result = sm.evaluate(
            detected_color=None,
            obstacle_distance=0.3,
            obstacle_direction="left",
            obstacle_zone="danger",
            obstacle_confidence=1.0,
        )
        # Ahora esta en fase TURN
        self.assertEqual(result, "left")

        # Simular que el obstaculo desapareció (distancia grande)
        # Nota: el timer debe estar >= OBSTACLE_MIN_TURN_TIME (1.0s)
        sm._obstacle_timer = time.time() - 1.5  # despues del minimo de giro
        sm.evaluate(
            detected_color=None,
            obstacle_distance=2.0,  # muy lejos
            obstacle_zone="clear",
            obstacle_confidence=0.0,
        )
        self.assertFalse(sm.obstacle_active)

    def test_obstacle_directionProperty(self):
        """obstacle_direction retorna la direccion del giro."""
        sm = StateMachine()
        sm.start_continuous()

        sm.evaluate(
            detected_color=None,
            obstacle_distance=0.2,
            obstacle_direction="right",
            obstacle_zone="danger",
            obstacle_confidence=1.0,
        )
        self.assertEqual(sm.obstacle_direction, "right")

    def test_obstacle_activeProperty(self):
        """obstacle_active retorna True cuando esta esquivando."""
        sm = StateMachine()
        sm.start_continuous()

        self.assertFalse(sm.obstacle_active)

        sm.evaluate(
            detected_color=None,
            obstacle_distance=0.3,
            obstacle_direction="left",
            obstacle_zone="danger",
            obstacle_confidence=1.0,
        )
        self.assertTrue(sm.obstacle_active)

    def test_reset_clears_obstacle(self):
        """reset() limpia el estado de obstaculo."""
        sm = StateMachine()
        sm.start_continuous()

        sm.evaluate(
            detected_color=None,
            obstacle_distance=0.3,
            obstacle_direction="left",
            obstacle_zone="danger",
            obstacle_confidence=1.0,
        )
        self.assertTrue(sm.obstacle_active)

        sm.reset()
        self.assertFalse(sm.obstacle_active)
        self.assertIsNone(sm.obstacle_direction)

    def test_obstacle_never_backs_up(self):
        """El robot NUNCA retrocede — solo escanea y gira."""
        sm = StateMachine()
        sm.start_continuous()

        # Detectar obstaculo
        sm.evaluate(
            detected_color=None,
            obstacle_distance=0.3,
            obstacle_direction="left",
            obstacle_zone="danger",
            obstacle_confidence=1.0,
        )

        # Verificar que nunca envía "back"
        import time
        for _ in range(20):
            sm._obstacle_timer = time.time()
            result = sm.evaluate(
                detected_color=None,
                obstacle_distance=0.3,
                obstacle_direction="left",
                obstacle_zone="danger",
                obstacle_confidence=1.0,
            )
            self.assertNotEqual(result, "back", "El robot NUNCA debe retroceder")

    def test_obstacle_scan_180_phases(self):
        """Verificar las fases del escaneo 180°: stop → left → stop → right → stop."""
        sm = StateMachine()
        sm.start_continuous()

        # Frame 1: detectar obstaculo → stop
        sm.evaluate(
            detected_color=None,
            obstacle_distance=0.3,
            obstacle_direction="left",
            obstacle_zone="danger",
            obstacle_confidence=1.0,
        )

        import time

        # Frame 2: transicion a scan_180 (retorna "stop")
        result = sm.evaluate(
            detected_color=None,
            obstacle_distance=0.3,
            obstacle_direction="left",
            obstacle_zone="danger",
            obstacle_confidence=1.0,
        )
        self.assertEqual(result, "stop")
        self.assertEqual(sm._obstacle_phase, "scan_180")

        # Fase 1: girar izquierda (0 - 0.7s)
        sm._obstacle_timer = time.time()
        result = sm.evaluate(
            detected_color=None,
            obstacle_distance=0.3,
            obstacle_direction="left",
            obstacle_zone="danger",
            obstacle_confidence=1.0,
        )
        self.assertEqual(result, "left")

        # Fase 2: parar y analizar izquierda (0.7 - 1.0s)
        sm._obstacle_timer = time.time() - 0.8
        result = sm.evaluate(
            detected_color=None,
            obstacle_distance=1.5,  # espacio libre a la izquierda
            obstacle_direction="left",
            obstacle_zone="clear",
            obstacle_confidence=0.0,
        )
        self.assertEqual(result, "stop")

        # Fase 3: girar derecha (1.0 - 1.7s)
        sm._obstacle_timer = time.time() - 1.2
        result = sm.evaluate(
            detected_color=None,
            obstacle_distance=0.3,
            obstacle_direction="right",
            obstacle_zone="danger",
            obstacle_confidence=1.0,
        )
        self.assertEqual(result, "right")

    def test_robot_never_stops_completely(self):
        """En CONTINUOUS, el robot nunca se detiene completamente (excepto 1 frame)."""
        sm = StateMachine()
        sm.start_continuous()

        # Simular 10 frames con obstáculos
        stop_count = 0
        for _ in range(10):
            result = sm.evaluate(
                detected_color=None,
                obstacle_distance=0.5,
                obstacle_confidence=0.5,  # sin confianza suficiente
            )
            if result == "stop":
                stop_count += 1

        # El robot no debería estar en "stop" más de 1-2 frames
        # (solo en la fase STOP inicial de evitación)
        self.assertLessEqual(stop_count, 2, "El robot nunca debería parar completamente")


if __name__ == '__main__':
    unittest.main()
