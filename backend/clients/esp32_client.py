"""
Cliente HTTP para comandar el ESP32.

Proporciona metodos simples para enviar comandos y consultar estado.
Circuit breaker con recuperacion: despues de 30s sin fallas, reintenta.
"""

import time
import threading
import requests
from typing import Optional
from config import ESP32_API


class ESP32Client:
    """
    Interfaz de comunicacion con el ESP32.

    Todos los comandos se envian como GET requests.
    Timeout corto (3s) para no bloquear el pipeline.
    """

    def __init__(self, base_url: str = ESP32_API):
        self.base_url = base_url
        self._session = requests.Session()
        self._timeout = 1.5  # reducido de 3s — ESP32 responde en <500ms normalmente
        self._lock = threading.Lock()
        self._session_lock = threading.Lock()  # proteger session并发 access
        self._consecutive_failures = 0
        self._max_failures = 5
        self._last_failure_time = 0.0
        self._recovery_timeout = 5.0  # reducido de 30s — ESP32 se recupera rapido

    def send_command(self, command: str) -> bool:
        """
        Envia un comando al ESP32.

        Args:
            command: "go" | "back" | "stop" | "left" | "right" |
                     "ledon" | "ledoff"

        Returns:
            True si el ESP32 respondio OK, False en otro caso.
        """
        # A5 fix: check del circuit breaker ANTES de tomar el lock,
        # y request HTTP FUERA del lock. Antes: el lock se sostenia
        # durante todo el round-trip HTTP (hasta 3s con timeout),
        # bloqueando a otros threads que quisieran enviar comandos.
        # Ahora: solo tomamos el lock para leer/escriturar el estado
        # del circuit breaker, y el HTTP corre sin lock.
        is_half_open = False
        with self._lock:
            if self._consecutive_failures >= self._max_failures:
                if time.time() - self._last_failure_time < self._recovery_timeout:
                    print(f"[ESP32] Circuit breaker open -- skipping '{command}'")
                    return False
                else:
                    print(f"[ESP32] Circuit breaker half-open -- retrying '{command}'")
                    is_half_open = True

        url = f"{self.base_url}/{command}"
        try:
            with self._session_lock:
                resp = self._session.get(url, timeout=self._timeout)
            ok = resp.ok
            resp.close()
        except requests.exceptions.RequestException as e:
            print(f"[ESP32] Error enviando '{command}': {e}")
            ok = False

        # Actualizar estado del circuit breaker
        with self._lock:
            if ok:
                self._consecutive_failures = 0
            else:
                if is_half_open:
                    # Half-open retry failed: volver a abrir circuit breaker
                    self._consecutive_failures = self._max_failures
                else:
                    self._consecutive_failures += 1
                self._last_failure_time = time.time()

        return ok

    def ping(self) -> bool:
        """Verifica si el ESP32 responde (GET /status rapido)."""
        is_half_open = False
        with self._lock:
            if self._consecutive_failures >= self._max_failures:
                if time.time() - self._last_failure_time < self._recovery_timeout:
                    return False
                is_half_open = True

        try:
            with self._session_lock:
                resp = self._session.get(
                    f"{self.base_url}/status",
                    timeout=2
                )
            ok = resp.ok
            resp.close()
        except requests.exceptions.RequestException:
            ok = False

        with self._lock:
            if ok:
                self._consecutive_failures = 0
            else:
                if is_half_open:
                    self._consecutive_failures = self._max_failures
                else:
                    self._consecutive_failures += 1
                self._last_failure_time = time.time()

        return ok

    def set_speed(self, value: int) -> bool:
        """Cambia la velocidad del robot (0-255).

        Tambien resetea el circuit breaker si tiene exito, igual que send_command.
        """
        value = max(0, min(255, value))
        is_half_open = False
        with self._lock:
            if self._consecutive_failures >= self._max_failures:
                if time.time() - self._last_failure_time < self._recovery_timeout:
                    return False
                is_half_open = True

        url = f"{self.base_url}/speed?value={value}"
        try:
            with self._session_lock:
                resp = self._session.get(url, timeout=self._timeout)
            ok = resp.ok
            resp.close()
        except requests.exceptions.RequestException:
            ok = False

        with self._lock:
            if ok:
                self._consecutive_failures = 0
                self._last_failure_time = 0.0  # reset recovery timer
            else:
                if is_half_open:
                    self._consecutive_failures = self._max_failures
                else:
                    self._consecutive_failures += 1
                self._last_failure_time = time.time()

        return ok

    def reset_circuit_breaker(self):
        """Force-reset circuit breaker (call on mode change / user stop)."""
        with self._lock:
            self._consecutive_failures = 0
            self._last_failure_time = 0.0
            print("[ESP32] Circuit breaker reset")

    def close(self):
        """Cierra la sesion HTTP."""
        self._session.close()
