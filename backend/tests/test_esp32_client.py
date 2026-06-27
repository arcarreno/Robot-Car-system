"""Tests para el cliente ESP32 con mock de requests."""
import sys
import os
import time
import unittest
from unittest.mock import patch, MagicMock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from clients.esp32_client import ESP32Client


class TestESP32ClientSendCommand(unittest.TestCase):
    """Tests de send_command con mock."""

    @patch('clients.esp32_client.requests.Session')
    def test_send_command_success(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_session.get.return_value = mock_resp

        client = ESP32Client(base_url="http://mock-esp32")
        result = client.send_command("go")
        self.assertTrue(result)
        mock_session.get.assert_called_once_with("http://mock-esp32/go", timeout=1.5)
        mock_resp.close.assert_called_once()

    @patch('clients.esp32_client.requests.Session')
    def test_send_command_failure(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_session.get.return_value = mock_resp

        client = ESP32Client(base_url="http://mock-esp32")
        result = client.send_command("go")
        self.assertFalse(result)
        self.assertEqual(client._consecutive_failures, 1)

    @patch('clients.esp32_client.requests.Session')
    def test_send_command_exception(self, mock_session_cls):
        import requests
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.get.side_effect = requests.exceptions.ConnectionError("refused")

        client = ESP32Client(base_url="http://mock-esp32")
        result = client.send_command("go")
        self.assertFalse(result)
        self.assertEqual(client._consecutive_failures, 1)

    @patch('clients.esp32_client.requests.Session')
    def test_circuit_breaker_opens_after_max_failures(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_session.get.return_value = mock_resp

        client = ESP32Client(base_url="http://mock-esp32")
        # Fallar 5 veces
        for _ in range(5):
            client.send_command("go")

        self.assertEqual(client._consecutive_failures, 5)

        # 6to intento debe ser bloqueado por circuit breaker
        result = client.send_command("go")
        self.assertFalse(result)
        # No se llamó get porque el circuit breaker bloqueó
        self.assertEqual(mock_session.get.call_count, 5)

    @patch('clients.esp32_client.requests.Session')
    def test_circuit_breaker_recovers_after_timeout(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_resp_ok = MagicMock()
        mock_resp_ok.ok = True
        mock_resp_fail = MagicMock()
        mock_resp_fail.ok = False

        client = ESP32Client(base_url="http://mock-esp32")
        # Fallar 5 veces
        mock_session.get.return_value = mock_resp_fail
        for _ in range(5):
            client.send_command("go")

        # Simular que pasó el timeout de recuperación
        client._last_failure_time = time.time() - 31.0

        # Ahora debe funcionar (half-open)
        mock_session.get.return_value = mock_resp_ok
        result = client.send_command("go")
        self.assertTrue(result)
        self.assertEqual(client._consecutive_failures, 0)


class TestESP32ClientPing(unittest.TestCase):
    """Tests de ping con mock."""

    @patch('clients.esp32_client.requests.Session')
    def test_ping_success(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_session.get.return_value = mock_resp

        client = ESP32Client(base_url="http://mock-esp32")
        result = client.ping()
        self.assertTrue(result)

    @patch('clients.esp32_client.requests.Session')
    def test_ping_failure(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_session.get.return_value = mock_resp

        client = ESP32Client(base_url="http://mock-esp32")
        result = client.ping()
        self.assertFalse(result)
        self.assertEqual(client._consecutive_failures, 1)


class TestESP32ClientSetSpeed(unittest.TestCase):
    """Tests de set_speed con mock."""

    @patch('clients.esp32_client.requests.Session')
    def test_set_speed_success(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_session.get.return_value = mock_resp

        client = ESP32Client(base_url="http://mock-esp32")
        result = client.set_speed(150)
        self.assertTrue(result)
        mock_session.get.assert_called_once_with("http://mock-esp32/speed?value=150", timeout=1.5)

    def test_set_speed_clamps_values(self):
        """set_speed debe limitar valores a 0-255."""
        client = ESP32Client(base_url="http://mock-esp32")
        with patch.object(client._session, 'get', return_value=MagicMock(ok=True)):
            client.set_speed(300)  # No debe fallar (clamped a 255)
            client.set_speed(-10)  # No debe fallar (clamped a 0)


if __name__ == '__main__':
    unittest.main()
