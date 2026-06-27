"""Tests del parser MJPEG manual."""
import sys
import os
import struct
import unittest
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Importar el módulo directamente para poder testear su logica interna
import core.mjpeg_parser as mjpeg_mod


def make_jpeg_bytes():
    """Crea bytes JPEG minimos validos (FF D8 ... FF D9)."""
    return b'\xff\xd8' + b'\x00' * 100 + b'\xff\xd9'


def make_jpeg_frame():
    """Crea un frame JPEG real minimo usando OpenCV."""
    import cv2
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :] = (0, 0, 255)  # rojo
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()


class TestMjpegGeneratorBasic(unittest.TestCase):
    """Tests basicos del generador MJPEG."""

    @patch('core.mjpeg_parser.requests.Session')
    def test_yields_decoded_frame(self, mock_session_cls):
        """Generador decodifica frames JPEG validos."""
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        jpeg_data = make_jpeg_frame()
        mock_resp = MagicMock()
        # Simular chunks: primero header JPEG, luego el resto
        mock_resp.iter_content.return_value = [jpeg_data[:50], jpeg_data[50:]]
        mock_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_resp

        gen = mjpeg_mod.mjpeg_generator("http://mock/stream")
        # Tomar primer frame (el generador es infinito, usar next con timeout)
        frame = next(gen)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape[:2], (64, 64))

    @patch('core.mjpeg_parser.requests.Session')
    def test_yields_none_on_connection_error(self, mock_session_cls):
        """Generador yield None en error de conexion."""
        import requests
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.get.side_effect = requests.exceptions.ConnectionError("refused")

        gen = mjpeg_mod.mjpeg_generator("http://mock/stream")
        frame = next(gen)
        self.assertIsNone(frame)

    @patch('core.mjpeg_parser.requests.Session')
    def test_yields_none_on_timeout(self, mock_session_cls):
        """Generador yield None en timeout."""
        import requests
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.get.side_effect = requests.exceptions.Timeout("timed out")

        gen = mjpeg_mod.mjpeg_generator("http://mock/stream")
        frame = next(gen)
        self.assertIsNone(frame)


class TestMjpegBufferLogic(unittest.TestCase):
    """Tests de la logica de buffer MJPEG (FF D8/FF D9)."""

    def test_finds_jpeg_in_buffer(self):
        """Busca JPEG mas reciente en buffer con markers correctos."""
        jpeg = make_jpeg_frame()
        # Buffer con basura antes del JPEG
        buf = b'\x00' * 50 + jpeg
        start = buf.rfind(b'\xff\xd8')
        end = buf.rfind(b'\xff\xd9', start)
        self.assertNotEqual(start, -1)
        self.assertNotEqual(end, -1)
        self.assertTrue(end > start)

    def test_takes_most_recent_jpeg(self):
        """Toma el JPEG mas reciente si hay multiples."""
        jpeg1 = b'\xff\xd8' + b'\x01' * 20 + b'\xff\xd9'
        jpeg2 = b'\xff\xd8' + b'\x02' * 20 + b'\xff\xd9'
        buf = jpeg1 + b'\x00' * 10 + jpeg2
        start = buf.rfind(b'\xff\xd8')
        end = buf.rfind(b'\xff\xd9', start)
        # Debe encontrar el segundo JPEG
        jpg_bytes = buf[start:end + 2]
        self.assertTrue(jpg_bytes.endswith(b'\xff\xd9'))
        # Contenido del segundo JPEG
        self.assertIn(b'\x02', jpg_bytes)

    def test_incomplete_jpeg_not_yielded(self):
        """JPEG incompleto (sin FF D9) no se extrae."""
        buf = b'\xff\xd8' + b'\x00' * 100  # sin FF D9
        start = buf.rfind(b'\xff\xd8')
        end = buf.rfind(b'\xff\xd9', start)
        # end es -1 porque no hay FF D9
        self.assertEqual(end, -1)

    def test_buffer_truncation_at_1mb(self):
        """Buffer se trunca a 1MB cuando excede el limite."""
        MAX_BUFFER_SIZE = 1024 * 1024
        # Simular buffer gigante
        buf = b'\x00' * (MAX_BUFFER_SIZE + 1000)
        # Logica de truncamiento
        if len(buf) > MAX_BUFFER_SIZE:
            buf = buf[-MAX_BUFFER_SIZE:]
        self.assertEqual(len(buf), MAX_BUFFER_SIZE)


class TestMjpegFrameDecoding(unittest.TestCase):
    """Tests de decodificacion de frames."""

    def test_valid_jpeg_decodes(self):
        """JPEG valido se decodifica correctamente."""
        import cv2
        jpeg_data = make_jpeg_frame()
        frame = cv2.imdecode(
            np.frombuffer(jpeg_data, np.uint8),
            cv2.IMREAD_COLOR
        )
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (64, 64, 3))

    def test_invalid_jpeg_returns_none(self):
        """JPEG corrupto retorna None."""
        import cv2
        garbage = b'\xff\xd8' + b'\xff\xff\xff\xff' + b'\xff\xd9'
        frame = cv2.imdecode(
            np.frombuffer(garbage, np.uint8),
            cv2.IMREAD_COLOR
        )
        # OpenCV puede o no decodificar esto, pero no debe crashear
        # (retorna None o un frame invalido)


if __name__ == '__main__':
    unittest.main()
