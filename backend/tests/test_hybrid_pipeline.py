"""Tests del pipeline híbrido YOLO+HSV en ColorDetector."""
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.color_detector import ColorDetector


def make_frame(color_bgr=(0, 0, 255), size=(64, 64)):
    """Crea un frame de color uniforme."""
    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    frame[:, :] = color_bgr
    return frame


class TestHybridPipeline(unittest.TestCase):
    """Tests del pipeline híbrido YOLO detecta → HSV lee color."""

    def setUp(self):
        self.yolo = MagicMock()
        self.yolo.is_available = True
        self.detector = ColorDetector(yolo_detector=self.yolo)

    def test_yolo_detects_semaforo_reads_hsv(self):
        """YOLO detecta semaforo, HSV lee verde dentro del bbox."""
        self.yolo.detect.return_value = {
            "semaforo": {"x1": 10, "y1": 10, "x2": 50, "y2": 50,
                         "confidence": 0.9, "class": "semaforo"},
            "arrows": [],
        }
        self.yolo.get_arrow_direction.return_value = None

        # Frame verde
        frame = make_frame(color_bgr=(0, 200, 0))
        # Ejecutar 3 veces para superar hysteresis
        for _ in range(3):
            result = self.detector.detect(frame)

        self.assertEqual(result["detected"], "green")
        self.yolo.detect.assert_called()

    def test_yolo_detects_red(self):
        """YOLO + HSV detecta rojo."""
        self.yolo.detect.return_value = {
            "semaforo": {"x1": 10, "y1": 10, "x2": 50, "y2": 50,
                         "confidence": 0.9, "class": "semaforo"},
            "arrows": [],
        }
        self.yolo.get_arrow_direction.return_value = None

        frame = make_frame(color_bgr=(0, 0, 255))
        for _ in range(3):
            result = self.detector.detect(frame)

        self.assertEqual(result["detected"], "red")

    def test_yolo_detects_yellow(self):
        """YOLO + HSV detecta amarillo."""
        self.yolo.detect.return_value = {
            "semaforo": {"x1": 10, "y1": 10, "x2": 50, "y2": 50,
                         "confidence": 0.9, "class": "semaforo"},
            "arrows": [],
        }
        self.yolo.get_arrow_direction.return_value = None

        frame = make_frame(color_bgr=(0, 255, 255))
        for _ in range(3):
            result = self.detector.detect(frame)

        self.assertEqual(result["detected"], "yellow")

    def test_yolo_not_available_falls_back_to_hsv(self):
        """Sin YOLO: fallback a HSV puro."""
        self.yolo.is_available = False
        detector = ColorDetector(yolo_detector=self.yolo)

        frame = make_frame(color_bgr=(0, 200, 0))
        for _ in range(3):
            result = detector.detect(frame)

        self.assertEqual(result["detected"], "green")
        # YOLO no fue llamado
        self.yolo.detect.assert_not_called()

    def test_no_semaforo_no_fallback(self):
        """YOLO no detecta semaforo → retorna None (sin fallback HSV completo)."""
        self.yolo.detect.return_value = {
            "semaforo": None,
            "arrows": [],
        }
        self.yolo.get_arrow_direction.return_value = None

        frame = make_frame(color_bgr=(0, 200, 0))
        for _ in range(3):
            result = self.detector.detect(frame)

        self.assertIsNone(result["detected"])

    def test_yolo_skip_frame_passes_flag(self):
        """YOLO skip: detect() pasa run_yolo=False a YoloDetector."""
        self.yolo.detect.return_value = {
            "semaforo": {"x1": 10, "y1": 10, "x2": 50, "y2": 50,
                         "confidence": 0.9, "class": "semaforo"},
            "arrows": [],
        }
        self.yolo.get_arrow_direction.return_value = None

        frame = make_frame(color_bgr=(0, 200, 0))
        # Primera deteccion (run_yolo=True)
        self.detector.detect(frame, run_yolo=True)
        self.yolo.detect.reset_mock()

        # Skip frame (run_yolo=False) - YOLO es llamado pero con flag False
        result = self.detector.detect(frame, run_yolo=False)
        self.yolo.detect.assert_called_once()
        call_args = self.yolo.detect.call_args
        self.assertFalse(call_args.kwargs.get("run_yolo", True))

    def test_overlay_reused_across_frames(self):
        """Overlay se reutiliza entre frames (no crea np.zeros cada vez)."""
        self.yolo.detect.return_value = {
            "semaforo": {"x1": 10, "y1": 10, "x2": 50, "y2": 50,
                         "confidence": 0.9, "class": "semaforo"},
            "arrows": [],
        }
        self.yolo.get_arrow_direction.return_value = None

        frame = make_frame(color_bgr=(0, 200, 0))
        self.detector.detect(frame)
        overlay1 = self.detector._overlay
        self.detector.detect(frame)
        overlay2 = self.detector._overlay
        # Misma referencia (reusado, no recreado)
        self.assertIs(overlay1, overlay2)


class TestArrowDetection(unittest.TestCase):
    """Tests de detección de flechas con YOLO."""

    def setUp(self):
        self.yolo = MagicMock()
        self.yolo.is_available = True
        self.detector = ColorDetector(yolo_detector=self.yolo)

    def test_arrow_right_detected(self):
        """YOLO detecta flecha arrow_right."""
        self.yolo.detect.return_value = {
            "semaforo": None,
            "arrows": [{"x1": 100, "y1": 50, "x2": 200, "y2": 150,
                        "confidence": 0.85, "class": "arrow_right"}],
        }
        self.yolo.get_arrow_direction.return_value = "right"

        frame = make_frame(color_bgr=(0, 0, 0))  # negro = sin color
        result = self.detector.detect(frame)

        self.assertEqual(result["arrow"], "right")

    def test_arrow_left_detected(self):
        """YOLO detecta flecha arrow_left."""
        self.yolo.detect.return_value = {
            "semaforo": None,
            "arrows": [{"x1": 100, "y1": 50, "x2": 200, "y2": 150,
                        "confidence": 0.85, "class": "arrow_left"}],
        }
        self.yolo.get_arrow_direction.return_value = "left"

        frame = make_frame(color_bgr=(0, 0, 0))
        result = self.detector.detect(frame)

        self.assertEqual(result["arrow"], "left")

    def test_no_arrow_returns_none(self):
        """Sin flecha detectada retorna None."""
        self.yolo.detect.return_value = {
            "semaforo": None,
            "arrows": [],
        }
        self.yolo.get_arrow_direction.return_value = None

        frame = make_frame(color_bgr=(0, 0, 0))
        result = self.detector.detect(frame)

        self.assertIsNone(result["arrow"])


class TestFrameDimensions(unittest.TestCase):
    """Tests de cambio de dimensiones del frame."""

    def setUp(self):
        self.yolo = MagicMock()
        self.yolo.is_available = True
        self.detector = ColorDetector(yolo_detector=self.yolo)
        self.yolo.detect.return_value = {"semaforo": None, "arrows": []}
        self.yolo.get_arrow_direction.return_value = None

    def test_overlay_resized_on_dimension_change(self):
        """Overlay se redimensiona cuando cambian las dims del frame."""
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        self.detector.detect(frame1)
        self.assertEqual(self.detector._overlay.shape, (480, 640, 3))

        frame2 = np.zeros((240, 320, 3), dtype=np.uint8)
        self.detector.detect(frame2)
        self.assertEqual(self.detector._overlay.shape, (240, 320, 3))


if __name__ == '__main__':
    unittest.main()
