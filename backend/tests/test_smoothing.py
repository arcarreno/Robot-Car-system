"""Tests de suavizado de bbox, decaimiento y arrow smoothing."""
import sys
import os
import unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.color_detector import ColorDetector


class TestSmoothBbox(unittest.TestCase):
    """Tests de _smooth_bbox."""

    def setUp(self):
        self.detector = ColorDetector()

    def test_first_detection_no_previous(self):
        """Primera detección sin prev: retorna new tal cual."""
        new = {"x1": 100, "y1": 200, "x2": 300, "y2": 400,
               "confidence": 0.9, "class": "semaforo"}
        result = self.detector._smooth_bbox(None, new)
        self.assertEqual(result["x1"], 100)
        self.assertEqual(result["y1"], 200)
        self.assertEqual(result["x2"], 300)
        self.assertEqual(result["y2"], 400)
        self.assertEqual(result["confidence"], 0.9)

    def test_smoothing_blends_coordinates(self):
        """Suavizado promedia coords con alpha=0.45."""
        prev = {"x1": 100, "y1": 100, "x2": 200, "y2": 200,
                "confidence": 0.8, "class": "semaforo"}
        new = {"x1": 200, "y1": 200, "x2": 300, "y2": 300,
               "confidence": 0.9, "class": "semaforo"}
        result = self.detector._smooth_bbox(prev, new)
        a = self.detector.BBOX_SMOOTH_ALPHA  # 0.45
        # x1 = 0.45*200 + 0.55*100 = 90 + 55 = 145
        self.assertEqual(result["x1"], int(round(a * 200 + (1 - a) * 100)))
        self.assertEqual(result["y1"], int(round(a * 200 + (1 - a) * 100)))

    def test_smoothing_preserves_confidence_and_class(self):
        """Suavizado usa confidence y class de la nueva detección."""
        prev = {"x1": 100, "y1": 100, "x2": 200, "y2": 200,
                "confidence": 0.5, "class": "semaforo"}
        new = {"x1": 200, "y1": 200, "x2": 300, "y2": 300,
               "confidence": 0.95, "class": "arrow_right"}
        result = self.detector._smooth_bbox(prev, new)
        self.assertEqual(result["confidence"], 0.95)
        self.assertEqual(result["class"], "arrow_right")

    def test_smoothing_returns_ints(self):
        """Suavizado retorna enteros (no floats) para OpenCV."""
        prev = {"x1": 101, "y1": 101, "x2": 201, "y2": 201,
                "confidence": 0.8, "class": "semaforo"}
        new = {"x1": 102, "y1": 102, "x2": 202, "y2": 202,
               "confidence": 0.9, "class": "semaforo"}
        result = self.detector._smooth_bbox(prev, new)
        for key in ["x1", "y1", "x2", "y2"]:
            self.assertIsInstance(result[key], int, f"{key} debe ser int")


class TestDecaySmooth(unittest.TestCase):
    """Tests de _decay_smooth."""

    def setUp(self):
        self.detector = ColorDetector()
        # Simular frame de 640x480
        self.detector._last_dims = (480, 640)

    def test_decay_none_returns_none(self):
        """Si prev es None, decay retorna None."""
        result = self.detector._decay_smooth(None)
        self.assertIsNone(result)

    def test_decay_moves_toward_center(self):
        """Decay mueve bbox hacia el centro del frame."""
        # bbox lejos del centro (320, 240)
        prev = {"x1": 50, "y1": 50, "x2": 150, "y2": 150,
                "confidence": 0.8, "class": "semaforo"}
        result = self.detector._decay_smooth(prev)
        self.assertIsNotNone(result)
        # Centro del bbox resultante debe estar más cerca del centro del frame
        prev_cx = (50 + 150) // 2  # 100
        result_cx = (result["x1"] + result["x2"]) // 2
        self.assertGreater(result_cx, prev_cx)  # se movió a la derecha

    def test_decay_returns_none_when_at_center(self):
        """Decay retorna None si bbox ya está en el centro."""
        # bbox centrado en (320, 240) ±1
        prev = {"x1": 318, "y1": 238, "x2": 322, "y2": 242,
                "confidence": 0.8, "class": "semaforo"}
        result = self.detector._decay_smooth(prev)
        self.assertIsNone(result)

    def test_decay_preserves_size(self):
        """Decay mantiene el tamaño del bbox (solo mueve centro)."""
        prev = {"x1": 50, "y1": 50, "x2": 150, "y2": 200,
                "confidence": 0.8, "class": "semaforo"}
        result = self.detector._decay_smooth(prev)
        prev_w = 150 - 50
        prev_h = 200 - 50
        result_w = result["x2"] - result["x1"]
        result_h = result["y2"] - result["y1"]
        self.assertEqual(result_w, prev_w)
        self.assertEqual(result_h, prev_h)


class TestSmoothArrowList(unittest.TestCase):
    """Tests de _smooth_arrow_list."""

    def setUp(self):
        self.detector = ColorDetector()

    def test_new_arrow_appears(self):
        """Flecha nueva sin prev: se agrega con coords de new."""
        new = [{"x1": 100, "y1": 200, "x2": 300, "y2": 400,
                "confidence": 0.9, "class": "arrow_right"}]
        result = self.detector._smooth_arrow_list([], new)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["class"], "arrow_right")

    def test_existing_arrow_smoothed(self):
        """Flecha existente se suaviza."""
        prev = [{"x1": 100, "y1": 100, "x2": 200, "y2": 200,
                 "confidence": 0.8, "class": "arrow_right"}]
        new = [{"x1": 150, "y1": 150, "x2": 250, "y2": 250,
                "confidence": 0.9, "class": "arrow_right"}]
        result = self.detector._smooth_arrow_list(prev, new)
        self.assertEqual(len(result), 1)
        # Coords suavizadas
        a = self.detector.BBOX_SMOOTH_ALPHA
        expected_x1 = int(round(a * 150 + (1 - a) * 100))
        self.assertEqual(result[0]["x1"], expected_x1)

    def test_disappearing_arrow_removed(self):
        """Flecha que desaparece: se elimina (sin decay)."""
        prev = [{"x1": 100, "y1": 100, "x2": 200, "y2": 200,
                 "confidence": 0.8, "class": "arrow_right"}]
        result = self.detector._smooth_arrow_list(prev, [])
        self.assertEqual(len(result), 0)

    def test_different_classes_not_mixed(self):
        """Flechas de distinta clase no se mezclan."""
        prev = [{"x1": 100, "y1": 100, "x2": 200, "y2": 200,
                 "confidence": 0.8, "class": "arrow_right"}]
        new = [{"x1": 300, "y1": 300, "x2": 400, "y2": 400,
                "confidence": 0.9, "class": "arrow_left"}]
        result = self.detector._smooth_arrow_list(prev, new)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["class"], "arrow_left")
        # x1 es la de new, no de prev
        self.assertEqual(result[0]["x1"], 300)


if __name__ == '__main__':
    unittest.main()
