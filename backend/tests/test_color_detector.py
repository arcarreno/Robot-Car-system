"""Tests para el detector de colores HSV."""
import sys
import os
import unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import cv2
from core.color_detector import ColorDetector


def create_color_frame(h, w, color_bgr):
    """Crea un frame de color solido."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = color_bgr
    return frame


def detect_stable(detector, frame, n=3):
    """B1 helper: llama detect() n veces para construir historial y confirmar color."""
    result = None
    for _ in range(n):
        result = detector.detect(frame)
    return result


class TestColorDetectorBasics(unittest.TestCase):
    """Tests basicos del detector."""

    def test_none_frame_returns_none(self):
        detector = ColorDetector()
        result = detector.detect(None)
        self.assertIsNone(result["detected"])
        self.assertIsNone(result["arrow"])

    def test_black_frame_returns_none(self):
        detector = ColorDetector()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = detector.detect(frame)
        self.assertIsNone(result["detected"])

    def test_result_has_required_keys(self):
        detector = ColorDetector()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = detector.detect(frame)
        self.assertIn("detected", result)
        self.assertIn("arrow", result)
        self.assertIn("colors", result)
        self.assertIn("overlay", result)


class TestColorDetection(unittest.TestCase):
    """Tests de deteccion de colores especificos."""

    def test_pure_red_detected(self):
        detector = ColorDetector()
        frame = create_color_frame(240, 320, (0, 0, 255))
        result = detect_stable(detector, frame)
        self.assertEqual(result["detected"], "red")

    def test_pure_green_detected(self):
        detector = ColorDetector()
        frame = create_color_frame(240, 320, (0, 255, 0))
        result = detect_stable(detector, frame)
        self.assertEqual(result["detected"], "green")

    def test_pure_yellow_detected(self):
        detector = ColorDetector()
        frame = create_color_frame(240, 320, (0, 255, 255))
        result = detect_stable(detector, frame)
        self.assertEqual(result["detected"], "yellow")

    def test_red_has_priority_over_green(self):
        detector = ColorDetector()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.circle(frame, (80, 120), 40, (0, 0, 255), -1)
        cv2.circle(frame, (240, 120), 40, (0, 255, 0), -1)
        result = detect_stable(detector, frame)
        self.assertEqual(result["detected"], "red")


class TestArrowDetection(unittest.TestCase):
    """Tests de deteccion de flechas por posicion."""

    def test_green_pixels_outside_circle_detected_as_arrow_left(self):
        """Circulo verde + pixeles verdes a la izquierda -> arrow_left."""
        detector = ColorDetector()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.circle(frame, (220, 120), 40, (94, 197, 34), -1)
        cv2.circle(frame, (50, 120), 20, (94, 197, 34), -1)
        result = detect_stable(detector, frame)
        self.assertEqual(result["detected"], "green")
        self.assertEqual(result["arrow"], "left")

    def test_arrow_right(self):
        """Circulo verde + pixeles verdes a la derecha -> arrow_right."""
        detector = ColorDetector()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.circle(frame, (100, 120), 40, (94, 197, 34), -1)
        cv2.circle(frame, (270, 120), 20, (94, 197, 34), -1)
        result = detect_stable(detector, frame)
        self.assertEqual(result["detected"], "green")
        self.assertEqual(result["arrow"], "right")

    def test_single_green_no_arrow(self):
        """Un solo circulo verde no debe detectar flecha."""
        detector = ColorDetector()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.circle(frame, (160, 120), 40, (94, 197, 34), -1)
        result = detect_stable(detector, frame)
        self.assertEqual(result["detected"], "green")
        self.assertIsNone(result["arrow"])

    def test_darker_green_still_detected(self):
        """Verde oscuro (V bajo) tambien se detecta como verde."""
        detector = ColorDetector()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.circle(frame, (160, 120), 40, (61, 128, 21), -1)
        result = detect_stable(detector, frame)
        self.assertEqual(result["detected"], "green")


class TestColorHysteresis(unittest.TestCase):
    """B1 tests: histéresis de color."""

    def test_single_frame_not_confirmed(self):
        """Un solo frame con color NO confirma (necesita 3+ frames)."""
        detector = ColorDetector()
        frame = create_color_frame(240, 320, (0, 0, 255))
        result = detector.detect(frame)
        self.assertIsNone(result["detected"])

    def test_three_frames_confirms_color(self):
        """3 frames con el mismo color SÍ confirma."""
        detector = ColorDetector()
        frame = create_color_frame(240, 320, (0, 0, 255))
        result = detect_stable(detector, frame, n=3)
        self.assertEqual(result["detected"], "red")

    def test_intermittent_color_not_confirmed(self):
        """Color que aparece 2 de 5 veces NO confirma."""
        detector = ColorDetector()
        red = create_color_frame(240, 320, (0, 0, 255))
        black = np.zeros((240, 320, 3), dtype=np.uint8)
        detector.detect(red)
        detector.detect(black)
        detector.detect(red)
        detector.detect(black)
        result = detector.detect(black)
        self.assertIsNone(result["detected"])


if __name__ == '__main__':
    unittest.main()
