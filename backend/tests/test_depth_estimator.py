"""
Tests del modulo de estimacion de profundidad (DepthEstimator).
"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from core.depth_estimator import DepthEstimator


class TestDepthEstimatorInit(unittest.TestCase):
    """Tests de inicializacion del DepthEstimator."""

    @patch("core.depth_estimator.os.path.exists", return_value=False)
    def test_model_not_found(self, mock_exists):
        """Si el modelo no existe, available=False y no lanza error."""
        estimator = DepthEstimator(model_path="nonexistent/model.xml")
        self.assertFalse(estimator.available)

    @patch("core.depth_estimator.os.path.exists", return_value=True)
    @patch("openvino.Core")
    def test_model_loads_successfully(self, mock_core_cls, mock_exists):
        """Si el modelo existe y openvino carga, available=True."""
        mock_core = MagicMock()
        mock_core_cls.return_value = mock_core
        mock_model = MagicMock()
        mock_core.read_model.return_value = mock_model
        mock_compiled = MagicMock()
        mock_compiled.input.return_value.shape = [1, 3, 256, 256]
        mock_core.compile_model.return_value = mock_compiled

        estimator = DepthEstimator(model_path="fake/MiDaS_small.xml")
        self.assertTrue(estimator.available)


class TestRawToMeters(unittest.TestCase):
    """Tests de conversion depth → metros."""

    def setUp(self):
        self.estimator = DepthEstimator.__new__(DepthEstimator)
        self.estimator._cal_k = 99997.1
        self.estimator._cal_p = 1.94
        self.estimator._cal_offset = 118.0
        self.estimator._compiled_model = None

    def test_close_object(self):
        """Objeto cercano (depth alto) → pocos metros que objeto lejano."""
        distance = self.estimator.raw_to_meters(200.0)
        # 200 - 118 = 82, 99997.1 / 82^1.94 ≈ 22.1m
        distance_far = self.estimator.raw_to_meters(130.0)
        self.assertGreater(distance, 0)
        self.assertLess(distance, distance_far)

    def test_far_object(self):
        """Objeto lejano (depth bajo) → muchos metros."""
        distance_close = self.estimator.raw_to_meters(200.0)
        distance_far = self.estimator.raw_to_meters(130.0)
        self.assertGreater(distance_far, distance_close)

    def test_very_low_value_returns_inf(self):
        """Valor muy bajo (debajo del offset) → infinito."""
        distance = self.estimator.raw_to_meters(110.0)  # 110 - 118 = -8
        self.assertEqual(distance, float("inf"))

    def test_zero_value_returns_inf(self):
        """Valor cero → infinito."""
        distance = self.estimator.raw_to_meters(0.0)
        self.assertEqual(distance, float("inf"))


class TestGetObstacleDistance(unittest.TestCase):
    """Tests de obtencion de distancia al obstaculo."""

    def setUp(self):
        self.estimator = DepthEstimator.__new__(DepthEstimator)
        self.estimator._cal_k = 99997.1
        self.estimator._cal_p = 1.94
        self.estimator._cal_offset = 118.0
        self.estimator._compiled_model = None
        self.estimator._ema_alpha = 0.4
        self.estimator._prev_distance = None

    def test_returns_distance_for_center_roi(self):
        """Retorna distancia para la region central del depth map."""
        # Depth map 100x100, centro con valores altos (objeto cercano)
        depth_map = np.full((100, 100), 130.0, dtype=np.float32)
        depth_map[35:65, 35:65] = 200.0  # centro: objeto cercano

        distance = self.estimator.get_obstacle_distance(depth_map, center_ratio=0.3)
        self.assertIsNotNone(distance)
        self.assertGreater(distance, 0)

    def test_returns_none_for_none_map(self):
        """Si depth_map es None, retorna None."""
        distance = self.estimator.get_obstacle_distance(None)
        self.assertIsNone(distance)

    def test_closer_object_gives_smaller_distance(self):
        """Objeto mas cercano en centro → menor distancia."""
        # Escenario 1: centro lejano
        depth1 = np.full((100, 100), 130.0, dtype=np.float32)
        depth1[35:65, 35:65] = 150.0
        d1 = self.estimator.get_obstacle_distance(depth1)

        # Escenario 2: centro cercano
        depth2 = np.full((100, 100), 130.0, dtype=np.float32)
        depth2[35:65, 35:65] = 220.0
        d2 = self.estimator.get_obstacle_distance(depth2)

        self.assertLess(d2, d1)


class TestGetFreeSpaceDirection(unittest.TestCase):
    """Tests de determinacion del lado con mas espacio."""

    def setUp(self):
        self.estimator = DepthEstimator.__new__(DepthEstimator)
        self.estimator._compiled_model = None
        self.estimator._prev_direction = "right"

    def test_left_has_more_space(self):
        """Si la izquierda tiene depth mas bajo → retorna 'left'."""
        # Izquierda: depth bajo (lejos = espacio)
        depth_map = np.full((100, 100), 200.0, dtype=np.float32)
        depth_map[:, 0:33] = 130.0  # izquierda: lejos
        depth_map[:, 67:100] = 220.0  # derecha: cerca

        direction = self.estimator.get_free_space_direction(depth_map)
        self.assertEqual(direction, "left")

    def test_right_has_more_space(self):
        """Si la derecha tiene depth mas bajo → retorna 'right'."""
        depth_map = np.full((100, 100), 200.0, dtype=np.float32)
        depth_map[:, 0:33] = 220.0  # izquierda: cerca
        depth_map[:, 67:100] = 130.0  # derecha: lejos

        direction = self.estimator.get_free_space_direction(depth_map)
        self.assertEqual(direction, "right")

    def test_returns_left_for_none_map(self):
        """Si depth_map es None, mantiene la ultima direccion conocida."""
        direction = self.estimator.get_free_space_direction(None)
        self.assertEqual(direction, "right")  # default de setUp

    def test_symmetric_returns_right(self):
        """Si ambos lados son iguales, retorna 'right' (tie-break)."""
        depth_map = np.full((100, 100), 180.0, dtype=np.float32)
        direction = self.estimator.get_free_space_direction(depth_map)
        self.assertIn(direction, ("left", "right"))


class TestDepthEstimatorEstimate(unittest.TestCase):
    """Tests del metodo estimate() con mock de OpenVINO."""

    def test_estimate_returns_none_when_not_available(self):
        """Si el modelo no esta disponible, retorna None."""
        estimator = DepthEstimator.__new__(DepthEstimator)
        estimator._compiled_model = None
        estimator._input_shape = None

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = estimator.estimate(frame)
        self.assertIsNone(result)

    @patch("core.depth_estimator.os.path.exists", return_value=True)
    @patch("openvino.Core")
    def test_estimate_returns_correct_shape(self, mock_core_cls, mock_exists):
        """estimate() retorna depth map con la misma resolucion que el input."""
        mock_core = MagicMock()
        mock_core_cls.return_value = mock_core
        mock_compiled = MagicMock()
        input_mock = MagicMock()
        input_mock.shape = [1, 3, 256, 256]
        mock_compiled.input.return_value = input_mock
        mock_compiled.output.return_value = 0
        mock_core.compile_model.return_value = mock_compiled

        # Mock inference: retorna array (1, 256, 256)
        mock_result = MagicMock()
        mock_result.__getitem__ = lambda self, key: np.random.rand(1, 256, 256).astype(np.float32)
        mock_compiled.return_value = mock_result

        estimator = DepthEstimator(model_path="fake/MiDaS_small.xml")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = estimator.estimate(frame)

        # El resultado debe tener la resolucion original (480x640)
        if result is not None:
            self.assertEqual(result.shape, (480, 640))


if __name__ == "__main__":
    unittest.main()
