"""Tests para el modelo FrameResult."""
import sys
import os
import unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.frame_result import FrameResult


class TestFrameResult(unittest.TestCase):
    """Tests de serializacion de FrameResult."""

    def test_basic_to_dict(self):
        result = FrameResult(
            state="COLOR_CHECK",
            detected_color="green",
            command_sent="go",
        )
        d = result.to_dict()
        self.assertEqual(d["state"], "COLOR_CHECK")
        self.assertEqual(d["detected_color"], "green")
        self.assertIsNone(d["detected_arrow"])
        self.assertEqual(d["command_sent"], "go")

    def test_arrow_included_in_dict(self):
        result = FrameResult(
            state="COLOR_CHECK",
            detected_color="green",
            command_sent="left",
            detected_arrow="left",
        )
        d = result.to_dict()
        self.assertEqual(d["detected_arrow"], "left")

    def test_route_fields_included_when_set(self):
        result = FrameResult(
            state="ROUTE",
            detected_color="green",
            command_sent="go",
            route_progress=0.5,
            route_phase="out",
        )
        d = result.to_dict()
        self.assertEqual(d["route_progress"], 0.5)
        self.assertEqual(d["route_phase"], "out")

    def test_route_fields_excluded_when_none(self):
        result = FrameResult(
            state="COLOR_CHECK",
            detected_color="red",
            command_sent="stop",
        )
        d = result.to_dict()
        self.assertNotIn("route_progress", d)
        self.assertNotIn("route_phase", d)

    def test_colors_serialized_correctly(self):
        result = FrameResult(
            state="IDLE",
            detected_color=None,
            command_sent=None,
            colors=[{"name": "red", "area": 1000}, {"name": "green", "area": 5000}],
        )
        d = result.to_dict()
        self.assertEqual(len(d["colors"]), 2)
        self.assertEqual(d["colors"][0]["name"], "red")


if __name__ == '__main__':
    unittest.main()
