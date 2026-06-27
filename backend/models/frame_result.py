"""
Modelo de datos para el resultado del procesamiento de un frame.

Se envia al frontend via WebSocket como JSON.
"""


from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FrameResult:
    """
    Resultado completo del procesamiento de un frame.
    Se serializa a JSON para el WebSocket.
    """
    state: str                     # estado actual de la maquina
    detected_color: Optional[str]  # None | "red" | "yellow" | "green"
    command_sent: Optional[str]    # None | "go" | "stop" | "left" | etc
    detected_arrow: Optional[str] = None  # None | "left" | "right"
    colors: list = field(default_factory=list)
    fps: float = 0.0
    frame_b64: Optional[str] = None  # frame anotado en base64 JPEG
    route_progress: Optional[float] = None  # 0.0 - 1.0, solo en estado ROUTE
    route_phase: Optional[str] = None       # "out" | "turn" | "back" | "done" | None
    # Deteccion de obstaculos (MiDaS depth)
    obstacle_distance: Optional[float] = None  # metros al obstaculo mas cercano
    obstacle_detected: bool = False             # True si hay obstaculo bajo umbral
    obstacle_direction: Optional[str] = None    # "left" | "right" — lado con mas espacio
    obstacle_ttc: Optional[float] = None        # time-to-collision en segundos
    obstacle_approach_speed: float = 0.0        # m/s de aproximacion (positivo = acercandose)

    def to_dict(self):
        d = {
            "state": self.state,
            "detected_color": self.detected_color,
            "detected_arrow": self.detected_arrow,
            "command_sent": self.command_sent,
            "colors": [
                {"name": c["name"], "area": c["area"]}
                for c in self.colors
            ],
            "fps": round(self.fps, 1),
            "frame_b64": self.frame_b64,
        }
        if self.route_progress is not None:
            d["route_progress"] = round(self.route_progress, 2)
        if self.route_phase is not None:
            d["route_phase"] = self.route_phase
        if self.obstacle_distance is not None:
            d["obstacle_distance"] = round(self.obstacle_distance, 3)
        d["obstacle_detected"] = self.obstacle_detected
        if self.obstacle_direction is not None:
            d["obstacle_direction"] = self.obstacle_direction
        if self.obstacle_ttc is not None:
            d["obstacle_ttc"] = round(self.obstacle_ttc, 2)
        d["obstacle_approach_speed"] = round(self.obstacle_approach_speed, 3)
        return d
