"""
Anotador de frames: dibuja overlay semitransparente de colores detectados.

Los textos (color detectado, estado, FPS) se muestran en el frontend
via el panel de estado, NO en el video.
"""

import cv2
import numpy as np
from typing import Optional


def annotate_frame(
    frame: np.ndarray,
    color_overlay: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Dibuja overlay sobre el frame.

    Args:
        frame: frame BGR original.
        color_overlay: imagen BGR con dibujos (boxes, lineas) sobre fondo negro.
                       Se superpone directamente donde hay píxeles no-negros.

    Returns:
        Frame BGR con overlay.
    """
    if frame is None:
        return frame

    img = frame.copy()

    # Superponer dibujos del overlay (boxes, flechas) sobre el frame
    if color_overlay is not None:
        mask = np.any(color_overlay > 0, axis=2)
        if np.any(mask):
            img[mask] = color_overlay[mask]

    return img


def frame_to_b64(frame: np.ndarray,
                  quality: int = 80) -> Optional[str]:
    """
    Convierte un frame BGR a base64 JPEG para enviar por WebSocket.

    Args:
        frame: frame BGR de OpenCV.
        quality: calidad JPEG (0-100, 80 es buen balance).

    Returns:
        str: base64 del JPEG, o None si falla.
    """
    import base64

    if frame is None:
        return None

    success, buffer = cv2.imencode('.jpg', frame, [
        cv2.IMWRITE_JPEG_QUALITY, quality,
        cv2.IMWRITE_JPEG_OPTIMIZE, 1,
    ])
    if not success:
        return None

    return base64.b64encode(buffer).decode('utf-8')
