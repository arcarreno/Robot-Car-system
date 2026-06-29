"""
Depth estimation monocular using MiDaS Small v2.1 (OpenVINO).

Estima profundidad relativa desde una imagen monoculular y la convierte
a distancia aproximada en metros usando calibracion empirica.

Pipeline de mejoras (v2 — "polished"):
  1. Temporal EMA filtering — suaviza ruido entre frames consecutivos
  2. Spatial median filter — elimina salt-and-pepper noise del depth map
  3. Multi-threshold zones — pre-caution (1.0m), caution (0.5m), danger (0.3m)
  4. Minimum detection duration — confirma obstaculo por N frames consecutivos
  5. Weighted free-space — pondera por distancia, no solo promedio
  6. Adaptive turn — gira hasta que el camino este despejado (no timer fijo)

Uso:
    estimator = DepthEstimator(model_path, device="CPU")
    depth_map = estimator.estimate(frame)
    result = estimator.analyze(depth_map)  # → ObstacleResult
"""

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ObstacleResult:
    """Resultado del analisis de obstaculos en un frame."""
    distance: Optional[float]       # metros al obstaculo mas cercano (None = sin datos)
    direction: str                  # "left" | "right" — lado con mas espacio
    zone: str                       # "clear" | "pre_caution" | "caution" | "danger"
    raw_depth_center: float         # valor raw de depth en el centro (debug)
    confidence: float               # 0.0-1.0, cuantos frames consecutivos confirmaron
    ttc: Optional[float]            # time-to-collision en segundos (None = infinito)
    approach_speed: float           # m/s de aproximacion (positivo = acercándose)
    close_ratio: float = 0.0        # 0.0-1.0, % del ROI central con raw > umbral


class DepthEstimator:
    """Estimador de profundidad monocular con MiDaS Small v2.1."""

    def __init__(
        self,
        model_path: str,
        device: str = "CPU",
        calibration_k: float = 99997.1,
        calibration_p: float = 1.94,
        calibration_offset: float = 140.0,
        # Filtros v2
        ema_alpha: float = 0.35,          # suavizado temporal (0.35 = 35% nuevo + 65% anterior)
        spatial_kernel_size: int = 5,      # kernel del filtro mediana espacial
        min_detection_frames: int = 2,     # frames consecutivos para confirmar obstaculo
        zone_caution: float = 0.8,         # zona de precaucion (metros)
        zone_danger: float = 0.4,          # zona de peligro (metros)
        zone_pre_caution: float = 1.2,     # zona de pre-caucion (metros)
        # Bilateral filter (preserva bordes, suaviza ruido)
        bilateral_d: int = 7,
        bilateral_sigma_color: float = 50.0,
        bilateral_sigma_space: float = 50.0,
        # Deteccion close_ratio: para paredes/muebles grandes que llenan la pantalla
        close_raw_threshold: float = 450.0,   # raw value umbral para considerar "cerca"
        close_ratio_threshold: float = 0.7,   # % minimo del ROI para trigger danger
        # Distancia minima libre para analyze_directions
        free_distance: float = 0.7,           # metros — distancia minima para considerar "libre"
    ):
        """
        Args:
            model_path: Ruta al .xml del modelo OpenVINO IR.
            device: Dispositivo OpenVINO ("CPU", "GPU", "AUTO").
            calibration_k: Constante K para conversion depth→metros.
            calibration_p: Exponente P para conversion depth→metros.
            calibration_offset: Offset para conversion depth→metros.
            ema_alpha: Factor de suavizado temporal (0.35 = 35% nuevo + 65% anterior).
            spatial_kernel_size: Tamaño del kernel mediana para suavizado espacial.
            min_detection_frames: Frames consecutivos para confirmar un obstaculo.
            zone_caution: Distancia de precaucion en metros.
            zone_danger: Distancia de peligro en metros.
            zone_pre_caution: Distancia de pre-caucion (aviso temprano).
            bilateral_d: Diametro del filtro bilateral (奇数, 5-9).
            bilateral_sigma_color: Sigma color del bilateral (mayor = mas suavizado).
            bilateral_sigma_space: Sigma espacio del bilateral (mayor = mas rango).
        """
        self._cal_k = calibration_k
        self._cal_p = calibration_p
        self._cal_offset = calibration_offset
        self._compiled_model = None
        self._input_key = None
        self._output_key = None
        self._input_shape = None  # [1, 3, H, W]

        # Filtros v2
        self._ema_alpha = ema_alpha
        self._spatial_kernel = spatial_kernel_size
        self._min_detection_frames = min_detection_frames
        self._zone_caution = zone_caution
        self._zone_danger = zone_danger
        self._zone_pre_caution = zone_pre_caution

        # Bilateral filter
        self._bilateral_d = bilateral_d
        self._bilateral_sigma_color = bilateral_sigma_color
        self._bilateral_sigma_space = bilateral_sigma_space

        # Close ratio detection
        self._close_raw_threshold = close_raw_threshold
        self._close_ratio_threshold = close_ratio_threshold

        # Free distance threshold for analyze_directions
        self._free_distance = free_distance

        # Estado temporal (ANTES del check de modelo para que existan siempre)
        self._prev_distance: Optional[float] = None   # EMA del frame anterior
        self._prev_direction: str = "right"            # ultima dirección conocida
        self._consecutive_detection: int = 0           # frames consecutivos con obstaculo
        self._consecutive_clear: int = 0               # frames consecutivos sin obstaculo

        # TTC: Time-to-Collision tracking
        self._prev_distance_for_ttc: Optional[float] = None  # distancia raw para TTC
        self._prev_time_for_ttc: float = 0.0                 # timestamp anterior
        self._approach_speed: float = 0.0                     # m/s de aproximacion
        self._ttc: Optional[float] = None                    # seconds to collision

        if not os.path.exists(model_path):
            logger.warning(
                "[DepthEstimator] Modelo no encontrado: %s. "
                "Depth estimation deshabilitado. "
                "Ejecute: python scripts/download_midas.py",
                model_path,
            )
            return

        try:
            import openvino as ov

            core = ov.Core()
            model = core.read_model(model_path)
            self._compiled_model = core.compile_model(
                model=model, device_name=device
            )
            self._input_key = self._compiled_model.input(0)
            self._output_key = self._compiled_model.output(0)
            self._input_shape = list(self._input_key.shape)  # [1, 3, 256, 256]

            logger.info(
                "[DepthEstimator] Modelo cargado: %s (device=%s, input=%s)",
                os.path.basename(model_path),
                device,
                self._input_shape,
            )
            print(f"[DepthEstimator] Modelo cargado OK: device={device}, "
                  f"input={self._input_shape}")
        except Exception as e:
            logger.error(
                "[DepthEstimator] Error cargando modelo: %s. "
                "Depth estimation deshabilitado.",
                e,
            )
            self._compiled_model = None

    @property
    def available(self) -> bool:
        """True si el modelo esta cargado y listo."""
        return self._compiled_model is not None

    # =========================================================================
    # Core inference
    # =========================================================================

    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Estima mapa de profundidad desde un frame BGR.

        Args:
            frame: Imagen BGR (HxWx3, uint8).

        Returns:
            Mapa de profundidad suavizado (HxW, float32) con valores relativos
            (mayor = mas cerca), o None si el modelo no esta disponible.
        """
        if not self.available:
            return None

        try:
            h, w = frame.shape[:2]
            net_h, net_w = self._input_shape[2], self._input_shape[3]

            # Resize a resolucion del modelo
            resized = cv2.resize(frame, (net_w, net_h))

            # Convertir BGR → RGB, HWC → CHW, agregar batch
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_image = np.expand_dims(
                np.transpose(rgb, (2, 0, 1)), 0
            ).astype(np.float32)

            # Inferencia
            result = self._compiled_model([input_image])[self._output_key]
            depth_map = result.squeeze()  # (256, 256)

            # Resize de vuelta a resolucion original
            depth_map = cv2.resize(depth_map, (w, h))

            # Filtro bilateral: preserva bordes de obstaculos, suaviza ruido
            if self._bilateral_d > 0:
                depth_map = cv2.bilateralFilter(
                    depth_map.astype(np.float32),
                    self._bilateral_d,
                    self._bilateral_sigma_color,
                    self._bilateral_sigma_space,
                )

            # Filtro mediana: elimina salt-and-pepper residual
            if self._spatial_kernel > 1:
                depth_map = cv2.medianBlur(
                    depth_map.astype(np.float32),
                    self._spatial_kernel,
                )

            return depth_map

        except Exception as e:
            logger.error("[DepthEstimator] Error en inference: %s", e)
            return None

    # =========================================================================
    # Conversion depth → metros
    # =========================================================================

    def raw_to_meters(self, depth_value: float) -> float:
        """
        Convierte un valor de depth relativo a distancia en metros.

        Usa la formula: D = K / (value - offset)^P

        Args:
            depth_value: Valor del mapa de profundidad.

        Returns:
            Distancia estimada en metros (float).
        """
        adjusted = depth_value - self._cal_offset
        if adjusted <= 0:
            return float("inf")
        return self._cal_k / (adjusted ** self._cal_p)

    # =========================================================================
    # Analisis de obstaculos (v2 — mejorado)
    # =========================================================================

    def get_obstacle_distance(
        self,
        depth_map: np.ndarray,
        center_ratio: float = 0.4,
    ) -> Optional[float]:
        """
        Obtiene la distancia al obstaculo mas cercano en la region central.

        v2: Usa ROI mas grande (40%) y filtro temporal EMA.

        Args:
            depth_map: Mapa de profundidad (HxW, float32).
            center_ratio: Proporcion del frame central a analizar (0.0-1.0).

        Returns:
            Distancia en metros al obstaculo mas cercano, o None si no hay datos.
        """
        if depth_map is None:
            return None

        h, w = depth_map.shape
        cy, cx = h // 2, w // 2
        half_h = int(h * center_ratio / 2)
        half_w = int(w * center_ratio / 2)

        # Extraer region central
        roi = depth_map[
            cy - half_h : cy + half_h,
            cx - half_w : cx + half_w,
        ]

        if roi.size == 0:
            return None

        # --- Validacion de calidad del depth map ---
        roi_std = float(np.std(roi))
        roi_range = float(np.max(roi) - np.min(roi))

        # Si muy poca varianza (modelo no detectó nada) o demasiada (ruido total),
        # mantener valor anterior para no envenenar el EMA
        if (roi_std < 0.01 or roi_range > 500) and self._prev_distance is not None:
            return self._prev_distance

        # --- Rechazo de outliers: eliminar pixeles fuera de 2σ ---
        median_val = float(np.median(roi))
        if roi_std > 1.0:  # solo si hay varianza real
            mask = np.abs(roi - median_val) < 2 * roi_std
            filtered_roi = roi[mask]
            if filtered_roi.size > roi.size * 0.3:  # al menos 30% de pixeles validos
                roi = filtered_roi

        # Usar percentil 90 en vez de max para filtrar outliers restantes
        p90_depth = float(np.percentile(roi, 90))
        raw_distance = self.raw_to_meters(p90_depth)

        # Guard: si raw_distance no es finito (inf/nan), no envenenar EMA
        if not math.isfinite(raw_distance):
            return self._prev_distance  # mantener ultimo valor conocido

        # Filtro EMA temporal: suavizar entre frames
        # Jump detector: si salta >0.8m en un frame, confiar en el raw
        # (evita lag cuando un obstaculo aparece de golpe)
        JUMP_THRESHOLD = 0.8
        if self._prev_distance is None:
            smoothed = raw_distance
        elif abs(raw_distance - self._prev_distance) > JUMP_THRESHOLD:
            smoothed = raw_distance  # salto grande → confiar en medicion cruda
        else:
            smoothed = (
                self._ema_alpha * raw_distance
                + (1.0 - self._ema_alpha) * self._prev_distance
            )

        self._prev_distance = smoothed

        # --- TTC: Time-to-Collision tracking ---
        now = time.time()
        if not hasattr(self, '_prev_distance_for_ttc'):
            self._prev_distance_for_ttc = None
            self._prev_time_for_ttc = 0.0
            self._approach_speed = 0.0
            self._ttc = None

        if self._prev_distance_for_ttc is not None and self._prev_time_for_ttc > 0:
            dt = now - self._prev_time_for_ttc
            if dt > 0.05:  # ignorar differences minusculas (min 50ms)
                raw_speed = (self._prev_distance_for_ttc - smoothed) / dt
                # Clamp speed: robot real va max ~1 m/s. Valores mayores son ruido.
                clamped_speed = max(-1.0, min(1.0, raw_speed))
                # Suavizar speed con EMA (α=0.4)
                if not hasattr(self, '_ema_speed'):
                    self._ema_speed = 0.0
                self._ema_speed = 0.4 * clamped_speed + 0.6 * self._ema_speed
                self._approach_speed = self._ema_speed
                # TTC = distancia / velocidad_aproximacion (solo si se acerca)
                if self._approach_speed > 0.1:  # umbral mas alto para evitar ruido
                    self._ttc = smoothed / self._approach_speed
                else:
                    self._ttc = None  # no se esta acercando
        self._prev_distance_for_ttc = smoothed
        self._prev_time_for_ttc = now

        return smoothed

    def get_free_space_direction(
        self,
        depth_map: np.ndarray,
        center_ratio: float = 0.3,
    ) -> str:
        """
        Determina qué lado tiene más espacio libre.

        v2: Usa scoring ponderado por distancia (no solo promedio).
        Un lado con un obstaculo muy cerca recibe penalización extra.

        Args:
            depth_map: Mapa de profundidad (HxW, float32).
            center_ratio: Proporcion central a ignorar (zona muerta del robot).

        Returns:
            "left" o "right" — el lado con mas espacio libre.
        """
        if depth_map is None:
            return self._prev_direction  # mantener ultima dirección conocida

        h, w = depth_map.shape
        cy = h // 2
        dead_half_w = int(w * center_ratio / 2)

        # Dividir en izquierda / derecha (excluyendo zona muerta central)
        left_zone = depth_map[:, dead_half_w : w // 2]
        right_zone = depth_map[:, w // 2 : w - dead_half_w]

        if left_zone.size == 0 or right_zone.size == 0:
            return self._prev_direction

        # Scoring ponderado: menor depth promedio = mas lejos = mas espacio
        # Pero tambien penalizar si hay CUALQUIER pixel muy cerca (< 0.3m raw)
        left_mean = float(np.mean(left_zone))
        right_mean = float(np.mean(right_zone))

        # Penalty: si algun lado tiene pixeles muy cerca, penalizar
        # MiDaS: valor alto = mas cerca. Usar MAX (pixel mas cercano)
        left_max = float(np.max(left_zone))
        right_max = float(np.max(right_zone))

        # Score final: promedio + penalty por cercania extrema
        # Un obstaculo muy cerca (depth alto) penaliza mucho
        left_score = left_mean + 0.5 * left_max
        right_score = right_mean + 0.5 * right_max

        if left_score < right_score:
            direction = "left"
        else:
            direction = "right"

        self._prev_direction = direction
        return direction

    # =========================================================================
    # Analisis completo (v2 — punto de entrada principal)
    # =========================================================================

    def analyze(self, depth_map: np.ndarray) -> ObstacleResult:
        """
        Analisis completo de obstaculos con filtrado temporal.

        Combina distancia, direccion, zonas y confirmación multi-frame.

        Args:
            depth_map: Mapa de profundidad (HxW, float32).

        Returns:
            ObstacleResult con toda la informacion.
        """
        if depth_map is None:
            self._consecutive_detection = 0
            self._consecutive_clear += 1
            return ObstacleResult(
                distance=None,
                direction=self._prev_direction,
                zone="clear",
                raw_depth_center=0.0,
                confidence=0.0,
                ttc=None,
                approach_speed=0.0,
            )

        h, w = depth_map.shape
        cy, cx = h // 2, w // 2

        # Obtener distancia y dirección
        distance = self.get_obstacle_distance(depth_map)
        direction = self.get_free_space_direction(depth_map)

        # Valor raw del centro (para debug/calibracion)
        center_roi = depth_map[cy - 10 : cy + 10, cx - 10 : cx + 10]
        raw_center = float(np.mean(center_roi)) if center_roi.size > 0 else 0.0

        # --- Close ratio: detectar pared/mueble grande que llena la pantalla ---
        # Calcula que porcentaje del ROI central tiene valores raw altos ("cerca")
        half_h = int(h * 0.4 / 2)  # 40% del frame (mismo que get_obstacle_distance)
        half_w = int(w * 0.4 / 2)
        roi_full = depth_map[
            cy - half_h : cy + half_h,
            cx - half_w : cx + half_w,
        ]
        close_ratio = 0.0
        if roi_full.size > 0:
            close_mask = roi_full > self._close_raw_threshold
            close_ratio = float(np.sum(close_mask)) / roi_full.size

        # Determinar zona
        # PRIORIDAD: close_ratio override → distance-based zones
        if close_ratio > self._close_ratio_threshold:
            # Pared/mueble grande llenando la pantalla → danger directo
            zone = "danger"
        elif distance is None:
            zone = "clear"
        elif distance < self._zone_danger:
            zone = "danger"
        elif distance < self._zone_caution:
            zone = "caution"
        elif distance < self._zone_pre_caution:
            zone = "pre_caution"
        else:
            zone = "clear"

        # Confirmacion multi-frame
        if zone in ("danger", "caution"):
            self._consecutive_detection += 1
            self._consecutive_clear = 0
        elif zone == "clear":
            # Solo "clear" resetea el contador — pre_caution no
            self._consecutive_clear += 1
            if self._consecutive_clear >= self._min_detection_frames:
                self._consecutive_detection = 0

        confidence = min(
            self._consecutive_detection / self._min_detection_frames,
            1.0,
        )

        # Debug: mostrar estado cada 30 frames
        self._debug_count = getattr(self, '_debug_count', 0) + 1
        if self._debug_count % 30 == 0:
            dist_str = f"{distance:.2f}m" if distance else "None"
            print(f"[DEPTH] dist={dist_str} raw_center={raw_center:.1f} "
                  f"close_ratio={close_ratio:.2f} zone={zone} dir={direction} "
                  f"conf={confidence:.1f}")

        return ObstacleResult(
            distance=distance,
            direction=direction,
            zone=zone,
            raw_depth_center=raw_center,
            confidence=confidence,
            ttc=self._ttc,
            approach_speed=self._approach_speed,
            close_ratio=close_ratio,
        )

    # =========================================================================
    # Analisis de direcciones para escaneo (nuevo comportamiento)
    # =========================================================================

    def analyze_directions(self, depth_map: np.ndarray) -> list:
        """
        Analiza el mapa de profundidad y retorna distancias por dirección.

        Divide el frame en 3 zonas (izquierda, centro, derecha) y calcula
        la distancia al obstáculo más cercano en cada una.

        Args:
            depth_map: Mapa de profundidad (HxW, float32).

        Returns:
            Lista de diccionarios con:
            - "direction": "left" | "center" | "right"
            - "distance": metros al obstáculo más cercano
            - "free": True si distance > OBSTACLE_FREE_DISTANCE
        """
        if depth_map is None:
            return []

        h, w = depth_map.shape
        results = []

        # Dividir en 3 zonas: izquierda, centro, derecha
        third_w = w // 3
        zones = {
            "left": depth_map[:, :third_w],
            "center": depth_map[:, third_w:2*third_w],
            "right": depth_map[:, 2*third_w:],
        }

        for direction, zone in zones.items():
            if zone.size == 0:
                results.append({"direction": direction, "distance": float("inf"), "free": True})
                continue

            # Obtener distancia más cercana en esta zona (percentil 90 para filtrar outliers)
            p90 = float(np.percentile(zone, 90))
            distance = self.raw_to_meters(p90)
            free = distance > self._free_distance  # distancia mínima libre

            results.append({
                "direction": direction,
                "distance": distance,
                "free": free,
            })

        return results

    def get_best_free_direction(self, depth_map: np.ndarray) -> str:
        """
        Encuentra la dirección con más espacio libre.

        Analiza el mapa de profundidad y retorna la dirección con más espacio
        para avanzar. Si no hay espacio libre en ninguna dirección, retorna
        la dirección por defecto (izquierda).

        Args:
            depth_map: Mapa de profundidad (HxW, float32).

        Returns:
            "left" | "right" | "center" — dirección con más espacio libre.
        """
        directions = self.analyze_directions(depth_map)

        if not directions:
            return self._prev_direction

        # Filtrar solo direcciones libres
        free_dirs = [d for d in directions if d["free"]]

        if not free_dirs:
            # No hay espacio libre — retornar dirección por defecto
            return self._prev_direction

        # Retornar la dirección con más distancia (más espacio)
        best = max(free_dirs, key=lambda x: x["distance"])
        self._prev_direction = best["direction"]
        return best["direction"]

    def has_free_path(self, depth_map: np.ndarray, min_distance: float = 0.3) -> bool:
        """
        Verifica si hay al menos un camino libre en el mapa de profundidad.

        Args:
            depth_map: Mapa de profundidad (HxW, float32).
            min_distance: Distancia mínima para considerar "libre" (metros).

        Returns:
            True si hay al menos una dirección con espacio libre.
        """
        directions = self.analyze_directions(depth_map)
        return any(d["free"] for d in directions)

    def reset_temporal(self):
        """Resetea el estado temporal (llamar al cambiar de modo)."""
        self._prev_distance = None
        self._prev_direction = "right"
        self._prev_distance_for_ttc = None
        self._prev_time_for_ttc = 0.0
        self._approach_speed = 0.0
        self._ttc = None
        self._consecutive_detection = 0
        self._consecutive_clear = 0
