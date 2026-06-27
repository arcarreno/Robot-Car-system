"""
Deteccion de colores de semaforo usando YOLO + HSV hibrido.

Pipeline hibrido:
  1. YOLO detecta semaforo (bounding box) y flechas
  2. Se recorta el bounding box del semaforo
  3. HSV lee el color dentro del recorte
  4. YOLO determina la direccion de la flecha

Fallback: si YOLO no esta disponible, usa HSV puro.

Rangos HSV calibrados para ESP32-CAM (AWB shift +15):
  Rojo:     H 0-10 / 165-180
  Amarillo: H 15-50
  Verde:    H 50-85
"""

import cv2
import numpy as np
from collections import Counter
from config import COLOR_RANGES, COLOR_MIN_AREA, ARROW_MIN_AREA


class ColorDetector:
    # Suavizado exponencial del bbox: alpha es el peso de la nueva
    # deteccion de YOLO. alpha=1.0 = sin suavizado (salta entre frames);
    # alpha=0.2 = muy suave pero con latencia visible. 0.4 es un buen
    # balance para FASE 4 (YOLO corre cada 2 frames).
    BBOX_SMOOTH_ALPHA = 0.45

    # B1 fix: histéresis de color. Para confirmar un color, debe aparecer
    # en al menos COLOR_CONFIRM_THRESHOLD de los últimos COLOR_HISTORY_SIZE
    # frames. Esto evita que un parpadeo de cámara o una detección errónea
    # de un solo frame cause que el robot reaccione incorrectamente.
    COLOR_HISTORY_SIZE = 5
    COLOR_CONFIRM_THRESHOLD = 3

    def __init__(self, yolo_detector=None):
        """
        Args:
            yolo_detector: instancia de YoloDetector (opcional)
        """
        self._yolo = yolo_detector
        self._overlay = None
        self._last_dims = (0, 0)
        # Cache de bbox suavizado para que el dibujo no salte entre
        # detecciones de YOLO (FASE 4 salta 1 frame). El crop HSV sigue
        # usando el bbox crudo (necesita la posicion real).
        self._smooth_semaforo = None  # dict con x1, y1, x2, y2, confidence, class
        self._smooth_arrows = []     # lista de dicts suavizados
        # B1: historial de colores detectados (últimos N frames)
        self._color_history = []
        # B8 fix: kernel pre-allocado para no crear np.ones() en cada frame
        self._kernel = np.ones((3, 3), np.uint8)

    def _confirm_color_with_hysteresis(self, raw_color: str) -> str:
        """
        B1 fix: confirma un color solo si aparece en al menos
        COLOR_CONFIRM_THRESHOLD de los últimos COLOR_HISTORY_SIZE frames.
        Esto evita que un parpadeo de cámara o una detección errónea
        de un solo frame cause que el robot reaccione incorrectamente.

        Args:
            raw_color: color detectado en este frame (o None)

        Returns:
            color confirmado o None
        """
        self._color_history.append(raw_color)
        # Mantener solo los últimos N frames
        if len(self._color_history) > self.COLOR_HISTORY_SIZE:
            self._color_history = self._color_history[-self.COLOR_HISTORY_SIZE:]

        # Contar ocurrencias de cada color (excluyendo None)
        counts = Counter(c for c in self._color_history if c is not None)

        # El color más frecuente en el historial
        if not counts:
            return None

        most_common_color, most_common_count = counts.most_common(1)[0]

        # Confirmar solo si supera el umbral
        if most_common_count >= self.COLOR_CONFIRM_THRESHOLD:
            return most_common_color

        return None

    def _detect_color_hsv(self, crop: np.ndarray) -> str:
        """
        Detecta color dentro de un crop usando HSV puro.

        Devuelve el color con mayor area de pixels, siempre que supere
        el umbral minimo. Esto evita que rojos residuales del housing
        ganen por prioridad cuando el LED activo es verde o amarillo.
        """
        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        kernel = self._kernel
        crop_area = crop.shape[0] * crop.shape[1]

        # DEBUG: HSV stats cada 60 frames
        self._debug_hsv_count = getattr(self, '_debug_hsv_count', 0) + 1
        if self._debug_hsv_count % 60 == 0:
            h_mean, s_mean, v_mean = hsv[:,:,0].mean(), hsv[:,:,1].mean(), hsv[:,:,2].mean()
            print(f"[HSV] crop={crop.shape} H={h_mean:.0f} S={s_mean:.0f} V={v_mean:.0f}")

        # Escalar area minima segun tamaño del crop
        min_area = COLOR_MIN_AREA * (crop_area / (640 * 640))
        min_area = max(min_area, 50)

        # Calcular area de cada color
        color_areas = {}
        for color_name in ["red", "yellow", "green"]:
            if color_name == "red":
                mask1 = cv2.inRange(hsv, *COLOR_RANGES["red_low"])
                mask2 = cv2.inRange(hsv, *COLOR_RANGES["red_high"])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv, *COLOR_RANGES[color_name])

            cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            area = cv2.countNonZero(cleaned)
            color_areas[color_name] = area

            if self._debug_hsv_count % 60 == 0:
                print(f"  {color_name}: area={area} min={min_area}")

        # Devolver el color con MAYOR area (no prioridad fija)
        best_color = None
        best_area = 0
        for color_name, area in color_areas.items():
            if area >= min_area and area > best_area:
                best_area = area
                best_color = color_name

        return best_color

    def _detect_color_hsv_full(self, frame: np.ndarray) -> str:
        """
        Detecta color en el frame completo usando HSV (fallback sin YOLO).
        Prioridad: ROJO > AMARILLO > VERDE
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width = frame.shape[:2]
        total_area = height * width
        kernel = self._kernel

        for color_name in ["red", "yellow", "green"]:
            if color_name == "red":
                mask1 = cv2.inRange(hsv, *COLOR_RANGES["red_low"])
                mask2 = cv2.inRange(hsv, *COLOR_RANGES["red_high"])
                mask = cv2.bitwise_or(mask1, mask2)
            else:
                mask = cv2.inRange(hsv, *COLOR_RANGES[color_name])

            cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            area = cv2.countNonZero(cleaned)

            if area >= COLOR_MIN_AREA:
                return color_name

        return None

    def detect(self, frame: np.ndarray, run_yolo: bool = True) -> dict:
        """
        Detecta colores de semaforo en un frame BGR.

        Args:
            frame: imagen BGR de OpenCV
            run_yolo: si False, NO ejecuta YOLO este frame (FASE 4 / B3 skip).
                YoloDetector devolvera su cache. HSV siempre corre.

        Returns:
            dict con:
                - detected: "red" | "yellow" | "green" | None
                - arrow: "left" | "right" | None
                - colors: lista de colores detectados
                - overlay: np.ndarray con visualizacion
        """
        if frame is None:
            return {"detected": None, "arrow": None, "colors": [], "overlay": None}

        height, width = frame.shape[:2]

        # P2 fix: usar el buffer pre-asignado self._overlay en vez de crear
        # uno nuevo con np.zeros() cada frame (230KB/frame de garbage). El
        # truco: dimensionar la primera vez, despues solo "limpiar" con [:]=0
        # (operacion O(N) en sitio, sin asignacion).
        if self._last_dims != (height, width):
            self._overlay = np.zeros((height, width, 3), dtype=np.uint8)
            self._last_dims = (height, width)
        else:
            self._overlay[:] = 0

        # Atajo local para no escribir self._overlay en cada cv2.rectangle
        overlay = self._overlay

        detected = None
        arrow = None
        results = []

        # ========================================
        # MODO YOLO + HSV (hibrido)
        # ========================================
        if self._yolo and self._yolo.is_available:
            # FASE 4 (B3): pasar flag de skip a YoloDetector
            yolo_results = self._yolo.detect(frame, run_yolo=run_yolo)

            # Si YOLO detecto el semaforo, recortar y leer color con HSV
            if yolo_results["semaforo"]:
                bbox = yolo_results["semaforo"]
                crop = frame[bbox["y1"]:bbox["y2"], bbox["x1"]:bbox["x2"]]

                # DEBUG: mostrar bbox cada 60 frames
                self._debug_crop_count = getattr(self, '_debug_crop_count', 0) + 1
                if self._debug_crop_count % 60 == 0:
                    print(f"[CROP] x1={bbox['x1']} y1={bbox['y1']} "
                          f"x2={bbox['x2']} y2={bbox['y2']} "
                          f"crop={crop.shape} conf={bbox.get('confidence', '?')}")

                detected = self._detect_color_hsv(crop)

                # Suavizar bbox para que el dibujo no salte entre frames
                # (FASE 4 corre YOLO cada 2 frames; sin suavizado, la caja
                # "salta" de la posicion vieja a la nueva en cada deteccion).
                # El crop HSV sigue usando bbox crudo (necesario para precision).
                self._smooth_semaforo = self._smooth_bbox(
                    self._smooth_semaforo, bbox)
                sb = self._smooth_semaforo
                cv2.rectangle(overlay,
                             (sb["x1"], sb["y1"]),
                             (sb["x2"], sb["y2"]),
                             (0, 255, 255), 2)

                if detected:
                    # B1 fix: precedencia correcta. Antes: bbox["x2"] - bbox["x1"] * bbox["y2"] - bbox["y1"]
                    # se evaluaba como x2 - (x1 * y2) - y1, produciendo un numero sin sentido.
                    # Ahora: (x2 - x1) * (y2 - y1) = area real del bbox.
                    bbox_area = (bbox["x2"] - bbox["x1"]) * (bbox["y2"] - bbox["y1"])
                    results.append({"name": detected, "area": bbox_area, "ratio": 0.0})
            else:
                # YOLO no detecto semaforo este frame. Si el suavizado
                # anterior existe, hacer "decay" (acercarse a None) en vez
                # de saltar abruptamente.
                self._smooth_semaforo = self._decay_smooth(self._smooth_semaforo)

            # Flechas directas de YOLO
            arrow = self._yolo.get_arrow_direction(yolo_results)

            # Suavizar flechas y dibujarlas. Matching por clase: cada flecha
            # cruda busca su contraparte suavizada por class name.
            new_arrows_raw = yolo_results.get("arrows", [])
            self._smooth_arrows = self._smooth_arrow_list(
                self._smooth_arrows, new_arrows_raw)
            for sb in self._smooth_arrows:
                # BGR en OpenCV: (0,255,0)=verde, (0,0,255)=rojo.
                # Bugfix: antes era (255,0,0) para arrow_left, que es AZUL.
                if sb["class"] == "arrow_right":
                    color = (0, 255, 0)     # verde
                elif sb["class"] == "arrow_left":
                    color = (0, 0, 255)     # rojo
                else:
                    color = (255, 255, 0)   # cyan (caso no esperado)
                cv2.rectangle(overlay,
                             (sb["x1"], sb["y1"]),
                             (sb["x2"], sb["y2"]),
                             color, 2)

            # NO fallback a HSV completo: si YOLO no detecto semaforo,
            # es porque no hay uno en el frame. El fallback HSV puro
            # causaba falsos positivos con cualquier objeto amarillo/verde
            # del ambiente (paredes, mesas, iluminacion).

        # ========================================
        # MODO HSV PURO (fallback)
        # ========================================
        else:
            detected = self._detect_color_hsv_full(frame)
            if detected:
                results.append({"name": detected, "area": 0, "ratio": 0.0})

            # Flecha por posicion (solo si hay verde)
            if detected == "green":
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask_green = cv2.inRange(hsv, *COLOR_RANGES["green"])
                arrow = self._detect_arrow_position(mask_green)

        # B1 fix: aplicar histéresis al color detectado.
        # Solo confirma un color si apareció en 3+ de los últimos 5 frames.
        # Esto filtra parpadeos de cámara y detecciones erróneas sueltas.
        confirmed = self._confirm_color_with_hysteresis(detected)

        return {
            "detected": confirmed,
            "arrow": arrow,
            "colors": results,
            "overlay": self._overlay,
        }

    def _detect_arrow_position(self, green_mask: np.ndarray) -> str:
        """
        Detecta flecha por POSICION (fallback sin YOLO):
        píxeles verdes FUERA del círculo principal.
        """
        kernel = self._kernel
        cleaned = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        moment = cv2.moments(largest)
        if moment["m00"] == 0:
            return None

        main_cx = int(moment["m10"] / moment["m00"])

        main_mask = np.zeros_like(cleaned)
        cv2.drawContours(main_mask, [largest], -1, 255, thickness=cv2.FILLED)

        outside_mask = cv2.bitwise_and(cleaned, cv2.bitwise_not(main_mask))
        outside_pixels = cv2.countNonZero(outside_mask)

        if outside_pixels < ARROW_MIN_AREA:
            return None

        outside_moment = cv2.moments(outside_mask)
        if outside_moment["m00"] == 0:
            return None
        outside_cx = int(outside_moment["m10"] / outside_moment["m00"])

        offset = outside_cx - main_cx
        if offset < -10:
            return "left"
        elif offset > 10:
            return "right"

        return None

    # ------------------------------------------------------------------
    # Suavizado exponencial de bboxes
    # ------------------------------------------------------------------

    def _smooth_bbox(self, prev: dict, new: dict) -> dict:
        """
        Combina el bbox anterior (suavizado) con el nuevo de YOLO usando
        interpolacion exponencial. Devuelve un dict nuevo con x1, y1, x2,
        y2 enteros (para cv2.rectangle) y confidence/class del nuevo.
        Si no hay previo, devuelve una copia del nuevo.
        """
        if prev is None:
            return dict(new)
        a = self.BBOX_SMOOTH_ALPHA
        return {
            "x1": int(round(a * new["x1"] + (1 - a) * prev["x1"])),
            "y1": int(round(a * new["y1"] + (1 - a) * prev["y1"])),
            "x2": int(round(a * new["x2"] + (1 - a) * prev["x2"])),
            "y2": int(round(a * new["y2"] + (1 - a) * prev["y2"])),
            "confidence": new["confidence"],
            "class": new["class"],
        }

    def _decay_smooth(self, prev: dict) -> dict:
        """
        Cuando YOLO no detecta el semaforo en un frame, hacemos decay del
        bbox suavizado: lo movemos hacia el centro del frame gradualmente.
        Asi la caja "se va" en vez de saltar a None. Si la perdida
        persiste, eventualmente queda en el centro y al proximo frame
        vuelve a dibujarse solo si YOLO la vuelve a detectar.
        """
        if prev is None:
            return None
        # Decay muy suave: 10% hacia el centro por frame
        a = 0.1
        cx_frame, cy_frame = self._last_dims[1] // 2, self._last_dims[0] // 2
        prev_cx = (prev["x1"] + prev["x2"]) // 2
        prev_cy = (prev["y1"] + prev["y2"]) // 2
        # Si el bbox esta muy cerca del centro, eliminarlo
        if abs(prev_cx - cx_frame) < 3 and abs(prev_cy - cy_frame) < 3:
            return None
        w = prev["x2"] - prev["x1"]
        h = prev["y2"] - prev["y1"]
        new_cx = int(round(a * cx_frame + (1 - a) * prev_cx))
        new_cy = int(round(a * cy_frame + (1 - a) * prev_cy))
        return {
            "x1": new_cx - w // 2,
            "y1": new_cy - h // 2,
            "x2": new_cx + w // 2,
            "y2": new_cy + h // 2,
            "confidence": prev["confidence"],
            "class": prev["class"],
        }

    def _smooth_arrow_list(self, prev_list: list, new_list: list) -> list:
        """
        Suaviza una lista de flechas matching por class. Las flechas son
        detecciones PUNTUALES (no objetos fisicos persistentes como el
        semaforo), asi que cuando YOLO deja de detectar una clase este
        frame, NO se hace decay: la flecha desaparece inmediatamente.
        Esto evita el bug del "boxe flotando hacia el centro" cuando
        la deteccion de YOLO es intermitente.
        """
        new_by_class = {a["class"]: a for a in new_list}
        prev_by_class = {a["class"]: a for a in prev_list}
        result = []
        # Suavizar las que aparecen en este frame (usando el suavizado
        # previo si existe, para que el movimiento entre detecciones
        # sea fluido en vez de saltar).
        for cls, new_arrow in new_by_class.items():
            prev_arrow = prev_by_class.get(cls)
            result.append(self._smooth_bbox(prev_arrow, new_arrow))
        # Las clases que estaban antes pero YA NO estan en este frame
        # se descartan (sin decay). Si YOLO no las ve, no se dibujan.
        return result
