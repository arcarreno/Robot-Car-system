"""
Backend OpenVINO para YOLO.

Wrapper que carga un modelo YOLO exportado a OpenVINO IR (.xml + .bin) y
expone la misma interfaz que la rama de PyTorch: un metodo infer() que
recibe un frame BGR de OpenCV y devuelve detecciones en el mismo formato
que ultralytics (lista de dicts con x1, y1, x2, y2, conf, class).

Preferencias de device:
  1) Si hay Intel iGPU disponible y compila, usa GPU (mucho mas rapido)
  2) Si no, usa CPU (tambien acelera vs PyTorch)
  3) Si falla todo, lanza excepcion para que el caller haga fallback

Preprocesamiento identico al que hace ultralytics internamente:
  - resize a 640x640
  - BGR -> RGB
  - uint8 [0,255] -> float32 [0,1]
  - HWC -> NCHW

Postprocesamiento:
  - parseo de output0 shape [1, 4+nc, na] = [1, 7, 8400] para YOLO11 3-clase
  - argmax sobre scores, filtro por conf_threshold
  - xywh (centro) -> xyxy (esquinas)
  - NMS basico con NumPy para paridad con ultralytics
"""

import os
import time
import numpy as np
import cv2


class OpenVinoYoloBackend:
    """Wrapper de modelo YOLO exportado a OpenVINO IR."""

    # Tamaño de input que espera YOLO11 exportado (fijo, viene del .xml)
    INPUT_SIZE = 640

    def __init__(self, model_xml_path: str, class_names: list,
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                 device_preference: str = "AUTO"):
        """
        Args:
            model_xml_path: ruta al archivo .xml del modelo OpenVINO IR.
            class_names: lista de nombres de clase en el orden del modelo
                (ej. ["semaforo", "arrow_left", "arrow_right"]).
            conf_threshold: umbral de confianza minima (0.0 - 1.0).
            iou_threshold: umbral IoU para NMS (0.0 - 1.0).
            device_preference: "AUTO" (GPU si hay, sino CPU), "GPU" o "CPU".
        """
        self.model_xml_path = model_xml_path
        self.class_names = class_names
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device_preference = device_preference

        self._compiled_model = None
        self._input_layer = None
        self._output_layer = None
        self._device_used = None
        self._nc = len(class_names)  # numero de clases

        self._load_time_ms = None

    @property
    def is_available(self) -> bool:
        return self._compiled_model is not None

    @property
    def device_used(self) -> str:
        return self._device_used or "none"

    def load(self) -> None:
        """
        Carga y compila el modelo. Si falla con el device preferido,
        intenta CPU como fallback. Lanza RuntimeError si ambos fallan.
        """
        if not os.path.exists(self.model_xml_path):
            raise FileNotFoundError(
                f"Modelo OpenVINO no encontrado: {self.model_xml_path}")

        # Import lazy para que no sea un hard dependency
        try:
            import openvino as ov
        except ImportError as e:
            raise ImportError(
                "openvino no esta instalado. pip install openvino==2024.6.0"
            ) from e

        core = ov.Core()
        available = core.available_devices
        print(f"[OpenVINO] Devices disponibles: {available}")

        t0 = time.perf_counter()
        model = core.read_model(self.model_xml_path)

        # Resolver device: AUTO -> GPU si esta, sino CPU
        candidates = self._resolve_device_candidates(available)
        last_err = None
        for device in candidates:
            try:
                print(f"[OpenVINO] Compilando en {device}...")
                self._compiled_model = core.compile_model(model, device)
                self._device_used = device
                break
            except Exception as e:
                print(f"[OpenVINO] No se pudo compilar en {device}: {e}")
                last_err = e

        if self._compiled_model is None:
            raise RuntimeError(
                f"No se pudo compilar el modelo en ningun device "
                f"({candidates}). Ultimo error: {last_err}"
            )

        self._input_layer = self._compiled_model.input(0)
        self._output_layer = self._compiled_model.output(0)
        self._load_time_ms = (time.perf_counter() - t0) * 1000

        in_shape = self._input_layer.partial_shape
        print(f"[OpenVINO] Compilado OK en {self._device_used} "
              f"(input shape: {in_shape}, load={self._load_time_ms:.0f}ms)")

    def _resolve_device_candidates(self, available: list) -> list:
        """Devuelve la lista priorizada de devices a intentar."""
        pref = (self.device_preference or "AUTO").upper()
        if pref == "AUTO":
            # GPU primero si hay, sino CPU
            gpu = [d for d in available if d.startswith("GPU")]
            cpu = [d for d in available if d.startswith("CPU")]
            return (gpu + cpu) or ["CPU"]
        if pref in available:
            return [pref]
        # Si el preferido no esta disponible, degradar a CPU
        return [d for d in available if d.startswith("CPU")] or ["CPU"]

    def warmup(self, n: int = 3) -> None:
        """Corre N inferencias dummy para que la primera llamada real
        no pague el costo de warm-up del backend."""
        if not self.is_available:
            return
        dummy = np.zeros(
            (self.INPUT_SIZE, self.INPUT_SIZE, 3), dtype=np.uint8)
        for _ in range(n):
            self._infer_raw(dummy)

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR uint8 HWC -> float32 NCHW normalizado [0,1], resized a 640x640."""
        if frame_bgr.shape[0] != self.INPUT_SIZE or frame_bgr.shape[1] != self.INPUT_SIZE:
            resized = cv2.resize(frame_bgr, (self.INPUT_SIZE, self.INPUT_SIZE))
        else:
            resized = frame_bgr
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))   # HWC -> CHW
        tensor = np.expand_dims(tensor, axis=0)     # -> NCHW
        return tensor

    def _infer_raw(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Corre una inferencia cruda, devuelve el output0 sin procesar."""
        inp = self._preprocess(frame_bgr)
        out = self._compiled_model([inp])[self._output_layer]
        return out

    def infer(self, frame_bgr: np.ndarray) -> dict:
        """
        Detecta semaforo y flechas en un frame BGR.

        Returns:
            dict con:
                - semaforo: dict | None con x1, y1, x2, y2, conf, class
                - arrows: list de dicts con la misma estructura
        """
        empty = {"semaforo": None, "arrows": []}
        if not self.is_available or frame_bgr is None:
            return empty

        # Guardar dimensiones originales para escalar coordenadas de vuelta
        orig_h, orig_w = frame_bgr.shape[:2]

        out = self._infer_raw(frame_bgr)

        # out shape esperado: [1, 4+nc, na]. Para YOLO11 + 3 clases: [1, 7, 8400]
        if out.ndim != 3 or out.shape[0] != 1:
            return empty
        preds = out[0]  # (4+nc, na)
        nc = self._nc
        if preds.shape[0] < 4 + nc:
            return empty

        # transponer a (na, 4+nc)
        preds = preds.T
        boxes_xywh = preds[:, :4]      # (na, 4) cx, cy, w, h en pixeles 640x640
        class_scores = preds[:, 4:4+nc]  # (na, nc)

        # best class y score por anchor
        best_class_ids = np.argmax(class_scores, axis=1)
        best_scores = class_scores[np.arange(class_scores.shape[0]), best_class_ids]

        # filtrar por conf
        mask = best_scores >= self.conf_threshold
        if not np.any(mask):
            return empty

        boxes_xywh = boxes_xywh[mask]
        best_scores = best_scores[mask]
        best_class_ids = best_class_ids[mask]

        # xywh (centro) -> xyxy (esquinas) en pixeles 640x640
        cx, cy, w, h = boxes_xywh.T
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Escalar coordenadas de 640x640 al frame original
        scale_x = orig_w / self.INPUT_SIZE
        scale_y = orig_h / self.INPUT_SIZE
        xyxy[:, 0] *= scale_x  # x1
        xyxy[:, 1] *= scale_y  # y1
        xyxy[:, 2] *= scale_x  # x2
        xyxy[:, 3] *= scale_y  # y2

        # Clip a limites del frame original
        xyxy[:, 0] = np.clip(xyxy[:, 0], 0, orig_w - 1)
        xyxy[:, 1] = np.clip(xyxy[:, 1], 0, orig_h - 1)
        xyxy[:, 2] = np.clip(xyxy[:, 2], 0, orig_w - 1)
        xyxy[:, 3] = np.clip(xyxy[:, 3], 0, orig_h - 1)

        # NMS
        keep = self._nms(xyxy, best_scores, self.iou_threshold)
        xyxy = xyxy[keep]
        best_scores = best_scores[keep]
        best_class_ids = best_class_ids[keep]

        detections = {"semaforo": None, "arrows": []}
        for i in range(len(xyxy)):
            cid = int(best_class_ids[i])
            if cid >= len(self.class_names):
                continue
            cls_name = self.class_names[cid]
            det = {
                "x1": int(xyxy[i, 0]),
                "y1": int(xyxy[i, 1]),
                "x2": int(xyxy[i, 2]),
                "y2": int(xyxy[i, 3]),
                "confidence": float(best_scores[i]),
                "class": cls_name,
            }
            if cls_name == "semaforo":
                # si hay mas de uno, quedate con el de mayor score
                if (detections["semaforo"] is None
                        or det["confidence"] > detections["semaforo"]["confidence"]):
                    detections["semaforo"] = det
            elif cls_name in ("arrow_left", "arrow_right"):
                detections["arrows"].append(det)

        return detections

    @staticmethod
    def _nms(xyxy: np.ndarray, scores: np.ndarray, iou_thr: float) -> list:
        """NMS clasico. Devuelve indices de boxes a conservar."""
        if len(xyxy) == 0:
            return []
        x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            inds = np.where(iou <= iou_thr)[0]
            order = order[inds + 1]
        return keep


def is_openvino_available() -> bool:
    """True si el paquete openvino esta instalado y se puede importar."""
    try:
        import openvino  # noqa: F401
        return True
    except ImportError:
        return False
