"""
Deteccion de semaforo y flechas usando YOLO11n.

Usa un modelo custom entrenado para detectar:
  - semaforo (bounding box del housing)
  - arrow_left (flecha izquierda)
  - arrow_right (flecha derecha)

Flujo:
  1. YOLO detecta el semaforo -> bounding box
  2. Se recorta el bounding box del frame
  3. HSV lee el color dentro del recorte
  4. YOLO detecta flechas -> determina direccion

Backends soportados (auto-detect en este orden):
  1) OpenVINO iGPU (si hay Intel iGPU y el modelo .xml/.bin existe)
  2) OpenVINO CPU  (fallback si no hay iGPU o falla la compilacion)
  3) PyTorch CPU  (fallback final si OpenVINO no esta disponible o falla)

El backend elegido queda expuesto en self.backend_name para debugging
("openvino_gpu", "openvino_cpu", "pytorch").
"""

import cv2
import numpy as np
from typing import Optional, List, Dict
import os


class YoloDetector:
    def __init__(self, model_path: str = "models/semaforo_yolo.pt",
                 conf_threshold: float = 0.25,
                 openvino_xml: Optional[str] = None,
                 openvino_device: str = "AUTO",
                 class_names: Optional[List[str]] = None):
        """
        Inicializa el detector YOLO con auto-detect de backend.

        Args:
            model_path: ruta al modelo .pt (usado si OpenVINO no aplica).
            conf_threshold: umbral de confianza minima (0.0 - 1.0).
            openvino_xml: ruta al .xml de OpenVINO. Si None, usa el default
                de config.OPENVINO_MODEL_XML.
            openvino_device: "AUTO" | "GPU" | "CPU".
            class_names: orden de clases del modelo. Si None, usa
                config.YOLO_CLASS_NAMES.
        """
        from config import (
            OPENVINO_MODEL_XML, OPENVINO_DEVICE_PREFERENCE, YOLO_CLASS_NAMES,
        )
        self.conf_threshold = conf_threshold
        self.class_names = class_names or YOLO_CLASS_NAMES

        # B5 fix: validar que class_names tiene 3 clases (semaforo, arrow_right, arrow_left)
        if len(self.class_names) != 3:
            print(f"[YOLO] WARNING: class_names tiene {len(self.class_names)} clases, "
                  f"se esperaban 3. Las detecciones pueden ser incorrectas.")
        if self.class_names[0] != "semaforo":
            print(f"[YOLO] WARNING: primera clase es '{self.class_names[0]}', "
                  f"se esperaba 'semaforo'. Verificar orden del modelo.")

        # Backend activo. Solo uno de los tres quedara instanciado.
        self._pytorch_model = None     # ultralytics.YOLO
        self._openvino_backend = None  # OpenVinoYoloBackend

        self._model_path = model_path
        self.backend_name = "none"  # "openvino_gpu" | "openvino_cpu" | "pytorch"

        # FASE 4 (B3): cache del ultimo resultado para soportar skip de frames.
        # Cuando color_detector llama con run_yolo=False, le devolvemos esto.
        self._last_result = {
            "semaforo": None, "arrows": [], "overlay": None,
        }

        # Resolver path del .xml de OpenVINO (default: el de config)
        ov_xml = openvino_xml or str(OPENVINO_MODEL_XML)
        ov_dev = openvino_device or OPENVINO_DEVICE_PREFERENCE

        # Intentar OpenVINO primero (es el camino rapido cuando aplica)
        if self._try_load_openvino(ov_xml, ov_dev):
            return

        # Si no, caer a PyTorch
        if self._try_load_pytorch(model_path):
            return

        # Ultimo recurso: nada cargado, detect() devolvera vacio
        print("[YOLO] Ningun backend disponible. detect() devolvera vacio.")

    # ---------------------------------------------------------------------
    # Carga de backends
    # ---------------------------------------------------------------------

    def _try_load_openvino(self, xml_path: str, device: str) -> bool:
        """Intenta cargar el modelo OpenVINO. Devuelve True si tuvo exito."""
        if not os.path.exists(xml_path):
            print(f"[YOLO] OpenVINO: modelo no encontrado en {xml_path} "
                  f"(siguiendo con PyTorch).")
            return False
        try:
            from inference.openvino_backend import is_openvino_available
            if not is_openvino_available():
                print("[YOLO] OpenVINO: paquete no instalado "
                      "(siguiendo con PyTorch).")
                return False
        except ImportError:
            print("[YOLO] OpenVINO: modulo inference.openvino_backend no "
                  "importable (siguiendo con PyTorch).")
            return False

        try:
            from inference.openvino_backend import OpenVinoYoloBackend
            backend = OpenVinoYoloBackend(
                model_xml_path=xml_path,
                class_names=self.class_names,
                conf_threshold=self.conf_threshold,
                device_preference=device,
            )
            backend.load()
            backend.warmup(n=3)
            self._openvino_backend = backend
            self.backend_name = f"openvino_{backend.device_used.lower()}"
            print(f"[YOLO] Backend activo: {self.backend_name}")
            return True
        except Exception as e:
            print(f"[YOLO] OpenVINO: fallo cargando ({e}). "
                  f"Siguiendo con PyTorch.")
            return False

    def _try_load_pytorch(self, model_path: str) -> bool:
        """Intenta cargar el modelo PyTorch/ultralytics. Devuelve True si OK."""
        if not os.path.exists(model_path):
            print(f"[YOLO] PyTorch: modelo no encontrado en {model_path}.")
            return False
        try:
            from ultralytics import YOLO
            self._pytorch_model = YOLO(model_path)
            self.backend_name = "pytorch"
            print(f"[YOLO] Modelo PyTorch cargado: {model_path}")
            return True
        except ImportError:
            print("[YOLO] ultralytics no instalado.")
            return False
        except Exception as e:
            print(f"[YOLO] Error cargando PyTorch: {e}")
            return False

    # ---------------------------------------------------------------------
    # API publica
    # ---------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """True si algun backend esta cargado y operativo."""
        return self._openvino_backend is not None or self._pytorch_model is not None

    def detect(self, frame: np.ndarray, run_yolo: bool = True) -> Dict:
        """
        Detecta semaforo y flechas en un frame.

        Args:
            frame: imagen BGR de OpenCV
            run_yolo: si False, devuelve el ultimo resultado cacheado sin
                ejecutar inferencia. Usado por FASE 4 (B3) skip de frames.

        Returns:
            dict con:
                - semaforo: dict | None con bbox y color
                - arrows: list de dicts con direccion y bbox
                - overlay: ya no se devuelve (P1 fix). El dibujo de boxes lo
                  hace color_detector; YoloDetector solo devuelve detecciones.
        """
        if not self.is_available or frame is None:
            return {"semaforo": None, "arrows": [], "overlay": None}

        # FASE 4 (B3): si run_yolo=False, devolver COPIA del cache.
        # B9 fix: devolver dict nuevo para que el caller no pueda
        # corromper el cache mutando el dict devuelto.
        if not run_yolo:
            return {
                "semaforo": self._last_result["semaforo"],
                "arrows": [dict(a) for a in self._last_result["arrows"]],
                "overlay": None,
            }

        # Despachar al backend que se haya cargado
        if self._openvino_backend is not None:
            detections = self._openvino_backend.infer(frame)
            detections["overlay"] = None  # P1 fix: no overlay aca
        else:
            # Camino PyTorch (comportamiento original)
            detections = self._detect_pytorch(frame)

        # Cachear para futuras llamadas con run_yolo=False
        self._last_result = detections

        return detections

    def _detect_pytorch(self, frame: np.ndarray) -> Dict:
        """Inferencia via ultralytics (comportamiento original)."""
        results = self._pytorch_model(frame, conf=self.conf_threshold, verbose=False)
        detections = {"semaforo": None, "arrows": [], "overlay": None}

        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                cls_name = result.names[cls_id]

                bbox = {"x1": int(x1), "y1": int(y1),
                        "x2": int(x2), "y2": int(y2),
                        "confidence": conf, "class": cls_name}

                if cls_name == "semaforo":
                    detections["semaforo"] = bbox
                elif cls_name in ("arrow_left", "arrow_right"):
                    detections["arrows"].append(bbox)

        return detections

    def get_arrow_direction(self, detections: Dict) -> Optional[str]:
        """
        Determina la direccion de la flecha basada en las detecciones de YOLO.

        Args:
            detections: resultado de detect()

        Returns:
            "left" | "right" | None
        """
        for arrow in detections.get("arrows", []):
            if arrow["class"] == "arrow_left":
                return "left"
            elif arrow["class"] == "arrow_right":
                return "right"
        return None
