"""
Configuracion centralizada del backend.

Soporta variables de entorno via .env (opcional).
"""

import os

# Cargar .env si python-dotenv esta disponible
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =============================================================================
# ESP32
# =============================================================================
ESP32_IP = os.getenv("ESP32_IP", "192.168.4.1")
ESP32_URL = f"http://{ESP32_IP}"
ESP32_STREAM = f"{ESP32_URL}:81/stream"
ESP32_API = ESP32_URL  # puerto 80

# =============================================================================
# Backend
# =============================================================================
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))

# =============================================================================
# Rangos HSV para deteccion de colores (OpenCV: H=0-179, S=0-255, V=0-255)
# =============================================================================
# IMPORTANTE: el amarillo PURO del semaforo React (#facc15) tiene H=24.
# El AWB del ESP32-CAM en ambientes con luz fluorescente puede shiftear
# el H entre 20 y 32. Si el rango del amarillo llega hasta H=50, cualquier
# amarillo shifteado a H 35-50 cae en el rango del VERDE y el robot
# avanza en vez de frenar. Por eso el amarillo va solo hasta H=38
# (gap de 12 con el verde que arranca en H=50).
COLOR_RANGES = {
    # Rojo puro: H 0-10 y 170-180 (gap de 5 con amarillo para evitar conflicto)
    "red_low":   ((0,   80,  80),   (10,  255, 255)),
    "red_high":  ((170, 80,  80),   (180, 255, 255)),
    # Amarillo: H 15-38 (cubre puro H=24 + AWB azulado hasta H=32; gap
    # de 12 con verde para evitar confusion amarillo<->verde)
    "yellow":    ((15,  100, 100),  (38,  255, 255)),
    # Verde (TODOS los verdes): H 50-90, S 80-255, V 40-255
    # V min bajado de 50 a 40 para cubrir green-700 (#15803d, V=61) con
    # margen ante AWB shift del ESP32-CAM que puede reducir brillo.
    "green":     ((50,  80,  40),   (90,  255, 255)),
}

# Area minima en pixeles — 800 para detectar semaforo a 15-30cm
COLOR_MIN_AREA = 800

# Area minima para flecha
ARROW_MIN_AREA = 200

# =============================================================================
# YOLO - Deteccion de semaforo y flechas
# =============================================================================
YOLO_MODEL_PATH = "models/semaforo_yolo.pt"  # relativo a backend/
YOLO_CONF_THRESHOLD = 0.4  # B2 fix: subido de 0.25 a 0.4 para reducir falsos positivos
YOLO_ENABLED = True  # usar YOLO para deteccion (False = solo HSV)
# FASE 4 (B3): correr YOLO cada N frames en lugar de cada uno. YOLO11n en CPU
# a 320x240 tarda ~80-150ms; alternar con HSV puro en frames intermedios da
# ~2-3x FPS total sin perder precision significativa (la escena cambia poco en
# 200ms). N=0 desactiva el skip (corre en TODOS los frames, comportamiento
# original). N=1 = cada 2do frame, N=2 = cada 3er frame. Recomendado: 1.
YOLO_SKIP_FRAMES = int(os.getenv("YOLO_SKIP_FRAMES", "1"))

# =============================================================================
# Streaming WS - Throttle del frame JPEG
# =============================================================================
# El frame anotado (JPEG base64) pesa ~10-20KB. Si lo mandamos en cada
# WS message, satura el ancho de banda y bloquea la entrega de la
# metadata de color/estado (que es lo que el frontend necesita ver
# rapido). Throttle: el JPEG se envia cada N frames. La metadata
# (detected_color, detected_arrow, state, fps) se sigue mandando a
# maxima frecuencia. N=0 desactiva el throttle (envia siempre).
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "3"))  # default: video a ~3-4 FPS a 15 FPS de proceso

# =============================================================================
# OpenVINO - Backend opcional para acelerar inferencia YOLO
# =============================================================================
# Si el modelo .xml/.bin existe y openvino esta instalado, el YoloDetector
# prefiere OpenVINO sobre PyTorch. AUTO = GPU si hay Intel iGPU, sino CPU.
# PYTHONHASHSEED no afecta. Si no hay modelo exportado o falla la carga, cae
# a PyTorch automaticamente (sin lanzar excepcion).
from pathlib import Path as _Path
_MODELS_DIR = _Path(__file__).parent / "models"
OPENVINO_MODEL_XML = _MODELS_DIR / "semaforo_yolo_openvino_model" / "semaforo_yolo.xml"
OPENVINO_DEVICE_PREFERENCE = os.getenv("OPENVINO_DEVICE", "AUTO")  # AUTO | GPU | CPU
YOLO_CLASS_NAMES = ["semaforo", "arrow_left", "arrow_right"]

# =============================================================================
# Watchdog
# =============================================================================
# Si no se procesa un frame por este tiempo -> STOP + IDLE
WATCHDOG_TIMEOUT = 3.0  # segundos

# =============================================================================
# MJPEG Parser
# =============================================================================
MJPEG_TIMEOUT = 10       # segundos timeout de conexion
MJPEG_CHUNK_SIZE = 1024  # bytes por lectura

# =============================================================================
# Threading
# =============================================================================
QUEUE_MAXSIZE = 2  # Tamaño maximo de colas (descarta frames viejos)

# =============================================================================
# Depth Estimation - MiDaS Small v2.1 (OpenVINO)
# =============================================================================
# Estimacion de profundidad monocular para deteccion de obstaculos.
# MiDaS output es depth relativo (no metric). Se calibra con formula:
#   distancia_metros = K / (depth_value - offset)^P
DEPTH_ENABLED = os.getenv("DEPTH_ENABLED", "true").lower() == "true"
DEPTH_MODEL_PATH = os.getenv("DEPTH_MODEL_PATH", str(_MODELS_DIR / "midas_small" / "MiDaS_small.xml"))
DEPTH_SKIP_FRAMES = int(os.getenv("DEPTH_SKIP_FRAMES", "2"))  # cada 3er frame (~6-7 FPS)
DEPTH_DEVICE = os.getenv("DEPTH_DEVICE", "CPU")  # CPU / GPU / AUTO

# Bilateral filter (preserva bordes de obstaculos while suaviza ruido)
DEPTH_BILATERAL_D = int(os.getenv("DEPTH_BILATERAL_D", "7"))              # diametro vecindario (奇数, 5-9)
DEPTH_BILATERAL_SIGMA_COLOR = float(os.getenv("DEPTH_BILATERAL_SIGMA_COLOR", "50"))  # sigma color (mayor = mas suavizado)
DEPTH_BILATERAL_SIGMA_SPACE = float(os.getenv("DEPTH_BILATERAL_SIGMA_SPACE", "50"))  # sigma espacio (mayor = mas rango)

# Calibracion MiDaS → metros (ajustar con calibracion real)
# offset mas alto = distancias mas largas (empuja valores hacia atras)
MIDAS_CALIBRATION_K = float(os.getenv("MIDAS_CALIBRATION_K", "99997.1"))
MIDAS_CALIBRATION_P = float(os.getenv("MIDAS_CALIBRATION_P", "1.94"))
MIDAS_CALIBRATION_OFFSET = float(os.getenv("MIDAS_CALIBRATION_OFFSET", "-335.7"))  # calibrado: 0.3m+0.5m, raw=368.6→0.3m, raw=205.5→0.5m

# Zonas de obstaculo (coordina con OBSTACLE_STOP_THRESHOLD)
DEPTH_ZONE_DANGER = float(os.getenv("DEPTH_ZONE_DANGER", "0.4"))      # frenar inmediatamente
DEPTH_ZONE_CAUTION = float(os.getenv("DEPTH_ZONE_CAUTION", "0.8"))    # reducir velocidad
DEPTH_ZONE_PRE_CAUTION = float(os.getenv("DEPTH_ZONE_PRE_CAUTION", "1.2"))  # monitorear

# Obstaculos
OBSTACLE_STOP_THRESHOLD = float(os.getenv("OBSTACLE_STOP_THRESHOLD", "0.8"))  # metros
OBSTACLE_TURN_TIME = float(os.getenv("OBSTACLE_TURN_TIME", "1.0"))  # segundos para girar
OBSTACLE_RESUME_FACTOR = float(os.getenv("OBSTACLE_RESUME_FACTOR", "2.0"))  # reanuda si distancia > threshold * factor

# Post-colision: backup antes de girar (OBSOLETO — se usa escaneo)
OBSTACLE_BACKUP_TIME = float(os.getenv("OBSTACLE_BACKUP_TIME", "0.8"))  # segundos retrocediendo (OBSOLETO)
OBSTACLE_COLLISION_DISTANCE = float(os.getenv("OBSTACLE_COLLISION_DISTANCE", "0.2"))  # metros — colisión confirmada
OBSTACLE_COLLISION_BACKUP_TIME = float(os.getenv("OBSTACLE_COLLISION_BACKUP_TIME", "1.5"))  # segundos retrocediendo post-colisión (OBSOLETO)

# Escaneo de obstáculos (nuevo comportamiento — reemplaza BACKUP)
# Cuando se detecta un obstáculo, el robot escanea el ambiente en vez de retroceder
# SCAN_180: escanea 180° (izquierda + derecha) en 2 segundos
# SCAN_360: escanea 360° en 4 segundos si no hay salida en 180°
# FREE_DISTANCE: distancia mínima para considerar un camino "libre"
OBSTACLE_SCAN_180_TIME = float(os.getenv("OBSTACLE_SCAN_180_TIME", "2.0"))  # segundos para escanear 180°
OBSTACLE_SCAN_360_TIME = float(os.getenv("OBSTACLE_SCAN_360_TIME", "4.0"))  # segundos para escanear 360°
OBSTACLE_FREE_DISTANCE = float(os.getenv("OBSTACLE_FREE_DISTANCE", "0.3"))  # metros — distancia mínima libre para avanzar

# Slowdown: reducir velocidad al acercarse
OBSTACLE_SLOWDOWN_DISTANCE = float(os.getenv("OBSTACLE_SLOWDOWN_DISTANCE", "1.5"))  # metros — reducir velocidad
OBSTACLE_SPEED_REDUCTION = float(os.getenv("OBSTACLE_SPEED_REDUCTION", "0.5"))  # factor de reducción (50%)

# TTC: Time-to-Collision (basado en velocidad de aproximacion)
OBSTACLE_TTC_STOP = float(os.getenv("OBSTACLE_TTC_STOP", "2.0"))  # segundos — detener si TTC < este valor
OBSTACLE_TTC_SLOWDOWN = float(os.getenv("OBSTACLE_TTC_SLOWDOWN", "4.0"))  # segundos — reducir velocidad si TTC < este valor
OBSTACLE_TTC_BACKUP_MIN = float(os.getenv("OBSTACLE_TTC_BACKUP_MIN", "0.6"))  # segundos — backup minimo (0.3s no alcanzaba)
OBSTACLE_TTC_BACKUP_MAX = float(os.getenv("OBSTACLE_TTC_BACKUP_MAX", "1.5"))  # segundos — backup maximo
OBSTACLE_TTC_BACKUP_SCALE = float(os.getenv("OBSTACLE_TTC_BACKUP_SCALE", "0.4"))  # factor de escalado del backup
