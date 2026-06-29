# Robot Car Control System

Sistema de control para ESP32-CAM 4WD Robot Car con visión por computadora
(YOLO + HSV híbrido).

## Arquitectura

```
┌─────────────────────────────────────────────────────────────┐
│                    ARQUITECTURA                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Laptop (Frontend + Backend)                               │
│   ┌─────────────────────┐    ┌─────────────────────────┐    │
│   │  React (Puerto 5173)│◄──►│  FastAPI (Puerto 8000)   │    │
│   │  WebSocket Client   │    │  WebSocket + OpenCV      │    │
│   └─────────────────────┘    └───────────┬─────────────┘    │
│                                          │                  │
│                                          │ HTTP              │
│                                          ▼                  │
│   ┌─────────────────────────────────────────────────────┐    │
│   │  ESP32-CAM Robot (Puerto 80: comandos, 81: stream)  │    │
│   └─────────────────────────────────────────────────────┘    │
│                                                              │
│   NOTA: el frontend NUNCA se conecta directo al puerto 81.  │
│   El backend consume el stream MJPEG y emite frames JPEG     │
│   individuales por WebSocket (campo `frame_b64`). Esto       │
│   evita saturar al ESP32-CAM (1 cliente maximo en :81).    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Requisitos

- Python 3.10+ (3.14 todavia no es estable a la fecha)
- Node.js 18+
- ESP32-CAM 4WD Robot Car Kit (LAFVIN con camara OV3660)

## Configuración por Variables de Entorno

### Backend (`backend/.env`)

```bash
# ESP32
ESP32_IP=192.168.4.1

# Backend server
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000

# YOLO skip frames (0=all, 1=every 2nd, 2=every 3rd)
YOLO_SKIP_FRAMES=1

# Frame JPEG throttle (0=always, 3=every 4th frame)
FRAME_SKIP=3

# OpenVINO device preference (AUTO, GPU, CPU)
OPENVINO_DEVICE=AUTO
```

### Frontend (`frontend/.env`)

```bash
VITE_ESP32_URL=http://192.168.4.1
VITE_BACKEND_WS=ws://192.168.4.2:8000/ws
```

Copiar `.env` a `.env.local` para overrides sin afectar el repositorio.

## Instalación

### Backend (Python)

```bash
cd backend
pip install -r requirements.txt
```

### Frontend (React)

```bash
cd frontend
npm install
```

## Ejecución

### 1. Iniciar Backend

```bash
cd backend
python main.py
```

El servidor backend estará disponible en `http://localhost:8000`.

### 2. Iniciar Frontend (en otra terminal)

```bash
cd frontend
npm run dev
```

El frontend estará disponible en `http://localhost:5173`.

### 3. Conectar al Robot

1. Conectar tu laptop/PC a la red WiFi del robot: `ESP32-CAM Robot`
2. Abrir el navegador en `http://localhost:5173`
3. Click en "Conectar" y verificar que el estado muestre "Conectado"

> **Nota**: Si el robot ya tiene el firmware flasheado, no es necesario usar Arduino IDE. Solo instalar dependencias y ejecutar. El flasheo del firmware es una sola vez por robot. Ver `FIRMWARE_FIXES.md` solo si se necesita re-flashear.

## Características

- **Control Manual**: Botones (mouse) o teclado (WASD/flechas/Space) para mover el robot
- **Video en Vivo**: Stream procesado por el backend (YOLO + bounding boxes)
- **Detección de Semáforos**: Pipeline hibrido **YOLO (caja) + HSV (color)**
  detecta rojo, amarillo y verde
- **Detección de Flechas**: YOLO detecta flecha izquierda/derecha dentro del
  circulo verde del semaforo
- **Modos autonomos**:
  - **Ruta Autonoma**: Ida y vuelta a una distancia configurable (1-10m)
  - **Ruta Continua**: El robot avanza siempre, respeta semaforos, gira
    segun la flecha detectada
- **Notificaciones**: Toasts en tiempo real via Sileo
- **Auto-Comando**: Detiene el robot automaticamente si detecta rojo/amarillo
- **Watchdog**: Si la camara deja de enviar frames, el robot se detiene

## Estructura del Proyecto

```
robot-car-system/
├── backend/                              # FastAPI + WebSocket + OpenCV
│   ├── main.py                           # Entry point: FastAPI app + lifespan
│   ├── config.py                         # Config centralizada (env vars)
│   ├── requirements.txt
│   ├── clients/
│   │   └── esp32_client.py               # HTTP client al ESP32 (comandos)
│   ├── core/                             # Logica de deteccion
│   │   ├── color_detector.py             # YOLO + HSV hibrido
│   │   ├── yolo_detector.py              # Wrapper de Ultralytics YOLO
│   │   ├── mjpeg_parser.py               # Parser del stream MJPEG del ESP32
│   │   └── state_machine.py              # Maquina de estados del robot
│   ├── threads/
│   │   ├── mjpeg_thread.py               # Hilo que consume el stream :81
│   │   └── process_thread.py             # Hilo de OpenCV (YOLO + HSV + WS)
│   ├── models/
│   │   ├── frame_result.py               # DTO del resultado por frame
│   │   └── semaforo_yolo.pt              # Modelo YOLO entrenado
│   ├── utils/
│   │   ├── anotador.py                   # Anota frames con bounding boxes
│   │   └── circuit_breaker.py
│   └── tests/                            # 101 tests unitarios (unittest)
│
├── frontend/                             # React 18 + Vite
│   ├── src/
│   │   ├── main.jsx                      # Entry point
│   │   ├── App.jsx                       # Componente raiz + toaster
│   │   ├── App.css                       # Estilos globales (incluye .video-placeholder)
│   │   ├── config.js                     # URLs (ESP32, backend)
│   │   ├── context/
│   │   │   ├── RobotContext.jsx          # Estado global (WS, comandos, conexion)
│   │   │   └── ThemeContext.jsx          # Tema claro/oscuro
│   │   └── components/
│   │       ├── Tabs.jsx                  # Tabs ARIA con navegacion por teclado
│   │       ├── ControlManual.jsx         # Control por mouse/botones
│   │       ├── RutaAutonoma.jsx          # Ida y vuelta autonoma
│   │       ├── RutaContinua.jsx          # Avance continuo con semaforos
│   │       ├── ConnectionPanel.jsx       # Modal de conexion ESP32
│   │       ├── KeyboardToggle.jsx        # Toggle del modo teclado
│   │       ├── SystemIndicators.jsx      # LEDs de estado (backend/esp32/kb)
│   │       ├── ThemeToggle.jsx           # Switch dia/noche
│   │       └── EarthLoader.jsx
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── train_yolo.py                         # CLI para entrenar el modelo YOLO
├── test_yolo.py                          # CLI para probar el modelo entrenado
├── convert_labels.py                     # Convierte export de Roboflow a formato YOLO
├── polygon_to_bbox.py                    # Convierte labels poligono a bbox
├── YOLOREADME.md                         # Guia paso a paso de reentrenamiento
└── README.md
```

## Comandos Disponibles

| Comando   | Descripcion              |
|-----------|--------------------------|
| `go`      | Avanzar                  |
| `back`    | Retroceder               |
| `left`    | Girar a la izquierda     |
| `right`   | Girar a la derecha       |
| `stop`    | Detener                  |
| `ledon`   | Encender LED             |
| `ledoff`  | Apagar LED               |

## Deteccion de Semaforos

El sistema detecta automaticamente colores de semaforo usando un pipeline
hibrido:

1. **YOLO** (YOLO11n) detecta la bounding box del semaforo y de las flechas
2. **HSV** lee el color (rojo/amarillo/verde) dentro del bounding box

Los rangos HSV estan calibrados para ESP32-CAM con AWB shift +15 (ver
`backend/config.py:COLOR_RANGES`).

Estados:
- **ROJO**: Robot se detiene automaticamente
- **AMARILLO**: Robot desacelera gradualmente y luego se detiene (1s)
- **VERDE**: Robot avanza (o gira si hay flecha)

**Performance**: YOLO se corre cada 2 frames (configurable via
`YOLO_SKIP_FRAMES`) para mantener ~20 FPS. El frame intermedio usa solo
HSV, que es mucho mas rapido.

## Deteccion YOLO (entrenamiento)

Para reentrenar el modelo con tus propios datos, ver
[`YOLOREADME.md`](./YOLOREADME.md). El pipeline:

1. Recolectar y etiquetar imagenes en [Roboflow](https://roboflow.com)
2. Exportar en formato YOLO v11 (compatible con v8)
3. `python convert_labels.py <carpeta_roboflow>` para aplanar el dataset
4. `python train_yolo.py --epochs 50 --device cpu` para entrenar
5. El modelo entrenado se copia automaticamente a `backend/models/`

## API Endpoints

### HTTP

- `GET /` - Sirve el frontend compilado
- `GET /api/status` - Estado del sistema
- `WS  /ws` - WebSocket bidireccional para frames y comandos

### Mensajes WebSocket

**Cliente → Servidor:**
- `{action: "comando", comando: "go"}` - Mover robot
- `{action: "tecla_up", comando: "go"}` - Soltar tecla
- `{action: "iniciar_ruta", distancia, velocidad, giro_ms}` - Ruta autonoma
- `{action: "detener_ruta"}`
- `{action: "iniciar_continua", speed}` - Ruta continua
- `{action: "detener_continua"}`

**Servidor → Cliente:**
- `{frame_b64, detected_color, detected_arrow, state, fps, ...}` - Frame procesado
- `{type: "heartbeat"}` - Keep-alive
- `{type: "route_status", status, progress, phase}` - Estado de ruta autonoma
- `{type: "continuous_status", status}` - Estado de ruta continua

## Troubleshooting

### El robot no gira 180° (giro incompleto o excesivo)

El giro de 180° se controla por **timer** en el backend (`giro_ms`), no por
encoders. El firmware solo tiene `robot_right()` que gira indefinidamente.
Para calibrar:

1. Abrir la pestaña "Ruta Autonoma" en el frontend
2. Ajustar el slider "Giro 180°":
   - **Si no llega a 180°**: subir el valor (ej: 2000ms)
   - **Si pasa de 180°**: bajar el valor (ej: 1200ms)
3. El valor se guarda automáticamente en `localStorage`
4. El tiempo se ajusta según la velocidad:
   - Baja (100 PWM): ×1.2 (más lento, necesita más tiempo)
   - Media (150 PWM): ×1.0 (baseline)
   - Alta (200 PWM): ×0.8 (más rápido, necesita menos tiempo)

### Error de conexion con ESP32
- Verificar que el robot este encendido y emitiendo WiFi
- Verificar que estes conectado a la red `ESP32-CAM Robot`
- Probar `http://192.168.4.1/status` en el navegador

### Video no carga
- El frontend NO consume el stream directo. Si el backend esta conectado
  pero no hay frames, el problema esta en `backend/threads/mjpeg_thread.py`
- Verificar que `http://192.168.4.1:81/stream` funcione directo en el navegador

### Backend no inicia
- Verificar que Python 3.10+ este instalado (`python --version`)
- Verificar dependencias: `pip install -r backend/requirements.txt`

### YOLO no detecta nada
- Verificar que `backend/models/semaforo_yolo.pt` exista (~20MB)
- Probar `python test_yolo.py --image <ruta>` con una imagen conocida
- Si `YOLO_ENABLED=False` en `config.py`, solo se usa HSV (fallback)

## Autores

Angel Armando Carreño Gonzalez angel.carreno@alumno.buap.mx

Ricardo Alvarez González ricardo.alvarez@correo.buap.mx
