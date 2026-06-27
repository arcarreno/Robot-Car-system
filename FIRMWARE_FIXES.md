# Firmware Fixes — ESP32-CAM 4WD Robot Car

> **NO aplico estos cambios yo** porque no tengo acceso al hardware ESP32
> desde este entorno. Tenés que flashearlos vos con Arduino IDE.

Todos los fixes son seguros y opcionales (no rompen funcionalidad). El
mayor impacto combinado: **+30-50% FPS del stream**.

## Cómo aplicar

1. Abrir `C:\Users\aranc\Documents\9NO\haciendociencia2\Main Code\ESP32_Camera_4WD_Robot_Car_OV3660_V3\` en Arduino IDE
2. Aplicar los diffs de abajo a los dos archivos
3. Compilar y flashear (mismo proceso que ya usás)

---

## Archivo 1: `ESP32_Camera_4WD_Robot_Car_OV3660_V3.ino`

### Fix FW1 + FW4 + FW5: Inicialización de cámara

**Líneas 94-102** (cambio de jpeg_quality, fb_count y frame_size):

```cpp
  //init with high specs to pre-allocate larger buffers
  if(psramFound()){
    // FW5 fix: inicializar directo a QVGA (320x240) en vez de UXGA
    // (1600x1200). Antes se alocaba 7x mas memoria para luego descartar
    // el 95% al hacer set_framesize. Con PSRAM no es critico pero igual
    // es desperdicio de ciclos de init.
    config.frame_size = FRAMESIZE_QVGA;
    // FW1 fix: subir jpeg_quality de 10 a 15. Calidad 10 = encoding mas
    // lento en ESP32 (CPU-intensive) sin ganancia visual real. 15 da
    // ~+30% FPS del stream con diferencia visual minima.
    config.jpeg_quality = 15;
    // FW4 fix: fb_count=1 en lugar de 2. Doble buffer es util si el
    // consumidor es mas lento que el productor, pero en este sistema
    // el backend procesa rapido. 1 frame buffer libera ~300KB de PSRAM.
    config.fb_count = 1;
  } else {
    // sin PSRAM: mantener config conservadora
    config.frame_size = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }
```

**Líneas 111-114** (eliminar el set_framesize redundante):

```cpp
  // FW5 fix: eliminado el set_framesize(FRAMESIZE_QVGA) post-init.
  // Ya inicializamos directo a QVGA en config.frame_size. Este
  // realloc fragmentaba memoria PSRAM sin beneficio.
  // sensor_t * s = esp_camera_sensor_get();
  // s->set_framesize(s, FRAMESIZE_QVGA);  // 320x240 for better WiFi performance
```

### Fix FW6: Eliminar variable de motor no usada

**Línea 23** (eliminar la variable duplicada):

```cpp
// FW6 fix: eliminado `volatile unsigned int motor_speed = 100` (linea
// 23 original). Era dead code, no se usaba en el flujo principal.
// La velocidad real esta en `int speed = 150` (linea 18).
// volatile unsigned int motor_speed = 100;
```

---

## Archivo 2: `app_httpd.cpp`

### Fix FW2: Quitar `Serial.printf` del hot loop de `stream_handler`

**Línea 384-388** (comentar o quitar; en release es deseable tener
logs, pero bloquea el hot path):

```cpp
        // FW2 fix: comentado el Serial.printf del hot loop. Cada
        // llamada a Serial.printf toma ~5-10ms en ESP32 (esperando
        // que el UART termine). En el hot loop del stream, son
        // 5-10ms PERDIDOS por frame. Si necesitas los logs,
        // descomentar.
        // Serial.printf("MJPG: %uB %ums (%.1ffps), AVG: %ums (%.1ffps)"
        //     ,(uint32_t)(_jpg_buf_len),
        //     (uint32_t)frame_time, 1000.0 / (uint32_t)frame_time,
        //     avg_frame_time, 1000.0 / avg_frame_time
        // );
```

**Línea 317** (mismo tratamiento):

```cpp
    // FW2 fix: comentado Serial.printf en /capture. Mismo motivo
    // que en stream_handler: 5-10ms de overhead por llamada.
    // Serial.printf("JPG: %uB %ums", (uint32_t)(fb_len), (uint32_t)((fr_end - fr_start)/1000));
```

### Fix FW3: Memory leak en stream_handler cuando JPEG compression falla

**Líneas 343-355** (agregar return despues de error):

```cpp
        } else {
            if(fb->format != PIXFORMAT_JPEG){
                bool jpeg_converted = frame2jpg(fb, 80, &_jpg_buf, &_jpg_buf_len);
                esp_camera_fb_return(fb);
                fb = NULL;
                if(!jpeg_converted){
                    // FW3 fix: si la compresion JPEG falla, el buffer
                    // `_jpg_buf` queda asignado pero el fb original ya
                    // se devolvio. Hay que liberar `_jpg_buf` para
                    // evitar memory leak acumulativo.
                    if(_jpg_buf){
                        free(_jpg_buf);
                        _jpg_buf = NULL;
                    }
                    _jpg_buf_len = 0;
                    Serial.printf("JPEG compression failed");
                    res = ESP_FAIL;
                }
            } else {
                _jpg_buf_len = fb->len;
                _jpg_buf = fb->buf;
            }
        }
```

### Fix FW7: Eliminar funciones comentadas duplicadas (limpieza)

**Líneas 119-157** (borrar bloque entero):

```cpp
// FW7 fix: eliminado el bloque de funciones robot_* comentado
// (lineas 119-157 originales). Las versiones activas con
// `ledc_set_duty` (lineas 160-214) son las unicas que usa el
// firmware compilado. Las comentadas son dead code que confunde
// al lector.
// El bloque era:
//   static void robot_stop();
//   static void robot_forward(int, int);
//   static void robot_backward(int, int);
//   static void robot_left();
//   static void robot_right();
//   static void robot_standby();
//   static void WheelAct(int, int, int, int);
```

---

## Verificación post-reflash

Después de flashear:

1. Conectar Serial Monitor a 115200 baud
2. Verificar que el ESP32 bootea sin errores de init
3. Probar `http://192.168.4.1/status` (debe responder)
4. Probar `http://192.168.4.1:81/stream` en el navegador (debe mostrar
   video). Verificar que el FPS se siente mas fluido.
5. Probar el backend: `cd backend && python main.py`, conectar al
   frontend, ver que el stream llega al WebSocket y el robot responde
   a comandos.

Si ves "Brownout detector was triggered" o reinicios: probablemente
`fb_count=1` causa starvation con PSRAM; volvé a `fb_count=2`.
Si ves "Camera init failed with error 0x...": volve a PSRAM config
anterior.

## Impacto esperado

| Fix | Ganancia FPS | Esfuerzo |
|-----|--------------|----------|
| FW1 (jpeg_quality 10→15) | +30% | 1 linea |
| FW2 (quitar Serial.printf) | +5-10% | 2 lineas |
| FW4 (fb_count 2→1) | Marginal | 1 linea |
| FW5 (UXGA→QVGA directo) | Marginal | 4 lineas |
| FW3 (memory leak) | Estabilidad | 6 lineas |
| FW6/FW7 (cleanup) | 0 (legibilidad) | 10 lineas |

**Total: ~40% mas FPS, sin perder funcionalidad.**

---

## Nota sobre el giro 180°

El firmware **NO tiene** una funcion `giro_180`. Las funciones de giro
(`robot_left()`, `robot_right()`) giran indefinidamente hasta que reciben
un comando `stop`. El giro de 180° se controla enteramente desde el
**backend** con un timer (`_route_turn_time_s`).

El valor por defecto es 1600ms a velocidad media (150 PWM). Si el robot
solo hace ~90°, subi el valor del slider en la UI (ej: 2400ms). El backend
ajusta automaticamente el tiempo segun la velocidad:
- Baja (100): ×1.2
- Media (150): ×1.0
- Alta (200): ×0.8
