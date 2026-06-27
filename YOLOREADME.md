# Entrenamiento YOLO - Semaforo Digital

Guia paso a paso para reentrenar el modelo YOLO del sistema con tus
propias imagenes.

## Paso a paso para entrenar el modelo

### Paso 1: Subir fotos a Roboflow

1. Ir a https://roboflow.com
2. Crear cuenta gratuita
3. Crear proyecto: "Semaforo Digital"
4. Subir las fotos de entrenamiento
5. Etiquetar cada foto con 3 clases:

| Clase | Que etiquetar |
|-------|---------------|
| `semaforo` | Bounding box del housing completo (rectangulo gris con 3 circulos) |
| `arrow_left` | Bounding box del circulo verde oscuro con flecha ↙ (solo cuando aparece) |
| `arrow_right` | Bounding box del circulo verde oscuro con flecha ↗ (solo cuando aparece) |

**Tips para etiquetar:**
- El semaforo debe incluir los 3 circulos (rojo, amarillo, verde)
- Las flechas solo aparecen cuando el verde esta activo
- Etiquetar desde distintos angulos y distancias
- Minimo 50 fotos etiquetadas para buenos resultados

### Paso 2: Exportar dataset de Roboflow

1. En Roboflow, ir a "Generate"
2. Seleccionar formato: **YOLO v11** (o v8, mismo formato de labels)
3. Descargar zip
4. Extraer en una carpeta (ej: `<ruta/al/dataset>`)

### Paso 3: Convertir labels (y opcionalmente polygon→bbox)

Si exportaste en formato bbox (YOLO v11/v8 estandar):

```bash
cd <repo-root>
python convert_labels.py "<ruta/al/dataset>"
```

Esto copia imagenes y labels a `yolo_dataset/` con la estructura
train/val que espera Ultralytics. Valida que `labels_dir` exista
(sino, el script aborta con un error claro en vez de copiar imagenes
sueltas silenciosamente).

Si exportaste en formato poligono (segmentacion), primero convierte
las labels a bbox:

```bash
# In-place sobre train/labels/
python polygon_to_bbox.py --input "<ruta/al/dataset>/train/labels"

# O a una carpeta separada
python polygon_to_bbox.py --input "<ruta/al/dataset>/train/labels" --output "<ruta/al/dataset>/train/labels_bbox"
```

Despues continua con `convert_labels.py` apuntando a la carpeta de
labels convertidos.

### Paso 4: Entrenar modelo

```bash
pip install ultralytics
python train_yolo.py
```

Para customizar hiperparametros:

```bash
# Ver todas las opciones
python train_yolo.py --help

# Ejemplo: 100 epochs en GPU
python train_yolo.py --epochs 100 --device 0

# Ejemplo: batch 16, learning rate bajo
python train_yolo.py --epochs 80 --batch 16 --lr0 0.005
```

**Tiempo estimado:** 30-60 minutos en CPU, ~5 minutos en GPU

### Paso 5: Probar modelo

```bash
# Probar con una imagen
python test_yolo.py --image <ruta/a/imagen.jpg>

# Probar con una carpeta (default: yolo_dataset/images/val)
python test_yolo.py --folder yolo_dataset/images/val

# Probar con umbral de confianza custom
python test_yolo.py --folder yolo_dataset/images/val --conf 0.4
```

### Paso 6: Integrar en el backend

El script `train_yolo.py` copia automaticamente los modelos finales
(`best.pt` y `best.onnx`) a `backend/models/`. Solo reinicia el
backend para que tome el modelo nuevo.

## Estructura del dataset

```
yolo_dataset/
├── data.yaml           # Configuracion del dataset
├── images/
│   ├── train/          # ~80% de las imagenes
│   └── val/            # ~20% de las imagenes
└── labels/
    ├── train/          # Labels .txt en formato YOLO
    └── val/
```

## Formato de labels YOLO

Cada imagen tiene un archivo .txt con el mismo nombre:
```
# sem_001.txt
# class_id x_center y_center width height (normalizado 0-1)
0 0.5 0.4 0.3 0.6    # semaforo
1 0.15 0.4 0.1 0.15  # arrow_left
```

## Requisitos

- Python 3.10+
- `pip install ultralytics`
- ~2GB de espacio en disco
- ~500MB de RAM para inferencia
