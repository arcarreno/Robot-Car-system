"""
Entrenamiento de YOLO para deteccion de semaforo digital + flechas.

Uso:
  python train_yolo.py                          # Defaults
  python train_yolo.py --epochs 100 --device 0 # Custom epochs + GPU
  python train_yolo.py --help                  # Ver todas las opciones

Requisitos:
  pip install ultralytics

El modelo entrenado se guarda en:
  runs/detect/semaforo_detector/weights/best.pt
"""

import argparse
from ultralytics import YOLO
import os
import shutil


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrena YOLO para deteccion de semaforo digital + flechas."
    )
    # Dataset
    parser.add_argument("--data", type=str, default="yolo_dataset/data.yaml",
                        help="Ruta al data.yaml del dataset")
    # Hiperparametros principales (Y7 fix: antes hardcoded)
    parser.add_argument("--epochs", type=int, default=50,
                        help="Cantidad de epochs de entrenamiento")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Tamaño de imagen de entrada")
    parser.add_argument("--batch", type=int, default=8,
                        help="Tamaño de batch")
    parser.add_argument("--device", type=str, default="cpu",
                        help='Device: "cpu" o "0" para GPU 0')
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (epochs sin mejora)")
    parser.add_argument("--lr0", type=float, default=0.01,
                        help="Learning rate inicial")
    # Augmentation
    parser.add_argument("--mosaic", type=float, default=1.0,
                        help="Mosaic augmentation factor")
    parser.add_argument("--mixup", type=float, default=0.1,
                        help="Mixup augmentation factor")
    parser.add_argument("--degrees", type=float, default=10.0,
                        help="Rotacion maxima en grados")
    parser.add_argument("--fliplr", type=float, default=0.5,
                        help="Probabilidad de flip horizontal")
    parser.add_argument("--hsv-h", type=float, default=0.015,
                        help="Variacion de tono (HSV-H)")
    parser.add_argument("--hsv-s", type=float, default=0.7,
                        help="Variacion de saturacion (HSV-S)")
    parser.add_argument("--hsv-v", type=float, default=0.4,
                        help="Variacion de brillo (HSV-V)")
    # Outputs
    parser.add_argument("--project", type=str, default="runs/detect",
                        help="Directorio raiz de outputs de Ultralytics")
    parser.add_argument("--name", type=str, default="semaforo_detector",
                        help="Nombre del run (carpeta bajo --project)")
    parser.add_argument("--base-model", type=str, default="yolo11n.pt",
                        help="Modelo pre-entrenado base para transfer learning")
    parser.add_argument("--output-dir", type=str, default="backend/models",
                        help="Directorio destino de los .pt y .onnx finales")
    return parser.parse_args()


def main():
    print("=" * 60)
    print("  Entrenamiento YOLO - Semaforo Digital + Flechas")
    print("=" * 60)

    args = parse_args()

    # Verificar que el dataset existe
    dataset_path = args.data
    if not os.path.exists(dataset_path):
        print(f"[ERROR] No se encontro {dataset_path}")
        # Y2 fix: mensaje correcto. Antes decia "python prepare_dataset.py"
        # pero ese script no existe. El script real es convert_labels.py.
        print("Primero ejecuta: python convert_labels.py <carpeta_roboflow>")
        return

    # Y3 fix: validar que train_path existe ANTES de listar. Antes si el
    # export de Roboflow no tenia train/ (ej. test-only), el script crasheaba
    # con FileNotFoundError en os.listdir(train_path).
    train_path = os.path.join(os.path.dirname(dataset_path), "images", "train")
    if not os.path.exists(train_path):
        print(f"[ERROR] No se encontro {train_path}")
        print("Verifica que el dataset tenga la estructura train/images/")
        return

    train_images = [f for f in os.listdir(train_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"\n[INFO] Imagenes de entrenamiento: {len(train_images)}")

    if len(train_images) < 10:
        print("[ERROR] Se necesitan al menos 10 imagenes etiquetadas")
        print("Etiqueta las imagenes en Roboflow y exporta el dataset")
        return

    # Verificar que hay imagenes en val
    val_path = os.path.join(os.path.dirname(dataset_path), "images", "val")
    if os.path.exists(val_path):
        val_images = [f for f in os.listdir(val_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"[INFO] Imagenes de validacion: {len(val_images)}")
    else:
        print("[WARN] No hay carpeta val/ - se usara 20% de train para validacion")

    # Verificar que las labels existen
    train_labels = os.path.join(os.path.dirname(dataset_path), "labels", "train")
    if os.path.exists(train_labels):
        label_files = [f for f in os.listdir(train_labels) if f.endswith('.txt')]
        print(f"[INFO] Labels de entrenamiento: {len(label_files)}")
    else:
        print("[ERROR] No hay carpeta labels/train/ con archivos .txt")
        print("Etiqueta las imagenes en Roboflow y exporta el dataset")
        return

    print(f"\n[INFO] Iniciando entrenamiento...")

    # Cargar modelo pre-entrenado (transfer learning)
    model = YOLO(args.base_model)

    # Entrenar
    try:
        results = model.train(
            data=dataset_path,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            patience=args.patience,
            lr0=args.lr0,
            mosaic=args.mosaic,
            mixup=args.mixup,
            degrees=args.degrees,
            fliplr=args.fliplr,
            hsv_h=args.hsv_h,
            hsv_s=args.hsv_s,
            hsv_v=args.hsv_v,
            project=args.project,
            name=args.name,
        )
    except Exception as e:
        print(f"[ERROR] Entrenamiento fallo: {e}")
        return

    print(f"\n[INFO] Entrenamiento completado!")
    print(f"[INFO] Resultados en: {args.project}/{args.name}/")

    # Validar
    print(f"\n[INFO] Validando modelo...")
    best_pt_path = os.path.join(args.project, args.name, "weights", "best.pt")
    if not os.path.exists(best_pt_path):
        print(f"[ERROR] No se encontro {best_pt_path}")
        return

    best_model = YOLO(best_pt_path)
    metrics = best_model.val()

    print(f"\n{'=' * 60}")
    print(f"  RESULTADOS DE VALIDACION")
    print(f"{'=' * 60}")
    print(f"  mAP50:       {metrics.box.map50:.4f}")
    print(f"  mAP50-95:    {metrics.box.map:.4f}")
    print(f"  Precision:   {metrics.box.mp:.4f}")
    print(f"  Recall:      {metrics.box.mr:.4f}")
    print(f"{'=' * 60}")

    # Exportar a ONNX para CPU mas rapido
    print(f"\n[INFO] Exportando a ONNX...")
    try:
        best_model.export(format="onnx", simplify=True)
    except Exception as e:
        print(f"[WARN] Export ONNX fallo: {e}")

    best_onnx_path = os.path.join(args.project, args.name, "weights", "best.onnx")
    print(f"[INFO] Modelo ONNX: {best_onnx_path}")

    # Copiar modelo final al directorio del backend
    backend_models = args.output_dir
    os.makedirs(backend_models, exist_ok=True)
    shutil.copy(best_pt_path, os.path.join(backend_models, "semaforo_yolo.pt"))
    print(f"[INFO] best.pt copiado a {backend_models}/semaforo_yolo.pt")
    if os.path.exists(best_onnx_path):
        shutil.copy(best_onnx_path, os.path.join(backend_models, "semaforo_yolo.onnx"))
        print(f"[INFO] best.onnx copiado a {backend_models}/semaforo_yolo.onnx")

    print(f"\n[INFO] ¡Listo! Ahora podes usar el modelo con:")
    print(f"  from ultralytics import YOLO")
    print(f"  model = YOLO('{backend_models}/semaforo_yolo.pt')")


if __name__ == "__main__":
    main()
