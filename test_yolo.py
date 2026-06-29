"""
Prueba del modelo YOLO entrenado para semaforo digital.

Uso:
  python test_yolo.py                          # Probar con carpeta default del dataset
  python test_yolo.py --image sem_001.jpg      # Probar con imagen especifica
  python test_yolo.py --folder yolo_dataset/images/val   # Carpeta custom
  python test_yolo.py --conf 0.5               # Umbral de confianza custom

"""

import cv2
import numpy as np
import os
import sys
import argparse
from collections import Counter


def test_single_image(model, image_path: str, conf: float = 0.25):
    """Prueba el modelo con una sola imagen."""
    if not os.path.exists(image_path):
        print(f"[ERROR] No se encontro: {image_path}")
        return None

    results = model(image_path, conf=conf)

    print(f"\n{'=' * 50}")
    print(f"  Resultado: {os.path.basename(image_path)}")
    print(f"{'=' * 50}")

    detections = []
    # F16 fix: el "(sin detecciones)" estaba dentro del for result, podia
    # imprimirse varias veces para imagenes multi-result. Ahora se cuenta
    # primero y se imprime una sola vez.
    total_boxes = sum(len(r.boxes) for r in results)
    if total_boxes == 0:
        print("  (sin detecciones)")

    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            conf_val = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])
            cls_name = result.names[cls_id]

            detections.append({
                "class": cls_name,
                "confidence": conf_val,
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })

            print(f"  {cls_name}: {conf_val:.2%} at [{int(x1)},{int(y1)},{int(x2)},{int(y2)}]")

        # Guardar imagen anotada
        annotated = result[0].plot()
        output_path = f"test_output_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, annotated)
        print(f"  Guardado: {output_path}")

    return detections


def test_folder(model, folder_path: str, conf: float = 0.25, class_names=None):
    """Prueba el modelo con todas las imagenes de una carpeta."""
    if not os.path.exists(folder_path):
        print(f"[ERROR] No se encontro la carpeta: {folder_path}")
        print("Verifica la ruta o pasa --folder <otra/ruta>")
        return

    if not os.path.isdir(folder_path):
        print(f"[ERROR] {folder_path} no es una carpeta")
        return

    images = [f for f in os.listdir(folder_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not images:
        print(f"[WARN] No se encontraron imagenes en {folder_path}")
        return

    print(f"\n[INFO] Probando {len(images)} imagenes de {folder_path}")

    # Y6 fix: las stats se construian con un dict hardcoded (semaforo,
    # arrow_left, arrow_right). Si el modelo se reentrena con clases
    # distintas, el resumen miente. Ahora se derivan dinamicamente de
    # model.names.
    if class_names is None:
        class_names = ["semaforo", "arrow_left", "arrow_right"]
    stats = Counter({name: 0 for name in class_names})
    stats["total"] = 0

    for img_name in sorted(images):
        img_path = os.path.join(folder_path, img_name)
        detections = test_single_image(model, img_path, conf=conf)

        for det in (detections or []):
            if det["class"] in stats:
                stats[det["class"]] += 1
            stats["total"] += 1

    print(f"\n{'=' * 50}")
    print(f"  RESUMEN")
    print(f"{'=' * 50}")
    # Imprimir cada clase que el modelo conoce
    for name in class_names:
        print(f"  {name}: {stats[name]}")
    print(f"  Total detecciones:     {stats['total']}")
    print(f"  Promedio por imagen:   {stats['total']/len(images):.1f}")
    print(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(description="Probar modelo YOLO de semaforo")
    parser.add_argument("--image", type=str, help="Imagen a probar")
    parser.add_argument("--folder", type=str, help="Carpeta con imagenes a probar")
    parser.add_argument("--model", type=str, default="backend/models/semaforo_yolo.pt",
                        help="Ruta al modelo entrenado")
    # Y15 fix: umbral de confianza configurable (antes hardcoded a 0.25)
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Umbral de confianza (default: 0.25)")
    args = parser.parse_args()

    # Verificar que el modelo existe
    if not os.path.exists(args.model):
        print(f"[ERROR] No se encontro el modelo: {args.model}")
        print("Primero ejecuta: python train_yolo.py")
        sys.exit(1)

    # Cargar modelo
    print(f"[INFO] Cargando modelo: {args.model}")
    from ultralytics import YOLO
    model = YOLO(args.model)
    # Y6 fix (parte 2): obtener nombres de clases del modelo real
    class_names = list(model.names.values())

    if args.image:
        test_single_image(model, args.image, conf=args.conf)
    elif args.folder:
        test_folder(model, args.folder, conf=args.conf, class_names=class_names)
    else:
        # Y4 fix: el default era "Flow/" que no existe y crasheaba con
        # FileNotFoundError. Ahora es la carpeta val del dataset (o train
        # como fallback). Si tampoco existe, mensaje claro.
        default = "yolo_dataset/images/val"
        if not os.path.exists(default):
            fallback = "yolo_dataset/images/train"
            if os.path.exists(fallback):
                default = fallback
            else:
                print(f"[ERROR] No se encontro {default}")
                print("Pasa --image <ruta> o --folder <ruta> explicitamente.")
                sys.exit(1)
        test_folder(model, default, conf=args.conf, class_names=class_names)


if __name__ == "__main__":
    main()
