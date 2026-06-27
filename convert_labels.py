r"""
Convierte labels exportados de Roboflow a formato YOLO estandar.

Roboflow exporta en formato YOLO, pero a veces las imagenes
vienen en subcarpetas. Este script unifica todo.

Uso:
  1. Exporta el dataset de Roboflow en formato YOLO
  2. Extrae el zip en una carpeta
  3. Ejecuta: python convert_labels.py <carpeta_roboflow>

Ejemplo:
  python convert_labels.py "C:\Users\aranc\Downloads\semaforo-dataset-2"

Nota: el prefijo r" " es para que Python no interprete \U como unicode escape.
"""

import os
import shutil
import sys


def convert_labels(roboflow_path: str, output_path: str = "yolo_dataset"):
    """Convierte labels de Roboflow a nuestro dataset YOLO."""

    print(f"[INFO] Leyendo dataset de: {roboflow_path}")

    # Buscar imagenes y labels
    images_dir = os.path.join(roboflow_path, "train", "images")
    labels_dir = os.path.join(roboflow_path, "train", "labels")

    if not os.path.exists(images_dir):
        # Roboflow a veces usa estructura diferente
        images_dir = os.path.join(roboflow_path, "images", "train")
        labels_dir = os.path.join(roboflow_path, "labels", "train")

    if not os.path.exists(images_dir):
        print(f"[ERROR] No se encontro carpeta de imagenes en {roboflow_path}")
        print("Estructura esperada:")
        print("  carpeta/")
        print("    train/")
        print("      images/")
        print("      labels/")
        return

    # Y1 fix: validar que labels_dir existe. Antes el codigo entraba a
    # chequear os.path.exists(src_label) por cada imagen y silenciosamente
    # copiaba imagenes sin labels si la carpeta no existia. Resultado: dataset
    # sin anotaciones y error confuso en el entrenamiento.
    if not os.path.exists(labels_dir):
        print(f"[ERROR] No se encontro carpeta de labels: {labels_dir}")
        print("Asegurate de exportar el dataset con las anotaciones.")
        return

    # Crear carpetas de salida
    os.makedirs(os.path.join(output_path, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labels", "val"), exist_ok=True)

    # Copiar imagenes y labels de train
    images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"[INFO] Encontradas {len(images)} imagenes en train/")

    for img_name in images:
        # Copiar imagen
        src_img = os.path.join(images_dir, img_name)
        dst_img = os.path.join(output_path, "images", "train", img_name)
        shutil.copy2(src_img, dst_img)

        # Copiar label correspondiente
        label_name = os.path.splitext(img_name)[0] + ".txt"
        src_label = os.path.join(labels_dir, label_name)
        if os.path.exists(src_label):
            dst_label = os.path.join(output_path, "labels", "train", label_name)
            shutil.copy2(src_label, dst_label)

    # Copiar imagenes y labels de val (si existen)
    val_images_dir = os.path.join(roboflow_path, "valid", "images")
    val_labels_dir = os.path.join(roboflow_path, "valid", "labels")

    if not os.path.exists(val_images_dir):
        val_images_dir = os.path.join(roboflow_path, "val", "images")
        val_labels_dir = os.path.join(roboflow_path, "val", "labels")

    if os.path.exists(val_images_dir):
        val_images = [f for f in os.listdir(val_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"[INFO] Encontradas {len(val_images)} imagenes en val/")

        for img_name in val_images:
            src_img = os.path.join(val_images_dir, img_name)
            dst_img = os.path.join(output_path, "images", "val", img_name)
            shutil.copy2(src_img, dst_img)

            label_name = os.path.splitext(img_name)[0] + ".txt"
            src_label = os.path.join(val_labels_dir, label_name)
            if os.path.exists(src_label):
                dst_label = os.path.join(output_path, "labels", "val", label_name)
                shutil.copy2(src_label, dst_label)
    else:
        print("[INFO] No hay carpeta val/ - se usara validacion automatica")

    # Copiar data.yaml si existe
    yaml_src = os.path.join(roboflow_path, "data.yaml")
    if os.path.exists(yaml_src):
        shutil.copy2(yaml_src, os.path.join(output_path, "data.yaml"))
        print("[INFO] data.yaml copiado")
    else:
        print("[WARN] No se encontro data.yaml - crea uno manualmente")

    # Resumen
    train_count = len(os.listdir(os.path.join(output_path, "images", "train")))
    val_count = len(os.listdir(os.path.join(output_path, "images", "val")))
    # Y1 fix (parte 2): sanity check de balance imagenes vs labels. Si las
    # cantidades difieren, el dataset esta silenciosamente roto. Antes el
    # usuario descubria esto recien al fallar el entrenamiento.
    train_labels_dir = os.path.join(output_path, "labels", "train")
    train_label_count = (
        len([f for f in os.listdir(train_labels_dir) if f.endswith('.txt')])
        if os.path.exists(train_labels_dir) else 0
    )
    val_labels_dir = os.path.join(output_path, "labels", "val")
    val_label_count = (
        len([f for f in os.listdir(val_labels_dir) if f.endswith('.txt')])
        if os.path.exists(val_labels_dir) else 0
    )
    print(f"\n[INFO] Dataset convertido exitosamente!")
    print(f"  Train: {train_count} imagenes, {train_label_count} labels")
    print(f"  Val:   {val_count} imagenes, {val_label_count} labels")
    if train_count != train_label_count:
        print(f"  [WARN] Train imbalance: {train_count} imagenes vs {train_label_count} labels")
    if val_count != val_label_count and val_count > 0:
        print(f"  [WARN] Val imbalance: {val_count} imagenes vs {val_label_count} labels")
    print(f"  Output: {output_path}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python convert_labels.py <carpeta_roboflow>")
        # Usamos comillas para que el path no sea malinterpretado por el shell
        print('Ejemplo: python convert_labels.py "C:\\Users\\aranc\\Downloads\\semaforo-dataset-2"')
        sys.exit(1)

    convert_labels(sys.argv[1])
