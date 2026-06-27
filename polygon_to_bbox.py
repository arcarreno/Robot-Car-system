"""
Convierte labels de polígono (segmentación) a bounding box (detección).

Formatos soportados:
  - Poligono:    class_id x1 y1 x2 y2 x3 y3 ... xn yn
  - Bounding box (YOLO): class_id x_center y_center width height (5 valores)

Formatos que se transforman a bbox:
  - Polygon: min/max de vertices
  - Otros >5 valores: asumidos poligono

Uso:
  python polygon_to_bbox.py --input <carpeta_labels> [--output <carpeta_salida>]
  python polygon_to_bbox.py --input train/labels/         # in-place si no se da --output
  python polygon_to_bbox.py --input labels/ --output labels_bbox/

Si no se pasa --input, usa ./train/labels
"""

import os
import glob
import argparse


def polygon_to_bbox(line: str) -> str:
    """Convierte una linea de poligono a bounding box (formato YOLO).

    Formato bbox de entrada (5 valores): class x_center y_center w h
      -> se devuelve tal cual.

    Formato poligono (>5 valores): class x1 y1 x2 y2 ... xn yn
      -> se calcula min/max de vertices.
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return line  # formato invalido, dejar como esta

    # Y5 fix (parte 1): si tiene exactamente 5 valores, es bbox YOLO (class +
    # cx, cy, w, h). El codigo original trataba esto como poligono y
    # calculaba min/max sobre [cx, cy, w, h], produciendo un bbox corrupto.
    if len(parts) == 5:
        return line

    class_id = parts[0]
    coords = [float(x) for x in parts[1:]]

    # Separar en pares (x, y)
    x_coords = coords[0::2]
    y_coords = coords[1::2]

    if not x_coords or not y_coords:
        return line

    # Calcular bounding box
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    # Convertir a formato YOLO (center, width, height - normalizado)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def convert_file(input_path: str, output_path: str) -> bool:
    """Convierte un archivo de labels. Devuelve True si se modifico algo."""
    with open(input_path, 'r') as f:
        lines = f.readlines()

    converted = []
    modified = False
    for line in lines:
        line = line.strip()
        if not line:
            converted.append('')
            continue
        new_line = polygon_to_bbox(line)
        if new_line != line:
            modified = True
        converted.append(new_line)

    with open(output_path, 'w') as f:
        f.write('\n'.join(converted) + '\n')

    return modified


def main():
    parser = argparse.ArgumentParser(
        description="Convierte labels de poligono a bounding box (formato YOLO)."
    )
    parser.add_argument("--input", type=str, default="train/labels",
                        help="Carpeta o archivo .txt de entrada (default: train/labels)")
    parser.add_argument("--output", type=str, default=None,
                        help="Carpeta de salida (default: in-place sobre --input)")
    args = parser.parse_args()

    # Y5 fix (parte 2): antes la ruta estaba hardcodeada a
    # C:/Users/aranc/Documents/9NO/HaciendoCiencia/Semaforo.yolov11
    # lo que rompia el script para cualquier otro usuario. Ahora argparse.

    in_path = args.input
    out_path = args.output

    if os.path.isfile(in_path):
        # Convertir un solo archivo
        if out_path is None:
            out_path = in_path  # in-place
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        modified = convert_file(in_path, out_path)
        print(f"[OK] {in_path} -> {out_path} ({'modificado' if modified else 'sin cambios'})")
        return

    # Buscar todos los archivos .txt en la carpeta
    label_files = sorted(glob.glob(os.path.join(in_path, "*.txt")))
    if not label_files:
        print(f"[ERROR] No se encontraron archivos .txt en {in_path}")
        return

    print(f"[INFO] Encontrados {len(label_files)} archivos de labels en {in_path}")
    print(f"[INFO] Convirtiendo poligonos a bounding boxes...")

    if out_path is not None:
        os.makedirs(out_path, exist_ok=True)

    converted_count = 0
    skipped_count = 0
    for label_path in label_files:
        if out_path is not None:
            out_file = os.path.join(out_path, os.path.basename(label_path))
        else:
            out_file = label_path  # in-place
        if convert_file(label_path, out_file):
            converted_count += 1
        else:
            skipped_count += 1

    print(f"[OK] {converted_count} archivos modificados (poligono -> bbox)")
    print(f"[OK] {skipped_count} archivos sin cambios (ya eran bbox)")

    # Verificar resultado
    if label_files:
        print(f"\n[INFO] Verificando primer label ({label_files[0]})...")
        with open(label_files[0], 'r') as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        print(f"  Archivo: {os.path.basename(label_files[0])}")
        print(f"  Lineas: {len(lines)}")
        for i, line in enumerate(lines[:3]):
            parts = line.strip().split()
            if len(parts) == 5:
                print(f"  Linea {i+1}: class={parts[0]}, bbox=[{parts[1]}, {parts[2]}, {parts[3]}, {parts[4]}]")


if __name__ == "__main__":
    main()
