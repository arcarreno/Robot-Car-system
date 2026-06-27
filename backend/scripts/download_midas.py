"""
Descarga el modelo MiDaS Small v2.1 en formato OpenVINO IR (.xml + .bin).

Uso:
    python scripts/download_midas.py

El modelo se descarga desde el repositorio oficial de OpenVINO notebooks
y se guarda en models/midas_small/.
"""

import os
import sys
import urllib.request

# URLs oficiales de Intel
BASE_URL = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/depth-estimation-midas/FP32/"
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "midas_small")
FILES = ["MiDaS_small.xml", "MiDaS_small.bin"]


def download():
    os.makedirs(MODEL_DIR, exist_ok=True)

    for filename in FILES:
        dest = os.path.join(MODEL_DIR, filename)
        if os.path.exists(dest):
            print(f"[OK] {filename} ya existe ({os.path.getsize(dest)} bytes)")
            continue

        url = BASE_URL + filename
        print(f"Descargando {filename}...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"[OK] {filename} descargado ({os.path.getsize(dest)} bytes)")
        except Exception as e:
            print(f"[ERROR] No se pudo descargar {filename}: {e}")
            sys.exit(1)

    print(f"\nModelo MiDaS Small listo en: {MODEL_DIR}")


if __name__ == "__main__":
    download()
