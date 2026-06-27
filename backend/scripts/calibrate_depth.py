"""
Script de calibracion MiDaS → metros (dos puntos).

Conecta al stream MJPEG del ESP32, captura frames, y muestra el valor raw
de MiDaS en el centro. El usuario pone un obstaculo a distancia conocida
y el script calcula los parametros de calibracion (K, P, offset).

Uso:
    # Calibrar un punto (guarda el valor raw):
    python scripts/calibrate_depth.py --real-distance 0.3 --save raw_03.json
    python scripts/calibrate_depth.py --real-distance 0.8 --save raw_08.json

    # Calcular K, P, offset con dos puntos:
    python scripts/calibrate_depth.py --calibrate-two-points raw_03.json raw_08.json

    # Calibracion rapida de un solo punto (comportamiento original):
    python scripts/calibrate_depth.py --real-distance 0.5
"""

import argparse
import json
import os
import sys
import time
import io
from pathlib import Path

import cv2
import numpy as np
import requests

# Agregar directorio padre al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ESP32_STREAM, DEPTH_MODEL_PATH, DEPTH_DEVICE
from core.depth_estimator import DepthEstimator


def capture_frame(stream_url: str, timeout: float = 3.0) -> np.ndarray:
    """Captura un frame del stream MJPEG usando requests (mas confiable que VideoCapture)."""
    try:
        # Conectar al stream con timeout
        resp = requests.get(stream_url, stream=True, timeout=timeout)
        resp.raise_for_status()

        # Leer chunks hasta encontrar un frame JPEG completo
        bytes_buffer = b""
        for chunk in resp.iter_content(chunk_size=4096):
            bytes_buffer += chunk
            # Buscar fin de JPEG (FFD9)
            start = bytes_buffer.find(b'\xff\xd8')  # inicio JPEG
            end = bytes_buffer.find(b'\xff\xd9')    # fin JPEG
            if start != -1 and end != -1:
                jpeg_data = bytes_buffer[start:end + 2]
                resp.close()

                # Decodificar JPEG
                nparr = np.frombuffer(jpeg_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return frame

        resp.close()
        return None
    except Exception as e:
        print(f"  [WARN] Error capturando frame: {e}")
        return None


def calibrate_single_point(stream_url: str, real_distance: float, samples: int,
                           estimator: DepthEstimator) -> dict:
    """
    Calibra un solo punto y retorna el valor raw promedio.

    Returns:
        dict con: real_distance, raw_avg, raw_std, raw_median, raw_values
    """
    print(f"INSTRUCCIONES:")
    print(f"  1. Ponga un obstaculo a EXACTAMENTE {real_distance}m de la camara")
    print(f"  2. Asegurese que el obstaculo este en el CENTRO del frame")
    print(f"  3. Presione ENTER para capturar {samples} frames...")
    print(f"  4. Para cancelar: Ctrl+C")
    print()

    try:
        input("  > Presione ENTER aqui... ")
    except KeyboardInterrupt:
        print("\n  Cancelado.")
        return None

    print()
    print("Capturando frames...")
    print()

    # Capturar samples
    raw_values = []
    for i in range(samples):
        print(f"  [{i+1}/{samples}] Capturando...", end=" ", flush=True)
        frame = capture_frame(stream_url)
        if frame is None:
            print("ERROR: no se pudo capturar frame")
            continue

        depth_map = estimator.estimate(frame)
        if depth_map is None:
            print("ERROR: depth estimation fallo")
            continue

        # Obtener valor raw del centro (sin conversion a metros)
        h, w = depth_map.shape
        cy, cx = h // 2, w // 2
        center_roi = depth_map[cy - 10:cy + 10, cx - 10:cx + 10]
        raw_val = float(np.mean(center_roi)) if center_roi.size > 0 else 0.0

        raw_values.append(raw_val)
        print(f"Raw: {raw_val:.1f}")

        time.sleep(0.5)  # medio segundo entre samples

    if not raw_values:
        print()
        print("[ERROR] No se capturaron frames validos")
        print("  Verifique que la ESP32 esta encendida y el stream funciona")
        return None

    # Calcular estadisticas
    avg_raw = np.mean(raw_values)
    std_raw = np.std(raw_values)
    median_raw = np.median(raw_values)

    return {
        "real_distance": real_distance,
        "raw_avg": float(avg_raw),
        "raw_std": float(std_raw),
        "raw_median": float(median_raw),
        "raw_values": raw_values,
        "samples": len(raw_values),
    }


def calibrate_two_points(file1: str, file2: str, reference_raw: float = 205.5,
                         reference_distance: float = 0.5) -> dict:
    """
    Calcula K, P, offset con dos puntos de calibracion mas un punto de referencia.

    Args:
        file1: Archivo JSON con calibracion del punto 1
        file2: Archivo JSON con calibracion del punto 2
        reference_raw: Valor raw de referencia (default: 222.8 a 0.5m)
        reference_distance: Distancia de referencia (default: 0.5m)

    Returns:
        dict con: K, P, offset, verificacion
    """
    # Cargar datos
    with open(file1, 'r') as f:
        data1 = json.load(f)
    with open(file2, 'r') as f:
        data2 = json.load(f)

    print(f"=== Calibración de Dos Puntos ===")
    print()
    print(f"Punto 1: raw={data1['raw_avg']:.1f} a {data1['real_distance']}m")
    print(f"Punto 2: raw={reference_raw:.1f} a {reference_distance}m (referencia)")
    print(f"Punto 3: raw={data2['raw_avg']:.1f} a {data2['real_distance']}m")
    print()

    # Ordenar puntos por distancia
    points = [
        (data1['raw_avg'], data1['real_distance']),
        (reference_raw, reference_distance),
        (data2['raw_avg'], data2['real_distance']),
    ]
    points.sort(key=lambda x: x[1])  # ordenar por distancia

    raw_vals = [p[0] for p in points]
    dist_vals = [p[1] for p in points]

    # Resolver sistema de ecuaciones: D = K / (raw - offset)^P
    # Usar optimizacion numerica para encontrar K, P, offset
    from scipy.optimize import minimize

    def objective(params):
        K, P, offset = params
        total_error = 0
        for raw, dist in zip(raw_vals, dist_vals):
            if raw - offset <= 0:
                return 1e10  # penalty
            predicted = K / ((raw - offset) ** P)
            total_error += (predicted - dist) ** 2
        return total_error

    # Valores iniciales basados en calibracion existente
    initial_K = 99997.1
    initial_P = 1.94
    initial_offset = -335.7

    # Optimizar
    result = minimize(
        objective,
        x0=[initial_K, initial_P, initial_offset],
        method='Nelder-Mead',
        options={'maxiter': 10000, 'xatol': 0.01, 'fatol': 1e-6}
    )

    K_opt, P_opt, offset_opt = result.x

    print(f"=== Nuevos parámetros de calibración ===")
    print(f"MIDAS_CALIBRATION_K = {K_opt:.1f}")
    print(f"MIDAS_CALIBRATION_P = {P_opt:.2f}")
    print(f"MIDAS_CALIBRATION_OFFSET = {offset_opt:.1f}")
    print()

    # Verificacion
    print(f"=== Verificación ===")
    for raw, dist in zip(raw_vals, dist_vals):
        predicted = K_opt / ((raw - offset_opt) ** P_opt)
        error = abs(predicted - dist)
        print(f"A {dist}m: D = {K_opt:.1f} / ({raw:.1f} - {offset_opt:.1f})^{P_opt:.2f} = {predicted:.2f}m (error: {error:.3f}m)")
    print()

    # Mostrar comparacion con calibracion anterior
    print(f"=== Comparación con calibración anterior ===")
    print(f"{'Distancia':>10} | {'Anterior':>10} | {'Nueva':>10} | {'Mejora':>10}")
    print(f"{'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")

    old_K, old_P, old_offset = 99997.1, 1.94, -335.7
    for raw, dist in zip(raw_vals, dist_vals):
        old_pred = old_K / ((raw - old_offset) ** old_P)
        new_pred = K_opt / ((raw - offset_opt) ** P_opt)
        old_error = abs(old_pred - dist)
        new_error = abs(new_pred - dist)
        improvement = old_error - new_error
        marker = " ✓" if new_error < old_error else ""
        print(f"{dist:>9.1f}m | {old_pred:>9.2f}m | {new_pred:>9.2f}m | {improvement:>+9.3f}m{marker}")

    print()
    print(f"=== Config sugerida ===")
    print(f"MIDAS_CALIBRATION_K = {K_opt:.1f}")
    print(f"MIDAS_CALIBRATION_P = {P_opt:.2f}")
    print(f"MIDAS_CALIBRATION_OFFSET = {offset_opt:.1f}")
    print()
    print("Copie estas lineas en backend/config.py")

    return {
        "K": K_opt,
        "P": P_opt,
        "offset": offset_opt,
        "points": [(raw, dist) for raw, dist in zip(raw_vals, dist_vals)],
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrar MiDaS → metros (dos puntos)")
    parser.add_argument(
        "--real-distance", type=float, default=None,
        help="Distancia real al obstaculo en metros (ej: 0.5)"
    )
    parser.add_argument(
        "--samples", type=int, default=10,
        help="Numero de frames a capturar (default: 10)"
    )
    parser.add_argument(
        "--stream-url", type=str, default=ESP32_STREAM,
        help=f"URL del stream MJPEG (default: {ESP32_STREAM})"
    )
    parser.add_argument(
        "--no-stream", action="store_true",
        help="Modo offline: usar imagen sintetica para probar la calibracion"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Guardar resultado de calibracion en archivo JSON"
    )
    parser.add_argument(
        "--calibrate-two-points", nargs=2, metavar=("FILE1", "FILE2"),
        help="Calcular K, P, offset con dos archivos de calibracion"
    )
    args = parser.parse_args()

    # Modo: calibrar dos puntos
    if args.calibrate_two_points:
        result = calibrate_two_points(args.calibrate_two_points[0], args.calibrate_two_points[1])
        if result:
            # Guardar resultado
            output_file = "calibration_result.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResultado guardado en: {output_file}")
        return

    # Modo: calibrar un punto
    if args.real_distance is None:
        parser.error("--real-distance es requerido para calibracion de un punto")

    print(f"=== Calibracion MiDaS (un punto) ===")
    print(f"Distancia real: {args.real_distance}m")
    print(f"Samples: {args.samples}")
    print(f"Stream: {args.stream_url}")
    print()

    # Crear estimator sin calibration offset (probaremos diferentes valores)
    estimator = DepthEstimator(
        model_path=DEPTH_MODEL_PATH,
        device=DEPTH_DEVICE,
        calibration_k=99997.1,
        calibration_p=1.94,
        calibration_offset=0.0,  # offset base para medir raw
    )

    if not estimator.available:
        print("[ERROR] Modelo MiDaS no disponible")
        print(f"  Ejecute: python scripts/download_midas.py")
        return

    print(f"[OK] Modelo MiDaS cargado")
    print()

    if args.no_stream:
        print("[MODO OFFLINE] Usando imagen sintetica")
        print("  Esto solo prueba la logica de calibracion, no la camara real")
        print()

        # Crear imagen sintetica con obstaculo en centro
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # fondo gris oscuro
        cv2.rectangle(frame, (120, 80), (200, 160), (200, 200, 200), -1)  # obstaculo gris claro

        depth_map = estimator.estimate(frame)
        if depth_map is not None:
            h, w = depth_map.shape
            cy, cx = h // 2, w // 2
            center_roi = depth_map[cy - 10:cy + 10, cx - 10:cx + 10]
            raw_val = float(np.mean(center_roi)) if center_roi.size > 0 else 0.0
            print(f"  Raw value centro: {raw_val:.1f}")
            print(f"  (Esto es solo un ejemplo con imagen sintetica)")
        return

    # Calibrar un punto
    result = calibrate_single_point(args.stream_url, args.real_distance, args.samples, estimator)
    if result is None:
        return

    print()
    print(f"=== Resultados ===")
    print(f"Raw promedio: {result['raw_avg']:.1f} ± {result['raw_std']:.1f}")
    print(f"Raw mediana:  {result['raw_median']:.1f}")
    print()

    # Calcular offset para la distancia objetivo
    K = 99997.1
    P = 1.94
    real_dist = args.real_distance

    # Formula: D = K / (raw - offset)^P
    # Resolviendo para offset: offset = raw - (K/D)^(1/P)
    term = (K / real_dist) ** (1.0 / P)
    suggested_offset = result['raw_avg'] - term

    print(f"=== Offset sugerido ===")
    print(f"Formula: D = {K} / (raw - offset)^{P}")
    print(f"Raw promedio: {result['raw_avg']:.1f}")
    print(f"Distancia real: {real_dist}m")
    print(f"Offset optimo: {suggested_offset:.1f}")
    print()

    # Mostrar verificacion
    verify_dist = K / (result['raw_avg'] - suggested_offset) ** P
    print(f"Verificacion: D = {K} / ({result['raw_avg']:.1f} - {suggested_offset:.1f})^{P} = {verify_dist:.2f}m")
    print()

    # Probar diferentes offsets
    print(f"=== Comparacion de offsets ===")
    print(f"{'Offset':>8} | {'Distancia':>10} | {'Error':>8}")
    print(f"{'-'*8} | {'-'*10} | {'-'*8}")

    for offset in [0, 20, 40, 60, 80, 100, 120, round(suggested_offset)]:
        if result['raw_avg'] - offset <= 0:
            continue
        dist = K / (result['raw_avg'] - offset) ** P
        error = abs(dist - real_dist)
        marker = " ← OPTIMO" if offset == round(suggested_offset) else ""
        print(f"{offset:>8} | {dist:>9.2f}m | {error:>7.2f}m{marker}")

    print()
    print(f"=== Config sugerida ===")
    print(f"MIDAS_CALIBRATION_OFFSET = {suggested_offset:.1f}")
    print()
    print("Copie esta linea en backend/config.py")

    # Guardar en JSON si se pidio
    if args.save:
        output = {
            "real_distance": result['real_distance'],
            "raw_avg": result['raw_avg'],
            "raw_std": result['raw_std'],
            "raw_median": result['raw_median'],
            "suggested_offset": suggested_offset,
            "K": K,
            "P": P,
        }
        with open(args.save, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResultado guardado en: {args.save}")


if __name__ == "__main__":
    main()
