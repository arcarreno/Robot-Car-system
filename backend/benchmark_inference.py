"""
Benchmark de inferencia YOLO - PyTorch vs OpenVINO.

Mide latencia en ms por inferencia usando TU modelo custom
(semaforo_yolo.pt) sobre frames sinteticos de 320x240 (los mismos
que produce el ESP32-CAM).

Uso:
    cd backend
    python benchmark_inference.py
"""

import time
import sys
import os
import numpy as np
import cv2

MODEL_PATH = "backend/models/semaforo_yolo.pt"
NUM_WARMUP = 10
NUM_ITERATIONS = 100
FRAME_SIZE = (240, 320, 3)  # alto, ancho, canales (formato OpenCV)

def benchmark_pytorch():
    """Mide latencia de YOLO con PyTorch (CPU)."""
    from ultralytics import YOLO
    print(f"[PyTorch] Cargando modelo: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # Frame sintetico (lo que se ve en una escena con semaforo)
    frame = np.random.randint(0, 255, FRAME_SIZE, dtype=np.uint8)

    # Warmup
    print(f"[PyTorch] Warmup ({NUM_WARMUP} iteraciones)...")
    for _ in range(NUM_WARMUP):
        _ = model(frame, verbose=False, conf=0.25)

    # Benchmark
    print(f"[PyTorch] Benchmark ({NUM_ITERATIONS} iteraciones)...")
    times = []
    for _ in range(NUM_ITERATIONS):
        t0 = time.perf_counter()
        _ = model(frame, verbose=False, conf=0.25)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    times = np.array(times)
    print(f"[PyTorch] Resultados:")
    print(f"  Latencia media:  {times.mean():.2f} ms")
    print(f"  Latencia p50:    {np.median(times):.2f} ms")
    print(f"  Latencia p95:    {np.percentile(times, 95):.2f} ms")
    print(f"  Latencia min:    {times.min():.2f} ms")
    print(f"  Latencia max:    {times.max():.2f} ms")
    print(f"  FPS teoricos:    {1000 / times.mean():.1f}")
    return times


def benchmark_openvino(model_path="backend/models/semaforo_yolo_openvino_model/semaforo_yolo.xml"):
    """Mide latencia de YOLO con OpenVINO (CPU)."""
    import openvino as ov
    print(f"\n[OpenVINO] Cargando modelo: {model_path}")
    if not os.path.exists(model_path):
        print(f"[OpenVINO] ERROR: modelo no encontrado. Ejecuta train primero para exportarlo.")
        return None

    core = ov.Core()
    # Listar devices disponibles
    devices = core.available_devices
    print(f"[OpenVINO] Devices disponibles: {devices}")

    # Compilar para CPU primero
    print(f"[OpenVINO] Compilando para CPU...")
    compiled = core.compile_model(model_path, "CPU")

    # Frame sintetico - resize a 640x640 (lo que YOLO espera internamente)
    frame = np.random.randint(0, 255, FRAME_SIZE, dtype=np.uint8)
    frame_640 = cv2.resize(frame, (640, 640))
    # OpenVINO espera float32, normalizado a [0,1], en formato NCHW
    input_tensor = frame_640.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC -> CHW
    input_tensor = np.expand_dims(input_tensor, axis=0)    # -> NCHW

    # Warmup
    print(f"[OpenVINO] Warmup ({NUM_WARMUP} iteraciones)...")
    for _ in range(NUM_WARMUP):
        _ = compiled([input_tensor])

    # Benchmark
    print(f"[OpenVINO] Benchmark ({NUM_ITERATIONS} iteraciones)...")
    times = []
    for _ in range(NUM_ITERATIONS):
        t0 = time.perf_counter()
        _ = compiled([input_tensor])
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    times = np.array(times)
    print(f"[OpenVINO] Resultados:")
    print(f"  Latencia media:  {times.mean():.2f} ms")
    print(f"  Latencia p50:    {np.median(times):.2f} ms")
    print(f"  Latencia p95:    {np.percentile(times, 95):.2f} ms")
    print(f"  Latencia min:    {times.min():.2f} ms")
    print(f"  Latencia max:    {times.max():.2f} ms")
    print(f"  FPS teoricos:    {1000 / times.mean():.1f}")
    return times


def benchmark_openvino_gpu(model_path="backend/models/semaforo_yolo_openvino_model/semaforo_yolo.xml"):
    """Mide latencia de YOLO con OpenVINO (GPU integrada Intel)."""
    import openvino as ov
    print(f"\n[OpenVINO GPU] Cargando modelo: {model_path}")
    if not os.path.exists(model_path):
        print(f"[OpenVINO GPU] ERROR: modelo no encontrado.")
        return None

    core = ov.Core()
    try:
        print(f"[OpenVINO GPU] Compilando para GPU.0 (Iris Xe)...")
        compiled = core.compile_model(model_path, "GPU")
    except Exception as e:
        print(f"[OpenVINO GPU] No se pudo compilar para GPU: {e}")
        return None

    frame = np.random.randint(0, 255, FRAME_SIZE, dtype=np.uint8)
    frame_640 = cv2.resize(frame, (640, 640))
    input_tensor = frame_640.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # Warmup
    print(f"[OpenVINO GPU] Warmup ({NUM_WARMUP} iteraciones)...")
    for _ in range(NUM_WARMUP):
        _ = compiled([input_tensor])

    # Benchmark
    print(f"[OpenVINO GPU] Benchmark ({NUM_ITERATIONS} iteraciones)...")
    times = []
    for _ in range(NUM_ITERATIONS):
        t0 = time.perf_counter()
        _ = compiled([input_tensor])
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times = np.array(times)
    print(f"[OpenVINO GPU] Resultados:")
    print(f"  Latencia media:  {times.mean():.2f} ms")
    print(f"  Latencia p50:    {np.median(times):.2f} ms")
    print(f"  Latencia p95:    {np.percentile(times, 95):.2f} ms")
    print(f"  Latencia min:    {times.min():.2f} ms")
    print(f"  Latencia max:    {times.max():.2f} ms")
    print(f"  FPS teoricos:    {1000 / times.mean():.1f}")
    return times


def main():
    print("=" * 70)
    print(f"  BENCHMARK DE INFERENCIA YOLO")
    print(f"  Modelo: {MODEL_PATH}")
    print(f"  Frame size: {FRAME_SIZE}")
    print(f"  Iterations: {NUM_ITERATIONS} (warmup: {NUM_WARMUP})")
    print("=" * 70)

    results = {}

    # 1) PyTorch CPU
    try:
        results["pytorch_cpu"] = benchmark_pytorch()
    except Exception as e:
        print(f"[PyTorch] Error: {e}")

    # 2) OpenVINO CPU
    try:
        results["openvino_cpu"] = benchmark_openvino()
    except Exception as e:
        print(f"[OpenVINO CPU] Error: {e}")

    # 3) OpenVINO GPU (Iris Xe)
    try:
        results["openvino_gpu"] = benchmark_openvino_gpu()
    except Exception as e:
        print(f"[OpenVINO GPU] Error: {e}")

    # Resumen final
    print("\n" + "=" * 70)
    print(f"  RESUMEN COMPARATIVO")
    print("=" * 70)
    print(f"  {'Backend':<25} {'Latencia (ms)':<15} {'FPS teoricos':<15} {'Speedup':<10}")
    print("-" * 70)

    baseline = results.get("pytorch_cpu")
    if baseline is not None:
        baseline_fps = 1000 / baseline.mean()
    else:
        baseline_fps = None

    for name, times in results.items():
        if times is None:
            continue
        fps = 1000 / times.mean()
        speedup = (baseline_fps / fps) if baseline_fps else 1.0
        # Mostrar speedup vs PyTorch (el mas lento)
        if name == "pytorch_cpu":
            speedup_str = "1.0x (baseline)"
        else:
            speedup_str = f"{baseline_fps / fps:.2f}x vs PyTorch" if baseline_fps else "?"
        print(f"  {name:<25} {times.mean():>10.2f}    {fps:>10.1f}    {speedup_str}")

    # Estimacion de FPS en el sistema completo (con pipeline completo)
    print("\n" + "=" * 70)
    print(f"  ESTIMACION EN TU DEMO COMPLETO (pipeline real)")
    print("=" * 70)
    print(f"  Overhead no-YOLO (MJPEG, HSV, WS, render): ~40-60 ms/frame")
    print()
    if baseline is not None:
        demo_yolo_pytorch = baseline.mean() + 50
        fps_pytorch = 1000 / demo_yolo_pytorch
        print(f"  PyTorch CPU:  YOLO {baseline.mean():.0f}ms + overhead 50ms = {demo_yolo_pytorch:.0f}ms/frame = ~{fps_pytorch:.1f} FPS")

    if "openvino_cpu" in results and results["openvino_cpu"] is not None:
        ov_cpu = results["openvino_cpu"].mean()
        demo_ov_cpu = ov_cpu + 50
        fps_ov_cpu = 1000 / demo_ov_cpu
        print(f"  OpenVINO CPU: YOLO {ov_cpu:.0f}ms + overhead 50ms = {demo_ov_cpu:.0f}ms/frame = ~{fps_ov_cpu:.1f} FPS")

    if "openvino_gpu" in results and results["openvino_gpu"] is not None:
        ov_gpu = results["openvino_gpu"].mean()
        demo_ov_gpu = ov_gpu + 50
        fps_ov_gpu = 1000 / demo_ov_gpu
        print(f"  OpenVINO iGPU: YOLO {ov_gpu:.0f}ms + overhead 50ms = {demo_ov_gpu:.0f}ms/frame = ~{fps_ov_gpu:.1f} FPS")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
