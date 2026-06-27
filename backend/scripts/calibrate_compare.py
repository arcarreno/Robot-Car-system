"""
Calibracion MiDaS con diferentes combinaciones de puntos.
"""
import numpy as np
from scipy.optimize import differential_evolution

# Todos los puntos disponibles
all_points = [
    (368.6, 0.3),   # 0.3m
    (205.5, 0.5),   # 0.5m (con regla)
    (277.4, 0.8),   # 0.8m (re-capturado)
]

def model(raw, K, P, offset):
    return K / np.power(raw - offset, P)

def fit_points(points):
    raw_vals = np.array([p[0] for p in points])
    dist_vals = np.array([p[1] for p in points])
    
    def objective(params):
        K, P, offset = params
        if K <= 0 or P <= 0:
            return 1e10
        if np.any(raw_vals - offset <= 0):
            return 1e10
        predicted = model(raw_vals, K, P, offset)
        # Weight closer points more (they're more important for obstacle avoidance)
        weights = 1.0 / dist_vals  # higher weight for closer points
        errors = np.abs(predicted - dist_vals) / dist_vals
        return np.sum(weights * errors ** 2)
    
    bounds = [(100, 100000), (0.5, 5.0), (-2000, min(raw_vals) - 1)]
    result = differential_evolution(objective, bounds, seed=42, maxiter=10000, tol=1e-12)
    return result.x

# === Combinacion 1: 0.3m + 0.5m ===
print("=== Combinacion 1: 0.3m + 0.5m ===")
pts1 = [(368.6, 0.3), (205.5, 0.5)]
K1, P1, off1 = fit_points(pts1)
for raw, dist in pts1:
    pred = model(raw, K1, P1, off1)
    err = abs(pred - dist) / dist * 100
    print(f"  {dist}m: {pred:.3f}m (error: {err:.1f}%)")
print(f"  K={K1:.1f}, P={P1:.2f}, offset={off1:.1f}")
print()

# === Combinacion 2: 0.3m + 0.5m + 0.8m ===
print("=== Combinacion 2: 0.3m + 0.5m + 0.8m (3 puntos) ===")
pts2 = all_points
K2, P2, off2 = fit_points(pts2)
for raw, dist in pts2:
    pred = model(raw, K2, P2, off2)
    err = abs(pred - dist) / dist * 100
    print(f"  {dist}m: {pred:.3f}m (error: {err:.1f}%)")
print(f"  K={K2:.1f}, P={P2:.2f}, offset={off2:.1f}")
print()

# === Combinacion 3: Solo 0.3m + 0.8m ===
print("=== Combinacion 3: 0.3m + 0.8m ===")
pts3 = [(368.6, 0.3), (277.4, 0.8)]
K3, P3, off3 = fit_points(pts3)
for raw, dist in pts3:
    pred = model(raw, K3, P3, off3)
    err = abs(pred - dist) / dist * 100
    print(f"  {dist}m: {pred:.3f}m (error: {err:.1f}%)")
print(f"  K={K3:.1f}, P={P3:.2f}, offset={off3:.1f}")
print()

# === Verificar mejor combinacion con TODOS los puntos ===
print("=== Verificacion de mejor combinacion con todos los puntos ===")
for name, K, P, off in [("0.3+0.5", K1, P1, off1), ("3 puntos", K2, P2, off2), ("0.3+0.8", K3, P3, off3)]:
    total_err = 0
    for raw, dist in all_points:
        pred = model(raw, K, P, off)
        total_err += abs(pred - dist)
    avg_err = total_err / len(all_points)
    print(f"  {name}: error promedio = {avg_err:.3f}m ({avg_err/0.5*100:.1f}%)")
