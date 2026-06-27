"""
Calibracion MiDaS mejorada con 3 puntos.
Usa minimos cuadrados con bounds para encontrar K, P, offset.
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution

# Datos de calibracion: (raw_avg, distancia_real)
points = [
    (368.6, 0.3),   # 0.3m
    (205.5, 0.5),   # 0.5m (con regla)
    (243.6, 0.8),   # 0.8m
]

raw_vals = np.array([p[0] for p in points])
dist_vals = np.array([p[1] for p in points])

def model(raw, K, P, offset):
    return K / np.power(raw - offset, P)

def objective(params):
    K, P, offset = params
    if K <= 0 or P <= 0:
        return 1e10
    if np.any(raw_vals - offset <= 0):
        return 1e10
    predicted = model(raw_vals, K, P, offset)
    # Error relativo porcentual
    errors = np.abs(predicted - dist_vals) / dist_vals
    return np.sum(errors ** 2)

# Bounds: K>0, P>0, offset < min(raw) - 1
bounds = [(100, 100000), (0.5, 5.0), (-2000, min(raw_vals) - 1)]

# Differential evolution (global optimizer)
result = differential_evolution(objective, bounds, seed=42, maxiter=10000, tol=1e-12)

K_opt, P_opt, offset_opt = result.x

print("=== Nuevos parametros de calibracion ===")
print(f"MIDAS_CALIBRATION_K = {K_opt:.1f}")
print(f"MIDAS_CALIBRATION_P = {P_opt:.2f}")
print(f"MIDAS_CALIBRATION_OFFSET = {offset_opt:.1f}")
print()

print("=== Verificacion ===")
total_error = 0
for raw, dist in zip(raw_vals, dist_vals):
    predicted = model(raw, K_opt, P_opt, offset_opt)
    error = abs(predicted - dist)
    pct = error / dist * 100
    total_error += error
    print(f"  A {dist}m: D = {K_opt:.1f} / ({raw:.1f} - ({offset_opt:.1f}))^{P_opt:.2f} = {predicted:.3f}m (error: {error:.3f}m = {pct:.1f}%)")
print(f"  Error promedio: {total_error/len(points):.3f}m")
print()

# Comparar con calibracion anterior
print("=== Comparacion con calibracion anterior ===")
print(f"{'Distancia':>10} | {'Anterior':>10} | {'Nueva':>10} | {'Mejora':>10}")
print(f"{'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")

    old_K, old_P, old_offset = 99997.1, 1.94, -335.7
for raw, dist in zip(raw_vals, dist_vals):
    old_pred = model(raw, old_K, old_P, old_offset)
    new_pred = model(raw, K_opt, P_opt, offset_opt)
    old_error = abs(old_pred - dist)
    new_error = abs(new_pred - dist)
    improvement = old_error - new_error
    marker = " OK" if new_error < old_error else ""
    print(f"{dist:>9.1f}m | {old_pred:>9.3f}m | {new_pred:>9.3f}m | {improvement:>+9.3f}m{marker}")

print()
print("=== Config para backend/config.py ===")
print(f"MIDAS_CALIBRATION_K = {K_opt:.1f}")
print(f"MIDAS_CALIBRATION_P = {P_opt:.2f}")
print(f"MIDAS_CALIBRATION_OFFSET = {offset_opt:.1f}")
