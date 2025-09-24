import numpy as np
from typing import Literal, List, Optional

def create_synthetic_anomaly(
    clean_data: np.ndarray, 
    anomaly_type: Literal["RUIDO", "SPURIA", "DROPOUT", "BLOCKING"], 
    level_db: float
) -> np.ndarray:

    if anomaly_type == "RUIDO":
      return level_db * np.random.randn(clean_data.shape[0], clean_data.shape[1])
    
    elif anomaly_type == "SPURIA":
        noisy_data = clean_data.copy()
        for i in range(clean_data.shape[0]):
            idx_pico = np.random.randint(0, clean_data.shape[1])
            ancho_pico = 5
            for j in range(max(0, idx_pico-ancho_pico), min(clean_data.shape[1], idx_pico+ancho_pico+1)):
                distancia = abs(j - idx_pico)
                factor = np.exp(-distancia/2)
                noisy_data[i, j] += level_db * factor
        return noisy_data
    
    elif anomaly_type == "DROPOUT":
        noisy_data = clean_data.copy()
        for i in range(clean_data.shape[0]):
            duracion_puntos = int(clean_data.shape[1] * 0.1)
            inicio = np.random.randint(0, clean_data.shape[1] - duracion_puntos)
            fin = inicio + duracion_puntos
            for j in range(inicio, fin):
                posicion_relativa = abs(j - (inicio + fin)/2) / (duracion_puntos/2)
                factor_caida = 1 - np.cos(np.pi/2 * posicion_relativa)
                noisy_data[i, j] -= level_db * factor_caida
        return noisy_data
    
    elif anomaly_type == "BLOCKING":
        elevacion = level_db * (0.3 + 0.7 * np.random.rand(clean_data.shape[0], clean_data.shape[1]))
        return clean_data + elevacion


def build_composite_anomaly(
    clean_data: np.ndarray,
    noise_types: Optional[List[str]] = None,
    levels: Optional[List[int]] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Build a composite anomaly set splitting rows uniformly across noise types/levels.

    Keeps total rows equal to input. Distributes remainders fairly across leading combos.
    """
    if noise_types is None:
        noise_types = ["RUIDO", "SPURIA", "DROPOUT", "BLOCKING"]
    if levels is None:
        levels = [1, 3, 5, 7]

    combos = [(t, l) for t in noise_types for l in levels]
    num_rows = clean_data.shape[0]
    if num_rows == 0 or len(combos) == 0:
        return np.empty_like(clean_data)

    base = num_rows // len(combos)
    remainder = num_rows % len(combos)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_rows)
    start = 0
    parts = []
    for i, (t, l) in enumerate(combos):
        count = base + (1 if i < remainder else 0)
        if count == 0:
            continue
        subset = clean_data[indices[start:start + count], :]
        parts.append(create_synthetic_anomaly(subset, t, l))
        start += count

    return np.vstack(parts) if parts else np.empty_like(clean_data)