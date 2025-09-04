import numpy as np
from typing import Literal

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