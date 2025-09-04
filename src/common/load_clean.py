import os
from typing import List
import pandas as pd
import numpy as np


def _load_txt(path: str) -> np.ndarray:
    df = pd.read_csv(path, sep='\t', decimal=',', header=0, names=['freq','nivel'])
    return df['nivel'].dropna().astype(float).values

def _collect(folder: str) -> List[np.ndarray]:
    data = []
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.endswith('.txt'):
                data.append(_load_txt(os.path.join(root, fn)))
    return data

def get_clean_data(folder: str) -> np.ndarray:
    va = _collect(folder) 
    gl = min(len(v) for v in va)
    return np.vstack([v[:gl] for v in va])
