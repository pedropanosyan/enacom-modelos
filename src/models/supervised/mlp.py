import os
from typing import Literal
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier


def train_mlp(X_train: np.ndarray, y_train: np.ndarray, frequency: Literal["SMA", "FM"] = "SMA"):
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, random_state=42)
    model.fit(X_train, y_train)
    
    if not os.path.exists(f'src/models/saved/mlp'):
        os.makedirs(f'src/models/saved/mlp')
    
    joblib.dump(model, f'src/models/saved/mlp/mlp_{frequency}.pkl')
    
    return model

def predict_mlp(model: MLPClassifier, X_test: np.ndarray):
    return model.predict(X_test)