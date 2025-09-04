import os
import joblib
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from typing import Literal

def train_lof(X_train: np.ndarray, frequency: Literal["SMA", "FM"] = "SMA"):
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
    lof.fit(X_train)

    if not os.path.exists(f'src/models/saved/lof'):
        os.makedirs(f'src/models/saved/lof')

    joblib.dump(lof, f'src/models/saved/lof/lof_{frequency}.pkl')
    return lof

def predict_lof(model, X_test):
    y_pred = model.predict(X_test)
    scores = model.decision_function(X_test)
    return y_pred, scores
