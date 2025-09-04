import os
from typing import Literal
import joblib
from sklearn.covariance import EllipticEnvelope
import numpy as np

def train_elliptic_envelope(X_train: np.ndarray, frequency: Literal["SMA", "FM"] = "SMA"):
    ee = EllipticEnvelope(contamination=0.1)
    ee.fit(X_train)

    if not os.path.exists(f'src/models/saved/elliptic_envelope'):
        os.makedirs(f'src/models/saved/elliptic_envelope')

    joblib.dump(ee, f'src/models/saved/elliptic_envelope/elliptic_envelope_{frequency}.pkl')
    return ee

def predict_elliptic_envelope(model, X_test: np.ndarray):
    return model.predict(X_test)
