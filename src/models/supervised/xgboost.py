import os
from typing import Literal
import joblib
import numpy as np
from xgboost import XGBClassifier

def train_xgboost(X_train: np.ndarray, y_train: np.ndarray, frequency: Literal["SMA", "FM"] = "SMA"):
    model = XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    if not os.path.exists(f'src/models/saved/xgboost'):
        os.makedirs(f'src/models/saved/xgboost')

    joblib.dump(model, f'src/models/saved/xgboost/xgboost_{frequency}.pkl')
    return model

def predict_xgboost(model: XGBClassifier, X_test: np.ndarray):
    return model.predict(X_test)