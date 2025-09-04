import os
from typing import Literal
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, frequency: Literal["SMA", "FM"] = "SMA"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    if not os.path.exists(f'src/models/saved/random_forest'):
        os.makedirs(f'src/models/saved/random_forest')

    joblib.dump(model, f'src/models/saved/random_forest/random_forest_{frequency}.pkl')
    return model

def predict_random_forest(model: RandomForestClassifier, X_test: np.ndarray):
    return model.predict(X_test)