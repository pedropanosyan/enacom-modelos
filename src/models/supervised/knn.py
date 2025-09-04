import os
from typing import Literal
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def train_knn(X_train: np.ndarray, y_train: np.ndarray, frequency: Literal["SMA", "FM"] = "SMA"):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    if not os.path.exists(f'src/models/saved/knn'):
        os.makedirs(f'src/models/saved/knn')

    joblib.dump(model, f'src/models/saved/knn/knn_{frequency}.pkl')
    return model

def predict_knn(model: KNeighborsClassifier, X_test: np.ndarray):
    return model.predict(X_test)