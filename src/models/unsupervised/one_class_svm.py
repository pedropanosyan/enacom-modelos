import os
import joblib
import numpy as np
from sklearn.svm import OneClassSVM
from typing import Literal

def train_svm(X_train: np.ndarray, frequency: Literal["SMA", "FM"] = "SMA"):
    model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.01)
    model.fit(X_train)

    if not os.path.exists(f'src/models/saved/one_class_svm'):
        os.makedirs(f'src/models/saved/one_class_svm')

    joblib.dump(model, f'src/models/saved/one_class_svm/one_class_svm_{frequency}.pkl')
    return model

def predict_svm(model, X_test: np.ndarray):
    return model.predict(X_test)
