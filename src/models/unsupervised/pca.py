import os
from typing import Literal
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


def train_pca_iforest(X_train: np.ndarray, frequency: Literal["SMA", "FM"] = "SMA"):
    pca = PCA(n_components=2, random_state=42)
    
    X_pca = pca.fit_transform(X_train)
    
    model = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        random_state=42
    )
    model.fit(X_pca)
    
    if not os.path.exists(f'src/models/saved/pca'):
        os.makedirs(f'src/models/saved/pca')
    
    joblib.dump(pca, f'src/models/saved/pca/pca_{frequency}.pkl')
    joblib.dump(model, f'src/models/saved/pca/iforest_{frequency}.pkl')
    
    return pca, model


def predict_pca_iforest(pca, model, X_test):
    X_pca = pca.transform(X_test)
    return model.predict(X_pca), model.decision_function(X_pca) 