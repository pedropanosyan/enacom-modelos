import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Tuple, Literal

def train_isolation_forest(
    X_train: np.ndarray,
    n_estimators: int = 100, 
    max_samples: str = 'auto',
    frequency: Literal["SMA", "FM"] = "SMA"
) -> IsolationForest:
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
    )
    model.fit(X_train)
    joblib.dump(model, f'src/models/saved/isolation_forest/isolation_forest_{frequency}.pkl')
    return model

def predict_isolation_forest(model: IsolationForest, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return model.predict(X_test), model.decision_function(X_test)
