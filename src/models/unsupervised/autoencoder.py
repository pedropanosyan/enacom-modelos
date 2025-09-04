import os
from typing import Literal
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import numpy as np

def train_autoencoder(X_train: np.ndarray, frequency: Literal["SMA", "FM"] = "SMA"):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(8,),
        max_iter=200,
        random_state=42,
        verbose=0
    )
    
    autoencoder.fit(X_scaled, X_scaled)

    if not os.path.exists(f'src/models/saved/autoencoder'):
        os.makedirs(f'src/models/saved/autoencoder')

    joblib.dump(scaler, f'src/models/saved/autoencoder/scaler_{frequency}.pkl')
    joblib.dump(autoencoder, f'src/models/saved/autoencoder/autoencoder_{frequency}.pkl')
    
    return scaler, autoencoder

def predict_autoencoder(scaler, autoencoder, X_test: np.ndarray):
    X_scaled = scaler.transform(X_test)
    predictions = autoencoder.predict(X_scaled)
    
    # Calculate reconstruction error as anomaly scores
    reconstruction_error = np.mean((X_scaled - predictions) ** 2, axis=1)
    
    # Convert to binary predictions (1 = anomaly, 0 = normal)
    # Using a threshold based on the 95th percentile of reconstruction errors
    threshold = np.percentile(reconstruction_error, 95)
    binary_predictions = np.where(reconstruction_error > threshold, -1, 1)
    
    return binary_predictions, reconstruction_error