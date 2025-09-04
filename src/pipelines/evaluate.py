import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import Dict, Any


class Evaluate:
    def __init__(self) -> None:
        pass

    def evaluate_unsupervised(self, y_true: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
        y_pred = np.where(predictions == -1, 1, 0)
        
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        
        tn, fp, fn, tp = conf_matrix.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'confusion_matrix': conf_matrix,
            'classification_report': report,
            'anomaly_scores': scores,
            'n_anomalies_detected': np.sum(y_pred),
            'n_anomalies_actual': np.sum(y_true)
        }
    
    def evaluate_supervised(self, y_true: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
        accuracy = accuracy_score(y_true, predictions)
        conf_matrix = confusion_matrix(y_true, predictions)
        report = classification_report(y_true, predictions, output_dict=True)
        
        tn, fp, fn, tp = conf_matrix.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'confusion_matrix': conf_matrix,
            'classification_report': report,
            'n_anomalies_detected': np.sum(predictions),
            'n_anomalies_actual': np.sum(y_true)
        }