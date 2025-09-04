import joblib
import numpy as np
from src.pipelines.setup import Setup
from src.pipelines.report import Report
from src.pipelines.evaluate import Evaluate
from src.common.load_clean import get_clean_data
from src.common.load_anomaly import create_synthetic_anomaly
from src.models.unsupervised.elliptic_envelope import predict_elliptic_envelope, train_elliptic_envelope

class EllipticEnvelope:
    def __init__(self) -> None:
       
        setup = Setup()
        self.report = Report()
        self.evaluator = Evaluate()
        clean_sma = get_clean_data("data/frecs/SMA")
        clean_fm = get_clean_data("data/frecs/FM")

        anomaly_sma = create_synthetic_anomaly(clean_sma, "RUIDO", 10)
        anomaly_fm = create_synthetic_anomaly(clean_fm, "RUIDO", 10)

        self.sma_train, self.sma_clean_test, self.sma_anomaly_test, self.sma_test = setup.get_train_data_unsupervised(clean_sma, anomaly_sma)
        self.fm_train, self.fm_clean_test, self.fm_anomaly_test, self.fm_test = setup.get_train_data_unsupervised(clean_fm, anomaly_fm)

    def test(self) -> None:
        try:
            model_sma = joblib.load('src/models/saved/elliptic_envelope/elliptic_envelope_SMA.pkl')
        except FileNotFoundError:
            model_sma = train_elliptic_envelope(self.sma_train, frequency="SMA")
            
        try:
            model_fm = joblib.load('src/models/saved/elliptic_envelope/elliptic_envelope_FM.pkl')
        except FileNotFoundError:
            model_fm = train_elliptic_envelope(self.fm_train, frequency="FM")

        y_true_sma = np.concatenate([np.zeros(len(self.sma_clean_test)), np.ones(len(self.sma_anomaly_test))])
        y_true_fm = np.concatenate([np.zeros(len(self.fm_clean_test)), np.ones(len(self.fm_anomaly_test))])
        
        pred_sma = predict_elliptic_envelope(model_sma, self.sma_test)
        pred_fm = predict_elliptic_envelope(model_fm, self.fm_test)
        
        results_sma = self.evaluator.evaluate_unsupervised(y_true_sma, pred_sma, pred_sma)
        results_fm = self.evaluator.evaluate_unsupervised(y_true_fm, pred_fm, pred_fm)
        
        self.report.generate_unsupervised_report('elliptic_envelope', 'SMA', results_sma)
        self.report.generate_unsupervised_report('elliptic_envelope', 'FM', results_fm)