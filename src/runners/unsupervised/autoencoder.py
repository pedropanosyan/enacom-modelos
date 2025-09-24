import joblib
import numpy as np
from src.common.load_anomaly import build_composite_anomaly
from src.common.load_clean import get_clean_data
from src.models.unsupervised.autoencoder import predict_autoencoder, train_autoencoder
from src.pipelines.evaluate import Evaluate
from src.pipelines.report import Report
from src.pipelines.setup import Setup


class Autoencoder:
    def __init__(self) -> None:
        setup = Setup()
        self.report = Report()
        self.evaluator = Evaluate()
        clean_sma = get_clean_data("data/frecs/SMA")
        clean_fm = get_clean_data("data/frecs/FM")

        anomaly_sma = build_composite_anomaly(clean_sma)
        anomaly_fm = build_composite_anomaly(clean_fm)

        self.sma_train, self.sma_clean_test, self.sma_anomaly_test, self.sma_test = setup.get_train_data_unsupervised(clean_sma, anomaly_sma)
        self.fm_train, self.fm_clean_test, self.fm_anomaly_test, self.fm_test = setup.get_train_data_unsupervised(clean_fm, anomaly_fm)


    def test(self) -> None:
        try:
            model_sma = joblib.load('src/models/saved/autoencoder/autoencoder_SMA.pkl')
            scaler_sma = joblib.load('src/models/saved/autoencoder/scaler_SMA.pkl')
        except FileNotFoundError:
            scaler_sma, model_sma = train_autoencoder(self.sma_train, frequency="SMA")
            
        try:
            model_fm = joblib.load('src/models/saved/autoencoder/autoencoder_FM.pkl')
            scaler_fm = joblib.load('src/models/saved/autoencoder/scaler_FM.pkl')
        except FileNotFoundError:
            scaler_fm, model_fm = train_autoencoder(self.fm_train, frequency="FM")

        y_true_sma = np.concatenate([np.zeros(len(self.sma_clean_test)), np.ones(len(self.sma_anomaly_test))])
        y_true_fm = np.concatenate([np.zeros(len(self.fm_clean_test)), np.ones(len(self.fm_anomaly_test))])
        
        pred_sma, scores_sma = predict_autoencoder(scaler_sma, model_sma, self.sma_test)
        pred_fm, scores_fm = predict_autoencoder(scaler_fm, model_fm, self.fm_test)
        
        results_sma = self.evaluator.evaluate_unsupervised(y_true_sma, pred_sma, scores_sma)
        results_fm = self.evaluator.evaluate_unsupervised(y_true_fm, pred_fm, scores_fm)
        
        self.report.generate_unsupervised_report('autoencoder', 'SMA', results_sma)
        self.report.generate_unsupervised_report('autoencoder', 'FM', results_fm)

    def train(self) -> None:
        train_autoencoder(self.sma_train, frequency="SMA")
        train_autoencoder(self.fm_train, frequency="FM")