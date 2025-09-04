import joblib
import numpy as np
from src.common.load_anomaly import create_synthetic_anomaly
from src.common.load_clean import get_clean_data
from src.models.supervised.knn import predict_knn, train_knn
from src.pipelines.evaluate import Evaluate
from src.pipelines.report import Report
from src.pipelines.setup import Setup

class KNN:
    def __init__(self) -> None:
        setup = Setup()
        self.report = Report()
        self.evaluator = Evaluate()
        clean_sma = get_clean_data("data/frecs/SMA")
        clean_fm = get_clean_data("data/frecs/FM")

        anomaly_sma = create_synthetic_anomaly(clean_sma, "RUIDO", 10)
        anomaly_fm = create_synthetic_anomaly(clean_fm, "RUIDO", 10)
        
        self.sma_x_train, self.sma_x_test, self.sma_y_train, self.sma_y_test = setup.get_train_data_supervised(clean_sma, anomaly_sma)
        self.fm_x_train, self.fm_x_test, self.fm_y_train, self.fm_y_test = setup.get_train_data_supervised(clean_fm, anomaly_fm)

    def test(self) -> None:
        try:
            model_sma = joblib.load('src/models/saved/knn/knn_SMA.pkl')
        except FileNotFoundError:
            model_sma = train_knn(self.sma_x_train, self.sma_y_train, frequency="SMA")
            
        try:
            model_fm = joblib.load('src/models/saved/knn/knn_FM.pkl')
        except FileNotFoundError:
            model_fm = train_knn(self.fm_x_train, self.fm_y_train, frequency="FM")

        pred_sma = predict_knn(model_sma, self.sma_x_test)
        pred_fm = predict_knn(model_fm, self.fm_x_test)
        
        results_sma = self.evaluator.evaluate_supervised(self.sma_y_test, pred_sma)
        results_fm = self.evaluator.evaluate_supervised(self.fm_y_test, pred_fm)

        self.report.generate_supervised_report('knn', 'SMA', results_sma)
        self.report.generate_supervised_report('knn', 'FM', results_fm)

