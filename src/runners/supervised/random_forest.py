import joblib
import numpy as np
from src.common.load_anomaly import build_composite_anomaly
from src.common.load_clean import get_clean_data
from src.models.supervised.random_forest import predict_random_forest, train_random_forest
from src.pipelines.evaluate import Evaluate
from src.pipelines.report import Report
from src.pipelines.setup import Setup

class RandomForest:
    def __init__(self) -> None:
        setup = Setup()
        self.report = Report()
        self.evaluator = Evaluate()
        clean_sma = get_clean_data("data/frecs/SMA")
        clean_fm = get_clean_data("data/frecs/FM")

        anomaly_sma = build_composite_anomaly(clean_sma)
        anomaly_fm = build_composite_anomaly(clean_fm)

        self.sma_x_train, self.sma_x_test, self.sma_y_train, self.sma_y_test = setup.get_train_data_supervised(clean_sma, anomaly_sma)
        self.fm_x_train, self.fm_x_test, self.fm_y_train, self.fm_y_test = setup.get_train_data_supervised(clean_fm, anomaly_fm)

    def test(self) -> None:
        try:
            model_sma = joblib.load('src/models/saved/random_forest/random_forest_SMA.pkl')
        except FileNotFoundError:
            model_sma = train_random_forest(self.sma_x_train, self.sma_y_train, frequency="SMA")
            
        try:
            model_fm = joblib.load('src/models/saved/random_forest/random_forest_FM.pkl')
        except FileNotFoundError:
            model_fm = train_random_forest(self.fm_x_train, self.fm_y_train, frequency="FM")

        
        pred_sma = predict_random_forest(model_sma, self.sma_x_test)
        pred_fm = predict_random_forest(model_fm, self.fm_x_test)
        
        results_sma = self.evaluator.evaluate_supervised(self.sma_y_test, pred_sma)
        results_fm = self.evaluator.evaluate_supervised(self.fm_y_test, pred_fm)
        
        self.report.generate_supervised_report('random_forest', 'SMA', results_sma)
        self.report.generate_supervised_report('random_forest', 'FM', results_fm)

    def train(self) -> None:
        train_random_forest(self.sma_x_train, self.sma_y_train, frequency="SMA")
        train_random_forest(self.fm_x_train, self.fm_y_train, frequency="FM")
