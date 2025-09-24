import os
import numpy as np
import joblib
from typing import Dict, Any
from src.common.load_anomaly import build_composite_anomaly
from src.models.unsupervised import isolation_forest
from src.models.unsupervised.isolation_forest import train_isolation_forest
from src.pipelines.evaluate import Evaluate
from src.common.load_clean import get_clean_data
from src.pipelines.report import Report
from src.pipelines.setup import Setup

class IsolationForest:
    def __init__(self) -> None:
        setup = Setup()
        self.report = Report()
        clean_sma = get_clean_data("data/frecs/SMA")
        clean_fm = get_clean_data("data/frecs/FM")

        anomaly_sma = build_composite_anomaly(clean_sma)
        anomaly_fm = build_composite_anomaly(clean_fm)

        self.sma_train, self.sma_clean_test, self.sma_anomaly_test, self.sma_test = setup.get_train_data_unsupervised(clean_sma, anomaly_sma)
        self.fm_train, self.fm_clean_test, self.fm_anomaly_test, self.fm_test = setup.get_train_data_unsupervised(clean_fm, anomaly_fm)

    def test(self) -> None:
        evaluator = Evaluate()
        
        try:
            model_sma = joblib.load('src/models/saved/isolation_forest/isolation_forest_SMA.pkl')
        except FileNotFoundError:
            model_sma = train_isolation_forest(self.sma_train, frequency="SMA")
            
        try:
            model_fm = joblib.load('src/models/saved/isolation_forest/isolation_forest_FM.pkl')
        except FileNotFoundError:
            model_fm = train_isolation_forest(self.fm_train, frequency="FM")

        y_true_sma = np.concatenate([np.zeros(len(self.sma_clean_test)), np.ones(len(self.sma_anomaly_test))])
        y_true_fm = np.concatenate([np.zeros(len(self.fm_clean_test)), np.ones(len(self.fm_anomaly_test))])
        
        pred_sma, scores_sma = isolation_forest.predict_isolation_forest(model_sma, self.sma_test)
        pred_fm, scores_fm = isolation_forest.predict_isolation_forest(model_fm, self.fm_test)
        
        results_sma = evaluator.evaluate_unsupervised(y_true_sma, pred_sma, scores_sma)
        results_fm = evaluator.evaluate_unsupervised(y_true_fm, pred_fm, scores_fm)
        
        self.report.generate_unsupervised_report('isolation_forest', 'SMA', results_sma)
        self.report.generate_unsupervised_report('isolation_forest', 'FM', results_fm)

    def train(self) -> None:
        train_isolation_forest(self.sma_train, frequency="SMA")
        train_isolation_forest(self.fm_train, frequency="FM")