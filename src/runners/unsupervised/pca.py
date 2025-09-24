import joblib
import numpy as np
from src.models.unsupervised.pca import predict_pca_iforest, train_pca_iforest
from src.pipelines.evaluate import Evaluate
from src.pipelines.report import Report
from src.pipelines.setup import Setup
from src.common.load_clean import get_clean_data
from src.common.load_anomaly import build_composite_anomaly


class PCA:
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
            pca_sma = joblib.load('src/models/saved/pca/pca_SMA.pkl')
            model_sma = joblib.load('src/models/saved/pca/iforest_SMA.pkl')
        except FileNotFoundError:
            pca_sma, model_sma = train_pca_iforest(self.sma_train, frequency="SMA")
            
        try:
            pca_fm = joblib.load('src/models/saved/pca/pca_FM.pkl')
            model_fm = joblib.load('src/models/saved/pca/iforest_FM.pkl')
        except FileNotFoundError:
            pca_fm, model_fm = train_pca_iforest(self.fm_train, frequency="FM")

        y_true_sma = np.concatenate([np.zeros(len(self.sma_clean_test)), np.ones(len(self.sma_anomaly_test))])
        y_true_fm = np.concatenate([np.zeros(len(self.fm_clean_test)), np.ones(len(self.fm_anomaly_test))])
        
        pred_sma, scores_sma = predict_pca_iforest(pca_sma, model_sma, self.sma_test)
        pred_fm, scores_fm = predict_pca_iforest(pca_fm, model_fm, self.fm_test)
        
        results_sma = self.evaluator.evaluate_unsupervised(y_true_sma, pred_sma, scores_sma)
        results_fm = self.evaluator.evaluate_unsupervised(y_true_fm, pred_fm, scores_fm)
        
        self.report.generate_unsupervised_report('pca', 'SMA', results_sma)
        self.report.generate_unsupervised_report('pca', 'FM', results_fm)