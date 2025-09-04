from src.runners.supervised.random_forest import RandomForest
from src.runners.unsupervised.autoencoder import Autoencoder
from src.runners.unsupervised.elliptic_envelope import EllipticEnvelope
from src.runners.unsupervised.isolation_forest import IsolationForest
from src.runners.unsupervised.lof import LOF
from src.runners.unsupervised.pca import PCA
from src.runners.supervised.knn import KNN
from src.runners.supervised.mlp import MLP
from src.runners.supervised.xgboost import XGBoost

isolation_forest = IsolationForest()
# isolation_forest.test()

# pca = PCA()
# pca.test()

# lof = LOF()
# lof.test()

#autoencoder = Autoencoder()
#autoencoder.test()

#elliptic_envelope = EllipticEnvelope()
#elliptic_envelope.test()

# knn = KNN()
# knn.test()

# random_forest = RandomForest()
# random_forest.test()

#xgboost = XGBoost()
#xgboost.test()

mlp = MLP()
mlp.test()