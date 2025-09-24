import argparse
import sys

# Supervised runners
from src.runners.supervised.knn import KNN
from src.runners.supervised.mlp import MLP
from src.runners.supervised.random_forest import RandomForest
from src.runners.supervised.xgboost import XGBoost

# Unsupervised runners
from src.runners.unsupervised.isolation_forest import IsolationForest
from src.runners.unsupervised.lof import LOF
from src.runners.unsupervised.elliptic_envelope import EllipticEnvelope
from src.runners.unsupervised.autoencoder import Autoencoder
from src.runners.unsupervised.pca import PCA

# Train functions (save inside)
from src.models.supervised.knn import train_knn
from src.models.supervised.mlp import train_mlp
from src.models.supervised.random_forest import train_random_forest
from src.models.supervised.xgboost import train_xgboost

from src.models.unsupervised.isolation_forest import train_isolation_forest
from src.models.unsupervised.lof import train_lof
from src.models.unsupervised.elliptic_envelope import train_elliptic_envelope
from src.models.unsupervised.autoencoder import train_autoencoder
from src.models.unsupervised.pca import train_pca_iforest


def get_all_runners():
    supervised = {
        "knn": KNN,
        "mlp": MLP,
        "random_forest": RandomForest,
        "xgboost": XGBoost,
    }
    unsupervised = {
        "isolation_forest": IsolationForest,
        "lof": LOF,
        "elliptic_envelope": EllipticEnvelope,
        "autoencoder": Autoencoder,
        "pca_iforest": PCA,
    }
    return supervised, unsupervised


def main():
    parser = argparse.ArgumentParser(description="Run all model runners to train/save models")
    parser.add_argument(
        "--which",
        choices=["all", "supervised", "unsupervised"],
        default="all",
        help="Subset of runners to execute",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional explicit list of models to run (by key name)",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        default="train",
        help="Whether to only train and save, or run test (train+eval)",
    )
    args = parser.parse_args()

    supervised, unsupervised = get_all_runners()

    if args.which in ("all", "supervised"):
        selected = supervised
        if args.models:
            selected = {name: supervised[name] for name in args.models if name in supervised}
        print("Training supervised models:" if args.mode == "train" else "Testing supervised models:")
        for name, Runner in selected.items():
            try:
                print(f"- {name} ...", end="", flush=True)
                runner = Runner()  # prepares data
                if args.mode == "test":
                    runner.test()
                else:
                    if name == "knn":
                        train_knn(runner.sma_x_train, runner.sma_y_train, frequency="SMA")
                        train_knn(runner.fm_x_train, runner.fm_y_train, frequency="FM")
                    elif name == "mlp":
                        train_mlp(runner.sma_x_train, runner.sma_y_train, frequency="SMA")
                        train_mlp(runner.fm_x_train, runner.fm_y_train, frequency="FM")
                    elif name == "random_forest":
                        train_random_forest(runner.sma_x_train, runner.sma_y_train, frequency="SMA")
                        train_random_forest(runner.fm_x_train, runner.fm_y_train, frequency="FM")
                    elif name == "xgboost":
                        train_xgboost(runner.sma_x_train, runner.sma_y_train, frequency="SMA")
                        train_xgboost(runner.fm_x_train, runner.fm_y_train, frequency="FM")
                print(" done")
            except Exception as exc:
                print(" failed")
                print(f"  Error in {name}: {exc}")

    if args.which in ("all", "unsupervised"):
        selected = unsupervised
        if args.models:
            selected = {name: unsupervised[name] for name in args.models if name in unsupervised}
        print("Training unsupervised models:" if args.mode == "train" else "Testing unsupervised models:")
        for name, Runner in selected.items():
            try:
                print(f"- {name} ...", end="", flush=True)
                runner = Runner()  # prepares data
                if args.mode == "test":
                    runner.test()
                else:
                    if name == "isolation_forest":
                        train_isolation_forest(runner.sma_train, frequency="SMA")
                        train_isolation_forest(runner.fm_train, frequency="FM")
                    elif name == "lof":
                        train_lof(runner.sma_train, frequency="SMA")
                        train_lof(runner.fm_train, frequency="FM")
                    elif name == "elliptic_envelope":
                        train_elliptic_envelope(runner.sma_train, frequency="SMA")
                        train_elliptic_envelope(runner.fm_train, frequency="FM")
                    elif name == "autoencoder":
                        train_autoencoder(runner.sma_train, frequency="SMA")
                        train_autoencoder(runner.fm_train, frequency="FM")
                    elif name == "pca_iforest":
                        train_pca_iforest(runner.sma_train, frequency="SMA")
                        train_pca_iforest(runner.fm_train, frequency="FM")
                print(" done")
            except Exception as exc:
                print(" failed")
                print(f"  Error in {name}: {exc}")


if __name__ == "__main__":
    sys.exit(main())


