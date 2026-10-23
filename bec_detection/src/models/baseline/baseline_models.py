"""
baseline_models.py — Tier 1 Classical Machine Learning Baselines

Models:
  - Logistic Regression (L2 / ElasticNet)
  - Random Forest Classifier (Gini / Entropy)
  - Gaussian & Multinomial Naive Bayes
  - Support Vector Classifier (RBF & Linear SVM)
"""

from typing import Dict, Any, Tuple
import numpy as np
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, f1_score


class BaselineModelSuite:
    """Suite of classical ML models for BEC feature classification."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1
            ),
            "naive_bayes": GaussianNB(),
            "svm": SVC(
                probability=True,
                kernel="rbf",
                class_weight="balanced",
                random_state=random_state
            )
        }

    def train_all(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Fit all baseline models on training feature matrix."""
        trained = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            trained[name] = model
        return trained

    def evaluate_all(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Evaluate all baseline models returning ROC-AUC, F1, Precision, Recall."""
        metrics = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                auc = float(roc_auc_score(y_test, y_prob))
            else:
                auc = 0.0

            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            metrics[name] = {
                "roc_auc": round(auc, 4),
                "f1_score": round(f1, 4)
            }
        return metrics

    def save(self, output_dir: str):
        """Save models to directory."""
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, p / f"{name}.joblib")

    def load(self, model_dir: str):
        """Load trained models from directory."""
        p = Path(model_dir)
        for name in list(self.models.keys()):
            f = p / f"{name}.joblib"
            if f.exists():
                self.models[name] = joblib.load(f)
