"""
stacking_ensemble.py — Meta-Learner Stacking & Soft Voting Ensemble

Combines predictions from multiple distinct model families:
  1. Gradient Boosting (XGBoost, LightGBM, CatBoost)
  2. Classical Baselines (Logistic Regression, Random Forest, SVM)
  3. Deep Learning (Bi-LSTM / TextCNN)
  4. Unsupervised Anomaly (Isolation Forest, Autoencoder)
  5. Semantic / LLM Intent Engine
Uses a Logistic Regression meta-learner or weighted soft voting with confidence estimation.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression


DEFAULT_WEIGHTS = {
    "xgboost": 0.25,
    "lightgbm": 0.20,
    "catboost": 0.15,
    "random_forest": 0.15,
    "logistic_regression": 0.10,
    "anomaly": 0.15
}


class BECStackingEnsemble:
    """Stacking and weighted ensemble combining base models into a single robust BEC verdict."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or DEFAULT_WEIGHTS
        self.meta_learner = LogisticRegression(max_iter=500, random_state=42)
        self.is_meta_trained = False

    def predict_weighted(self, model_predictions: Dict[str, float]) -> Dict[str, Any]:
        """
        Combine individual model probabilities using normalized soft voting.
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for model_name, prob in model_predictions.items():
            w = self.weights.get(model_name, 0.10)
            weighted_sum += float(prob) * w
            total_weight += w

        final_prob = (weighted_sum / total_weight) if total_weight > 0 else 0.5
        final_prob = round(float(np.clip(final_prob, 0.0, 1.0)), 4)

        # Risk classification
        if final_prob >= 0.75:
            risk_level = "CRITICAL"
        elif final_prob >= 0.55:
            risk_level = "HIGH"
        elif final_prob >= 0.30:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        is_bec = final_prob >= 0.50

        # Confidence: distance from uncertainty boundary (0.5)
        confidence = round(abs(final_prob - 0.5) * 2.0, 4)

        return {
            "risk_score": final_prob,
            "risk_level": risk_level,
            "is_bec": is_bec,
            "confidence": confidence,
            "model_contributions": model_predictions
        }

    def fit_meta_learner(self, meta_features: np.ndarray, y: np.ndarray):
        """Train meta-classifier on out-of-fold prediction probabilities."""
        self.meta_learner.fit(meta_features, y)
        self.is_meta_trained = True

    def predict_meta(self, meta_features: np.ndarray) -> np.ndarray:
        """Predict using trained meta learner."""
        if not self.is_meta_trained:
            return np.mean(meta_features, axis=1)
        return self.meta_learner.predict_proba(meta_features)[:, 1]

    def save(self, filepath: str):
        """Save meta learner to disk."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.meta_learner, p)

    def load(self, filepath: str):
        """Load meta learner from disk."""
        p = Path(filepath)
        if p.exists():
            self.meta_learner = joblib.load(p)
            self.is_meta_trained = True
