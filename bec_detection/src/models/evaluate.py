"""
evaluate.py — Cross-validation and model evaluation metrics calculator.
"""
from typing import Dict, Any
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score

def evaluate_predictions(y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_pred_proba)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
    }
