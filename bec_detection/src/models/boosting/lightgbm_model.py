"""
lightgbm_model.py  —  LightGBM BEC classifier (Tier 2 Boosting)

Fast gradient boosting that handles large datasets well.
Uses early stopping on AUC metric.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    LGBM_OK = True
except ImportError:
    LGBM_OK = False

BASE_DIR = Path(__file__).parent.parent.parent  # bec_detection/


def get_lgb_params(scale_pos_weight: float = 10) -> dict:
    return {
        "objective":         "binary",
        "metric":            "auc",
        "boosting_type":     "gbdt",
        "num_leaves":        63,
        "learning_rate":     0.05,
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "scale_pos_weight":  scale_pos_weight,   # Handle class imbalance
        "min_child_samples": 5,
        "verbose":           -1,
        "n_jobs":            1,
    }


def train_lightgbm(X_train, y_train, X_val, y_val, feature_names=None):
    """
    Train a LightGBM binary classifier.

    Returns:
        model    – trained lgb.Booster
        val_f1   – F1 score on validation set
        val_auc  – AUC on validation set
    """
    if not LGBM_OK:
        raise ImportError("lightgbm is not installed. Run: pip install lightgbm")

    pos_ratio  = max(1, (y_train == 0).sum() / max((y_train == 1).sum(), 1))
    params     = get_lgb_params(scale_pos_weight=pos_ratio)

    train_data = lgb.Dataset(
        X_train, label=y_train,
        feature_name=feature_names if feature_names else "auto"
    )
    val_data   = lgb.Dataset(X_val, label=y_val, reference=train_data)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=-1),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    val_proba = model.predict(X_val)
    val_pred  = (val_proba >= 0.5).astype(int)
    val_f1    = f1_score(y_val, val_pred, zero_division=0)
    val_auc   = roc_auc_score(y_val, val_proba)

    print(f"    [LightGBM] F1={val_f1:.4f}  AUC={val_auc:.4f}  "
          f"Trees={model.num_trees()}")
    return model, val_f1, val_auc


def predict_lightgbm(model, X) -> np.ndarray:
    """Return class-1 probability array."""
    return model.predict(X)


def save_lightgbm(model, path: str = None):
    path = path or str(BASE_DIR / "models" / "lightgbm.pkl")
    joblib.dump(model, path)
    print(f"    [LightGBM] Saved → {path}")


def load_lightgbm(path: str = None):
    path = path or str(BASE_DIR / "models" / "lightgbm.pkl")
    return joblib.load(path)


if __name__ == "__main__":
    """Quick standalone test."""
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from feature_extractor import FEATURE_NAMES
    X = np.random.rand(200, len(FEATURE_NAMES)).astype(np.float32)
    y = np.array([0]*140 + [1]*60, dtype=np.int32)
    np.random.shuffle(y)
    model, f1, auc = train_lightgbm(X[:160], y[:160], X[160:], y[160:], FEATURE_NAMES)
    print(f"Test passed — F1={f1:.4f}  AUC={auc:.4f}")
