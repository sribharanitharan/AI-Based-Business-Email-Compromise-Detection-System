"""
catboost_model.py  —  CatBoost BEC classifier (Tier 2 Boosting)

Excellent for categorical features (sender domain, country codes, etc.).
Handles class imbalance via class_weights.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

try:
    from catboost import CatBoostClassifier, Pool
    CATBOOST_OK = True
except ImportError:
    CATBOOST_OK = False

BASE_DIR = Path(__file__).parent.parent.parent  # bec_detection/


def train_catboost(X_train, y_train, X_val, y_val,
                   cat_features_idx=None, feature_names=None):
    """
    Train a CatBoost binary classifier.

    Args:
        X_train, y_train  – training data
        X_val, y_val      – validation data
        cat_features_idx  – list of column indices that are categorical
        feature_names     – optional list of feature names

    Returns:
        model    – trained CatBoostClassifier
        val_f1   – F1 on validation set
        val_auc  – AUC on validation set
    """
    if not CATBOOST_OK:
        raise ImportError("catboost is not installed. Run: pip install catboost")

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        loss_function="Logloss",
        eval_metric="AUC",
        class_weights=[1, 10],        # Weight BEC class 10x
        early_stopping_rounds=50,
        cat_features=cat_features_idx or [],
        verbose=0,
        random_seed=42,
        thread_count=1,
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    val_proba = model.predict_proba(X_val)[:, 1]
    val_pred  = (val_proba >= 0.5).astype(int)
    val_f1    = f1_score(y_val, val_pred, zero_division=0)
    val_auc   = roc_auc_score(y_val, val_proba)

    print(f"    [CatBoost] F1={val_f1:.4f}  AUC={val_auc:.4f}  "
          f"Trees={model.tree_count_}")
    return model, val_f1, val_auc


def predict_catboost(model, X) -> np.ndarray:
    """Return class-1 probability array."""
    return model.predict_proba(X)[:, 1]


def save_catboost(model, path: str = None):
    path = path or str(BASE_DIR / "models" / "catboost.pkl")
    joblib.dump(model, path)
    print(f"    [CatBoost] Saved → {path}")


def load_catboost(path: str = None):
    path = path or str(BASE_DIR / "models" / "catboost.pkl")
    return joblib.load(path)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from feature_extractor import FEATURE_NAMES
    X = np.random.rand(200, len(FEATURE_NAMES)).astype(np.float32)
    y = np.array([0]*140 + [1]*60, dtype=np.int32)
    np.random.shuffle(y)
    model, f1, auc = train_catboost(X[:160], y[:160], X[160:], y[160:])
    print(f"Test passed — F1={f1:.4f}  AUC={auc:.4f}")
