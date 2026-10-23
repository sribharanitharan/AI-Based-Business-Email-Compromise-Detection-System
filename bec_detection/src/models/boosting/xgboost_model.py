"""
xgboost_model.py — Standalone XGBoost classifier with imbalance weighting.
"""
import xgboost as xgb

def create_xgboost_classifier():
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=2,
        eval_metric="logloss",
        nthread=1,
        random_state=42,
        verbosity=0
    )
