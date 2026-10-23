"""
logistic_regression.py — Standalone Logistic Regression model with calibration.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

def create_logistic_regression():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42))
    ])
