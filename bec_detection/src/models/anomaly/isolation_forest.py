"""
isolation_forest.py — Isolation Forest Anomaly Model for BEC.
"""
from sklearn.ensemble import IsolationForest

def create_isolation_forest(contamination: float = 0.05):
    return IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=42,
        n_jobs=1
    )
