"""
random_forest.py — Random Forest classifier for BEC detection.
"""
from sklearn.ensemble import RandomForestClassifier

def create_random_forest():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    )
