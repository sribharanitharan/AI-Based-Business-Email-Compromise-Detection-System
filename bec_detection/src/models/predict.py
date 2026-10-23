"""
predict.py — Offline batch and streaming prediction runner.
"""
from typing import Dict, Any, List
import numpy as np

def run_prediction(email_dict: Dict[str, Any]) -> Dict[str, Any]:
    from src.features.extractors import BECFeatureExtractor
    from src.models.ensemble.stacking_ensemble import BECStackingEnsemble
    extractor = BECFeatureExtractor()
    ensemble = BECStackingEnsemble()
    feat_vec = extractor.extract(email_dict)
    # Basic prediction
    return ensemble.predict_weighted({"xgboost": 0.5, "lightgbm": 0.5})
