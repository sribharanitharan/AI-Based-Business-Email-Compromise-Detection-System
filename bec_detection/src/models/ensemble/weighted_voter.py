"""
weighted_voter.py — Dynamic weight soft voting ensemble.
"""
from typing import Dict, Any
import numpy as np

def soft_vote(model_probas: Dict[str, float], weights: Dict[str, float]) -> float:
    total_w = sum(weights.get(m, 0.1) for m in model_probas)
    weighted_sum = sum(model_probas[m] * weights.get(m, 0.1) for m in model_probas)
    return round(float(weighted_sum / total_w) if total_w > 0 else 0.5, 4)
