"""
lime_explainer.py — LIME (Local Interpretable Model-agnostic Explanations) for BEC

Generates local explanations for individual email risk scores:
  - Identifies which specific features nudged the prediction towards BEC vs Benign
  - Supports tabular feature perturbation
  - Formats outputs cleanly for analyst review and dashboard rendering
"""

from typing import Dict, Any, List, Callable, Optional
import numpy as np

try:
    from lime import lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False


class BECLimeExplainer:
    """LIME tabular explainer wrapper for BEC machine learning models."""

    def __init__(
        self,
        training_data: np.ndarray,
        feature_names: List[str],
        class_names: List[str] = None,
        mode: str = "classification"
    ):
        self.feature_names = feature_names
        self.class_names = class_names or ["Benign", "BEC"]
        self.explainer = None

        if HAS_LIME and training_data is not None and len(training_data) > 0:
            try:
                self.explainer = lime_tabular.LimeTabularExplainer(
                    training_data=training_data,
                    feature_names=feature_names,
                    class_names=self.class_names,
                    mode=mode,
                    random_state=42
                )
            except Exception:
                self.explainer = None

    def explain_instance(
        self,
        instance_vec: np.ndarray,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        num_features: int = 8
    ) -> Dict[str, Any]:
        """
        Explain a single feature vector instance using LIME.
        Returns a dict of feature names, weights, and directional impact.
        """
        if instance_vec.ndim > 1:
            instance_vec = instance_vec.flatten()

        explanations = []

        if self.explainer is not None:
            try:
                exp = self.explainer.explain_instance(
                    data_row=instance_vec,
                    predict_fn=predict_fn,
                    num_features=num_features,
                    labels=[1]
                )
                exp_list = exp.as_list(label=1)
                for feat_rule, weight in exp_list:
                    explanations.append({
                        "feature_rule": feat_rule,
                        "weight": round(float(weight), 4),
                        "direction": "supports_bec" if weight > 0 else "supports_benign"
                    })
                return {
                    "method": "LIME",
                    "num_features_explained": len(explanations),
                    "explanations": explanations
                }
            except Exception:
                pass

        # Fallback heuristic explanation if LIME library fails or has no background data
        top_indices = np.argsort(np.abs(instance_vec))[-num_features:][::-1]
        for idx in top_indices:
            val = float(instance_vec[idx])
            name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
            if val > 0:
                explanations.append({
                    "feature_rule": f"{name} = {val:.2f}",
                    "weight": round(val * 0.15, 4),
                    "direction": "supports_bec"
                })

        return {
            "method": "heuristic_fallback",
            "num_features_explained": len(explanations),
            "explanations": explanations
        }
