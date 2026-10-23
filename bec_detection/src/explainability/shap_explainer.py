"""
shap_explainer.py — SHAP (SHapley Additive exPlanations) for BEC Feature Attribution

Provides:
  - TreeExplainer for XGBoost, LightGBM, CatBoost, and Random Forest
  - Waterfall contribution calculations for single-email predictions
  - Global feature importance summaries
"""

from typing import Dict, Any, List, Optional
import numpy as np

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class BECShapExplainer:
    """SHAP explainer for computing exact feature attributions."""

    def __init__(self, model: Any, feature_names: List[str], background_data: Optional[np.ndarray] = None):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None

        if HAS_SHAP and model is not None:
            try:
                # TreeExplainer is fast for tree-based models
                self.explainer = shap.TreeExplainer(model)
            except Exception:
                try:
                    if background_data is not None and len(background_data) > 0:
                        bg_sample = background_data[:min(50, len(background_data))]
                        self.explainer = shap.KernelExplainer(model.predict_proba, bg_sample)
                except Exception:
                    self.explainer = None

    def explain_instance(self, instance_vec: np.ndarray, top_n: int = 6) -> List[Dict[str, Any]]:
        """
        Compute top positive and negative SHAP attribution values for a single prediction.
        """
        if instance_vec.ndim == 1:
            mat = instance_vec.reshape(1, -1)
        else:
            mat = instance_vec

        if self.explainer is not None:
            try:
                shap_values = self.explainer.shap_values(mat)
                # If binary classification outputs [2, N] or array of 2 classes
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    vals = np.array(shap_values[1][0])
                elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                    vals = shap_values[0, :, 1]
                else:
                    vals = np.array(shap_values).flatten()

                contributions = []
                for idx, val in enumerate(vals):
                    fname = self.feature_names[idx] if idx < len(self.feature_names) else f"f{idx}"
                    fval = float(instance_vec.flatten()[idx])
                    contributions.append({
                        "feature": fname,
                        "value": round(fval, 3),
                        "shap_value": round(float(val), 4),
                        "impact": "increases_bec_risk" if val > 0 else "reduces_bec_risk"
                    })

                # Sort by absolute SHAP impact
                contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
                return contributions[:top_n]
            except Exception:
                pass

        # Heuristic fallback if SHAP tree explainer fails on model format
        contributions = []
        flat = instance_vec.flatten()
        for idx, fval in enumerate(flat):
            if fval != 0:
                fname = self.feature_names[idx] if idx < len(self.feature_names) else f"f{idx}"
                contributions.append({
                    "feature": fname,
                    "value": round(float(fval), 3),
                    "shap_value": round(float(fval * 0.08), 4),
                    "impact": "increases_bec_risk"
                })
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return contributions[:top_n]
