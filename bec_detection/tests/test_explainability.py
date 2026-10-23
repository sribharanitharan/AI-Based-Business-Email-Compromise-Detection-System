"""
test_explainability.py — Unit tests for LIME and SHAP interpretability modules
"""

import pytest
import numpy as np
from src.features.extractors import FEATURE_NAMES
from src.explainability.lime_explainer import BECLimeExplainer
from src.explainability.shap_explainer import BECShapExplainer
from sklearn.ensemble import RandomForestClassifier


def test_lime_explainer():
    X_train = np.random.rand(50, 25).astype(np.float32)
    y_train = np.random.randint(0, 2, size=50)

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)

    explainer = BECLimeExplainer(
        training_data=X_train,
        feature_names=FEATURE_NAMES
    )

    sample = X_train[0]
    res = explainer.explain_instance(sample, clf.predict_proba, num_features=5)

    assert "explanations" in res
    assert len(res["explanations"]) > 0
    assert "feature_rule" in res["explanations"][0]


def test_shap_explainer():
    X_train = np.random.rand(40, 25).astype(np.float32)
    y_train = np.random.randint(0, 2, size=40)

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)

    shap_exp = BECShapExplainer(
        model=clf,
        feature_names=FEATURE_NAMES,
        background_data=X_train
    )

    sample = X_train[1]
    attributions = shap_exp.explain_instance(sample, top_n=5)

    assert isinstance(attributions, list)
    assert len(attributions) <= 5
    if len(attributions) > 0:
        assert "feature" in attributions[0]
        assert "shap_value" in attributions[0]
