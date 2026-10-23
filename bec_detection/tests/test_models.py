"""
test_models.py — Tests for ML, boosting, anomaly, and ensemble models
"""

import pytest
import numpy as np
from src.models.baseline.baseline_models import BaselineModelSuite
from src.models.anomaly.anomaly_detector import AnomalyDetectorSuite
from src.models.ensemble.stacking_ensemble import BECStackingEnsemble
from src.models.graph.interaction_graph import EmailInteractionGraph
from src.models.llm.llm_classifier import LLMBECClassifier


@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    # 60 samples, 25 features
    X = np.random.rand(60, 25).astype(np.float32)
    # Labels: 0 = Benign, 1 = BEC
    y = np.random.randint(0, 2, size=60)
    return X, y


def test_baseline_models_training(synthetic_data):
    X, y = synthetic_data
    suite = BaselineModelSuite()
    suite.train_all(X[:45], y[:45])
    metrics = suite.evaluate_all(X[45:], y[45:])

    assert "logistic_regression" in metrics
    assert "random_forest" in metrics
    assert "naive_bayes" in metrics
    assert "svm" in metrics
    for m in metrics.values():
        assert "roc_auc" in m
        assert "f1_score" in m


def test_anomaly_detector_training(synthetic_data):
    X, _ = synthetic_data
    detector = AnomalyDetectorSuite(contamination=0.1)
    detector.fit_benign(X[:30])
    score = detector.predict_anomaly_score(X[31])

    assert "isolation_forest_score" in score
    assert "combined_anomaly_score" in score
    assert 0.0 <= score["combined_anomaly_score"] <= 1.0


def test_stacking_ensemble_voting():
    ensemble = BECStackingEnsemble()
    preds = {
        "xgboost": 0.85,
        "lightgbm": 0.80,
        "catboost": 0.90,
        "random_forest": 0.75,
        "logistic_regression": 0.70,
        "anomaly": 0.65
    }
    result = ensemble.predict_weighted(preds)
    assert result["is_bec"] is True
    assert result["risk_level"] in ["HIGH", "CRITICAL"]
    assert 0.70 <= result["risk_score"] <= 1.0


def test_email_interaction_graph():
    graph = EmailInteractionGraph(vip_list=["cfo@acme.com"])
    # Initial benign traffic
    graph.add_interaction("alice@acme.com", "bob@acme.com")
    graph.add_interaction("bob@acme.com", "alice@acme.com")

    # Benign known contact
    analysis_benign = graph.analyze_interaction("alice@acme.com", "bob@acme.com")
    assert analysis_benign["is_new_edge"] is False
    assert analysis_benign["graph_anomaly_score"] < 0.5

    # Anomalous contact: external unknown contacting VIP CFO
    analysis_malicious = graph.analyze_interaction("attacker@spoofed.xyz", "cfo@acme.com")
    assert analysis_malicious["is_new_edge"] is True
    assert analysis_malicious["is_recipient_vip"] is True
    assert analysis_malicious["graph_anomaly_score"] >= 0.55


def test_llm_semantic_classifier():
    llm = LLMBECClassifier()
    sample = {
        "subject": "Urgent wire transfer instruction",
        "sender": "ceo@corp.com",
        "reply_to": "exec@gmail.com",
        "body": "Please wire transfer $45,000 to the updated account before noon."
    }
    res = llm.analyze_semantic(sample)
    assert res["is_bec"] is True
    assert res["bec_type"] in ["Invoice Fraud", "CEO Impersonation"]
