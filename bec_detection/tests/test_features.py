"""
test_features.py — Unit tests for the BEC Feature Extraction Engine
"""

import pytest
import numpy as np
from src.features.extractors import BECFeatureExtractor, levenshtein_ratio, FEATURE_NAMES


def test_levenshtein_ratio():
    assert levenshtein_ratio("company.com", "company.com") == 1.0
    # Typosquatting
    ratio = levenshtein_ratio("c0mpany.com", "company.com")
    assert 0.8 <= ratio < 1.0
    assert levenshtein_ratio("", "") == 1.0
    assert levenshtein_ratio("a", "") == 0.0


def test_feature_extractor_dimensions():
    extractor = BECFeatureExtractor(target_domains=["acme.com"])
    sample = {
        "subject": "Urgent wire transfer needed immediately",
        "body": "Please wire $50,000 to the attached bank account. This is confidential from the CEO.",
        "sender": "ceo@acme-support.xyz",
        "reply_to": "attacker@gmail.com",
        "headers": {"spf": "fail", "dkim": "fail"},
        "urls": ["http://192.168.1.1/invoice"],
        "attachments": ["invoice.exe"],
        "first_time_sender": True,
        "hour_sent": 2
    }
    vec = extractor.extract(sample)
    assert isinstance(vec, np.ndarray)
    assert len(vec) == 25
    assert len(FEATURE_NAMES) == 25

    # Test dictionary mapping
    f_dict = extractor.extract_dict(sample)
    assert len(f_dict) == 25
    assert f_dict["wire_transfer_flag"] == 1.0
    assert f_dict["reply_to_mismatch"] == 1.0
    assert f_dict["ip_in_url_flag"] == 1.0
    assert f_dict["attachment_suspicious_ext"] == 1.0


def test_empty_email_handling():
    extractor = BECFeatureExtractor()
    empty = {}
    vec = extractor.extract(empty)
    assert len(vec) == 25
    assert not np.isnan(vec).any()
