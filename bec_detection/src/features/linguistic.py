"""
linguistic.py — Deception, urgency, and sentiment tone analysis.
"""
from typing import Dict, Any

URGENCY_WORDS = {"urgent", "immediately", "asap", "emergency", "action required", "critical", "deadline"}

def extract_linguistic_features(email_dict: Dict[str, Any]) -> Dict[str, float]:
    subject = str(email_dict.get("subject", "")).lower()
    body = str(email_dict.get("body", "")).lower()

    subj_urg = min(sum(1 for w in URGENCY_WORDS if w in subject) / 2.0, 1.0)
    body_urg = min(sum(1 for w in URGENCY_WORDS if w in body) / 4.0, 1.0)

    orig_subj = str(email_dict.get("subject", ""))
    letters = [c for c in orig_subj if c.isalpha()]
    caps_ratio = (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0

    return {
        "subject_urgency_score": subj_urg,
        "body_urgency_score": body_urg,
        "all_caps_ratio": caps_ratio
    }
