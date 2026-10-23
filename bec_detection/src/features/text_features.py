"""
text_features.py — Text and lexical feature extractor.
"""
from typing import Dict, Any

FINANCIAL_TERMS = {"payment", "invoice", "bank", "account", "wire", "transfer", "swift", "ach", "dollars", "usd"}

def extract_text_features(email_dict: Dict[str, Any]) -> Dict[str, float]:
    subject = str(email_dict.get("subject", "")).lower()
    body = str(email_dict.get("body", "")).lower()
    return {
        "body_char_count": float(len(body)),
        "subject_char_count": float(len(subject)),
        "financial_keyword_count": float(sum(1 for w in FINANCIAL_TERMS if w in body or w in subject)),
        "exclamation_count": float(subject.count("!") + body.count("!"))
    }
