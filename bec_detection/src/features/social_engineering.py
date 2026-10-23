"""
social_engineering.py — Social engineering manipulation pattern detection.
"""
from typing import Dict, Any

AUTHORITY_WORDS = {"ceo", "cfo", "director", "president", "executive", "board", "founder", "manager"}

def extract_social_engineering_features(email_dict: Dict[str, Any]) -> Dict[str, float]:
    body = str(email_dict.get("body", "")).lower()
    subject = str(email_dict.get("subject", "")).lower()
    sender = str(email_dict.get("sender", "")).lower()

    wire_flag = 1.0 if any(k in body or k in subject for k in ["wire transfer", "swift", "routing number", "direct deposit", "ach transfer"]) else 0.0
    gift_flag = 1.0 if any(k in body for k in ["gift card", "itunes card", "steam card"]) else 0.0
    auth_count = float(sum(1 for w in AUTHORITY_WORDS if w in body or w in subject or w in sender))

    return {
        "wire_transfer_flag": wire_flag,
        "gift_card_flag": gift_flag,
        "authority_keyword_count": auth_count
    }
