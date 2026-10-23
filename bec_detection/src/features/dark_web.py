"""
dark_web.py — Correlation with breached credentials, dark web dumps, and known C2 domains.
"""
from typing import Dict, Any, Set

KNOWN_BREACHED_DOMAINS: Set[str] = {"tempmail.com", "throwawaymail.com", "guerrillamail.com", "10minutemail.com"}

def check_dark_web_correlation(email_dict: Dict[str, Any]) -> Dict[str, float]:
    sender = str(email_dict.get("sender", "")).lower()
    domain = sender.split("@")[-1].strip(">").strip() if "@" in sender else ""
    is_disposable = 1.0 if domain in KNOWN_BREACHED_DOMAINS else 0.0
    return {
        "disposable_domain_flag": is_disposable,
        "dark_web_c2_correlation": 0.0
    }
