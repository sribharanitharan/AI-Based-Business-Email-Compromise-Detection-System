"""
url_features.py — URL, domain reputation, and typosquatting features.
"""
import re
from typing import Dict, Any, List

def extract_url_features(email_dict: Dict[str, Any]) -> Dict[str, float]:
    urls = email_dict.get("urls", []) or []
    ip_in_url = 1.0 if any(re.search(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", u) for u in urls) else 0.0
    return {
        "url_count": float(len(urls)),
        "ip_in_url_flag": ip_in_url,
        "external_link_count": float(len(urls))
    }
