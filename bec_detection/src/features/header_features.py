"""
header_features.py — Email header and authentication security features.
"""
from typing import Dict, Any

def extract_header_features(email_dict: Dict[str, Any]) -> Dict[str, float]:
    sender = str(email_dict.get("sender", "")).lower()
    reply_to = str(email_dict.get("reply_to", "")).lower()
    headers = email_dict.get("headers", {}) or {}

    s_dom = sender.split("@")[-1].strip(">").strip() if "@" in sender else ""
    r_dom = reply_to.split("@")[-1].strip(">").strip() if "@" in reply_to else ""

    mismatch = 1.0 if (r_dom and s_dom and r_dom != s_dom) else 0.0
    spf_fail = 1.0 if str(headers.get("spf", "")).lower() in ["fail", "softfail"] else 0.0
    dkim_fail = 1.0 if str(headers.get("dkim", "")).lower() in ["fail", "neutral"] else 0.0
    dmarc_fail = 1.0 if str(headers.get("dmarc", "")).lower() in ["fail", "none"] else 0.0

    return {
        "reply_to_mismatch": mismatch,
        "spf_fail_flag": spf_fail,
        "dkim_fail_flag": dkim_fail,
        "dmarc_fail_flag": dmarc_fail
    }
