"""
metadata_forensics.py — Forensics on attachments, MIME structural headers, and mail clients.
"""
from typing import Dict, Any

SUSPICIOUS_EXTENSIONS = {".exe", ".scr", ".vbs", ".bat", ".cmd", ".ps1", ".jar", ".iso", ".zip", ".rar"}

def extract_metadata_features(email_dict: Dict[str, Any]) -> Dict[str, float]:
    attachments = email_dict.get("attachments", []) or []
    has_susp = 1.0 if any(any(att.lower().endswith(ext) for ext in SUSPICIOUS_EXTENSIONS) for att in attachments) else 0.0
    return {
        "attachment_count": float(len(attachments)),
        "attachment_suspicious_ext": has_susp
    }
