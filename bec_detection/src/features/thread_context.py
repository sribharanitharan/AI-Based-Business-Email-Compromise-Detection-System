"""
thread_context.py — Thread-level anomaly detection and topic hijacking detection.
"""
from typing import Dict, Any, List

def analyze_thread_context(thread_emails: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not thread_emails:
        return {"thread_length": 0, "has_hijack_indicator": False, "urgency_spike": 0.0}

    thread_len = len(thread_emails)
    last_email = thread_emails[-1]
    last_body = str(last_email.get("body", "")).lower()

    # Urgency spike in final message
    has_banking = any(w in last_body for w in ["wire", "account", "bank", "invoice", "routing", "payment"])
    first_senders = {str(e.get("sender", "")).lower() for e in thread_emails[:-1]}
    last_sender = str(last_email.get("sender", "")).lower()

    new_participant = last_sender not in first_senders if first_senders else False

    return {
        "thread_length": thread_len,
        "has_hijack_indicator": bool(new_participant and has_banking),
        "urgency_spike": 1.0 if has_banking else 0.0
    }
