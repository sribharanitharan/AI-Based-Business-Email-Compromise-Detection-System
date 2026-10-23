"""
behavioral.py — Behavioral baseline anomaly features.
"""
from typing import Dict, Any

def extract_behavioral_features(email_dict: Dict[str, Any]) -> Dict[str, float]:
    hour = email_dict.get("hour_sent", 12)
    off_hours = 1.0 if (hour < 7 or hour > 19) else 0.0
    first_time = 1.0 if email_dict.get("first_time_sender", False) else 0.0
    return {
        "off_hours_sent_flag": off_hours,
        "first_time_sender_flag": first_time
    }
