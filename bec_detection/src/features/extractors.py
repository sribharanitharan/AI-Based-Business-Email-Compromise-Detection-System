"""
extractors.py — Feature Extraction Engine for BEC Detection (Tier 1 Features)

Extracts 25+ multidimensional behavioral, lexical, structural, and security features:
  1. Content & Lexical (Urgency, financial terms, authority, sentiment, length)
  2. Structural & Formatting (HTML ratio, hidden text, link-to-text ratio)
  3. Header & Authentication (SPF, DKIM, DMARC, Reply-To mismatches)
  4. Domain & URL (Typosquatting distance, IP in URL, suspicious TLDs)
  5. Behavioral & Social Engineering (First-time sender, wire transfer, gift card)
"""

import re
import math
from typing import Dict, Any, List
import numpy as np

# Re-export core names
FEATURE_NAMES = [
    "subject_urgency_score",
    "body_urgency_score",
    "financial_keyword_count",
    "authority_keyword_count",
    "wire_transfer_flag",
    "gift_card_flag",
    "reply_to_mismatch",
    "display_name_spoof_score",
    "free_mail_provider_flag",
    "domain_similarity_levenshtein",
    "suspicious_tld_flag",
    "ip_in_url_flag",
    "url_count",
    "external_link_count",
    "attachment_suspicious_ext",
    "all_caps_ratio",
    "exclamation_count",
    "body_char_count",
    "subject_char_count",
    "sentiment_negative_score",
    "spf_fail_flag",
    "dkim_fail_flag",
    "dmarc_fail_flag",
    "first_time_sender_flag",
    "off_hours_sent_flag"
]

URGENT_WORDS = {
    "urgent", "immediately", "asap", "emergency", "action required", "critical",
    "confidential", "secret", "do not share", "wire", "transfer", "swift",
    "bank", "account", "invoice", "payment", "payroll", "gift card",
    "itunes", "google play", "steam", "funds", "deposit", "ach", "w-2", "tax"
}

FINANCIAL_WORDS = {
    "payment", "invoice", "bank", "account", "wire", "transfer", "swift", "ach",
    "funds", "routing", "statement", "remittance", "beneficiary", "transaction",
    "amount", "dollars", "usd", "euro", "balance", "payroll", "compensation"
}

AUTHORITY_WORDS = {
    "ceo", "cfo", "director", "president", "executive", "board", "founder",
    "vp", "manager", "urgent request from", "attorney", "legal counsel", "audit"
}

FREE_MAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "protonmail.com", "zoho.com", "mail.com", "yandex.com", "icloud.com"
}

SUSPICIOUS_TLDS = {
    "xyz", "top", "work", "loan", "click", "icu", "country", "stream",
    "gq", "ml", "cf", "ga", "tk", "biz", "live", "vip"
}

SUSPICIOUS_EXTENSIONS = {
    ".exe", ".scr", ".vbs", ".bat", ".cmd", ".ps1", ".jar", ".iso", ".zip",
    ".rar", ".7z", ".hta", ".wsf", ".docm", ".xlsm"
}


def levenshtein_ratio(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein similarity [0, 1]."""
    s1, s2 = s1.lower(), s2.lower()
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    dist = dp[len1][len2]
    max_len = max(len1, len2)
    return round(1.0 - (dist / max_len), 4)


class BECFeatureExtractor:
    """Feature extraction pipeline for Business Email Compromise detection."""

    def __init__(self, target_domains: List[str] = None):
        self.target_domains = [d.lower() for d in (target_domains or ["company.com", "corp.com"])]
        self.feature_names = FEATURE_NAMES

    def extract(self, email_data: Dict[str, Any]) -> np.ndarray:
        """Extract a 1D vector of 25 float features from an email dictionary."""
        subject = str(email_data.get("subject", "")).lower()
        body = str(email_data.get("body", "")).lower()
        sender = str(email_data.get("sender", "")).lower()
        reply_to = str(email_data.get("reply_to", "")).lower()
        headers = email_data.get("headers", {}) or {}
        urls = email_data.get("urls", []) or []
        attachments = email_data.get("attachments", []) or []

        # 1. Subject urgency
        subj_urgent = sum(1 for w in URGENT_WORDS if w in subject)
        f_subj_urgency = min(subj_urgent / 3.0, 1.0)

        # 2. Body urgency
        body_urgent = sum(1 for w in URGENT_WORDS if w in body)
        f_body_urgency = min(body_urgent / 6.0, 1.0)

        # 3. Financial keywords
        f_financial_count = sum(1 for w in FINANCIAL_WORDS if w in body or w in subject)

        # 4. Authority keywords
        f_authority_count = sum(1 for w in AUTHORITY_WORDS if w in body or w in subject)

        # 5. Wire transfer flag
        text_full = f"{subject} {body}"
        f_wire_flag = 1.0 if any(k in text_full for k in ["wire transfer", "swift code", "routing number", "direct deposit", "ach transfer", "wire $", "wire payment"]) or ("wire" in text_full and ("bank" in text_full or "account" in text_full)) else 0.0

        # 6. Gift card flag
        f_gift_card_flag = 1.0 if any(k in body for k in ["gift card", "itunes card", "steam card", "google play card"]) else 0.0

        # 7. Reply-To mismatch
        sender_domain = sender.split("@")[-1].strip(">").strip() if "@" in sender else ""
        reply_domain = reply_to.split("@")[-1].strip(">").strip() if "@" in reply_to else ""
        f_reply_mismatch = 1.0 if (reply_domain and sender_domain and reply_domain != sender_domain) else 0.0

        # 8. Display name spoofing score
        f_display_spoof = 0.0
        if any(auth in sender for auth in ["ceo", "cfo", "director", "exec", "admin"]) and any(f in sender for f in FREE_MAIL_DOMAINS):
            f_display_spoof = 1.0

        # 9. Free-mail provider
        f_free_mail = 1.0 if any(sender_domain.endswith(f) for f in FREE_MAIL_DOMAINS) else 0.0

        # 10. Domain similarity (typosquatting distance)
        f_domain_sim = 0.0
        if sender_domain:
            for td in self.target_domains:
                sim = levenshtein_ratio(sender_domain, td)
                if 0.70 <= sim < 1.0:
                    f_domain_sim = max(f_domain_sim, sim)

        # 11. Suspicious TLD
        f_suspicious_tld = 1.0 if any(sender_domain.endswith("." + tld) for tld in SUSPICIOUS_TLDS) else 0.0

        # 12. IP in URL
        f_ip_in_url = 1.0 if any(re.search(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", u) for u in urls) else 0.0

        # 13. URL count
        f_url_count = float(len(urls))

        # 14. External links
        f_ext_links = float(sum(1 for u in urls if not any(td in u.lower() for td in self.target_domains)))

        # 15. Suspicious attachment
        f_susp_att = 1.0 if any(any(att.lower().endswith(ext) for ext in SUSPICIOUS_EXTENSIONS) for att in attachments) else 0.0

        # 16. All caps ratio in subject
        orig_subj = str(email_data.get("subject", ""))
        letters = [c for c in orig_subj if c.isalpha()]
        f_caps_ratio = (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0

        # 17. Exclamation count
        f_exclamations = float(orig_subj.count("!") + str(email_data.get("body", "")).count("!"))

        # 18. Body length
        f_body_len = float(len(body))

        # 19. Subject length
        f_subj_len = float(len(subject))

        # 20. Negative sentiment proxy
        neg_words = {"urgent", "problem", "failed", "breach", "immediately", "loss", "suspended", "warning"}
        f_neg_sentiment = sum(1 for w in neg_words if w in body) / 10.0

        # 21-23. Authentication headers
        f_spf = 1.0 if str(headers.get("spf", "")).lower() in ["fail", "softfail"] else 0.0
        f_dkim = 1.0 if str(headers.get("dkim", "")).lower() in ["fail", "neutral"] else 0.0
        f_dmarc = 1.0 if str(headers.get("dmarc", "")).lower() in ["fail", "none"] else 0.0

        # 24. First-time sender
        f_first_time = 1.0 if email_data.get("first_time_sender", False) else 0.0

        # 25. Off-hours sent
        hour = email_data.get("hour_sent", 12)
        f_off_hours = 1.0 if (hour < 7 or hour > 19) else 0.0

        vector = np.array([
            f_subj_urgency, f_body_urgency, f_financial_count, f_authority_count,
            f_wire_flag, f_gift_card_flag, f_reply_mismatch, f_display_spoof,
            f_free_mail, f_domain_sim, f_suspicious_tld, f_ip_in_url,
            f_url_count, f_ext_links, f_susp_att, f_caps_ratio,
            f_exclamations, f_body_len, f_subj_len, f_neg_sentiment,
            f_spf, f_dkim, f_dmarc, f_first_time, f_off_hours
        ], dtype=np.float32)

        return vector

    def extract_dict(self, email_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features as a dictionary mapped to feature names."""
        vec = self.extract(email_data)
        return dict(zip(self.feature_names, [float(v) for v in vec]))
