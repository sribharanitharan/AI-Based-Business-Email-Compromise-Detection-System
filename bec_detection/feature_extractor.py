"""
feature_extractor.py
Extracts all features from an email dictionary:
{subject, body, from_addr, to_addr, reply_to, headers, urls}
"""

import re
import math
import string
import editdistance

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
    VADER_OK = True
except Exception:
    VADER_OK = False

try:
    import textstat
    TEXTSTAT_OK = True
except Exception:
    TEXTSTAT_OK = False

# ── Keyword lists ────────────────────────────────────────────────────────────
URGENCY_WORDS = [
    "urgent","immediately","asap","right away","time sensitive","critical",
    "emergency","deadline","overdue","today","now","must","required","quickly",
    "without delay","prompt","instantly","at once"
]
FINANCIAL_WORDS = [
    "wire","transfer","payment","invoice","account","routing","ach","swift",
    "bank","funds","amount","transaction","remittance","payable","deposit",
    "beneficiary","balance","credit","debit","fiscal","payroll"
]
SECRECY_WORDS = [
    "confidential","do not share","don't tell","keep quiet","don't forward",
    "between us","private","discreet","secret","sensitive"
]
AUTHORITY_WORDS = [
    "ceo","cfo","president","director","board","executive","management",
    "per request","on behalf","authorized","mandate","instruct"
]
FREE_PROVIDERS = [
    "gmail.com","yahoo.com","hotmail.com","outlook.com","aol.com",
    "protonmail.com","icloud.com","mail.com","yandex.com","zoho.com"
]
KNOWN_COMPANIES = [
    "microsoft","google","amazon","apple","paypal","facebook","netflix",
    "bank","wells","chase","citibank","barclays","hsbc"
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def _count_keywords(text: str, keywords: list) -> int:
    text_l = text.lower()
    return sum(1 for kw in keywords if kw in text_l)

def _domain(email_addr: str) -> str:
    email_addr = email_addr or ""
    m = re.search(r'@([\w.\-]+)', email_addr)
    return m.group(1).lower() if m else ""

def _entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    n = len(text)
    return -sum((f/n) * math.log2(f/n) for f in freq.values())

def _lookalike_score(domain: str, targets: list) -> float:
    """Min Levenshtein distance to any known company domain."""
    if not domain:
        return 0.0
    scores = [editdistance.eval(domain, t) for t in targets]
    min_dist = min(scores)
    # Normalize: 0=exact match (suspicious), >5=probably unrelated
    return max(0.0, 1.0 - min_dist / 6.0)

# ── Main extraction ──────────────────────────────────────────────────────────

def extract_features(email: dict) -> dict:
    subject  = str(email.get("subject", ""))
    body     = str(email.get("body", ""))
    from_    = str(email.get("from_addr", "") or email.get("sender", ""))
    to_      = str(email.get("to_addr", "") or email.get("to", ""))
    reply_to = str(email.get("reply_to", ""))
    urls     = email.get("urls", [])

    full_text = subject + " " + body

    from_domain    = _domain(from_)
    reply_domain   = _domain(reply_to)
    to_domain      = _domain(to_)

    # ── Content features ─────────────────────────────────────────────────────
    f = {}
    f["text_len"]            = len(full_text)
    f["word_count"]          = len(full_text.split())
    f["urgency_count"]       = _count_keywords(full_text, URGENCY_WORDS)
    f["financial_count"]     = _count_keywords(full_text, FINANCIAL_WORDS)
    f["secrecy_count"]       = _count_keywords(full_text, SECRECY_WORDS)
    f["authority_count"]     = _count_keywords(full_text, AUTHORITY_WORDS)
    f["caps_ratio"]          = sum(1 for c in full_text if c.isupper()) / max(len(full_text),1)
    f["exclamation_count"]   = full_text.count("!")
    f["question_count"]      = full_text.count("?")
    f["dollar_count"]        = full_text.count("$")
    f["number_count"]        = len(re.findall(r'\b\d{4,}\b', full_text))  # large numbers
    f["body_entropy"]        = _entropy(body[:500])

    # ── Sentiment ─────────────────────────────────────────────────────────────
    if VADER_OK:
        scores = vader.polarity_scores(full_text)
        f["sentiment_neg"]  = scores["neg"]
        f["sentiment_pos"]  = scores["pos"]
        f["sentiment_comp"] = scores["compound"]
    else:
        f["sentiment_neg"]  = 0.0
        f["sentiment_pos"]  = 0.0
        f["sentiment_comp"] = 0.0

    # ── Readability ───────────────────────────────────────────────────────────
    if TEXTSTAT_OK and len(body.split()) > 10:
        try:
            f["readability"] = textstat.flesch_reading_ease(body)
        except Exception:
            f["readability"] = 50.0
    else:
        f["readability"] = 50.0

    # ── Header / domain features ──────────────────────────────────────────────
    f["reply_to_mismatch"]   = 1 if reply_to and from_domain != reply_domain else 0
    f["from_free_provider"]  = 1 if from_domain in FREE_PROVIDERS else 0
    f["reply_free_provider"] = 1 if reply_domain in FREE_PROVIDERS else 0
    f["domain_lookalike"]    = _lookalike_score(from_domain, KNOWN_COMPANIES)
    f["domain_entropy"]      = _entropy(from_domain)

    # Cross-org: sender domain different from recipient domain (common in BEC)
    f["cross_org_flag"]      = 1 if from_domain and to_domain and from_domain != to_domain else 0

    # ── URL features ──────────────────────────────────────────────────────────
    f["url_count"]           = len(urls)
    f["http_url_count"]      = sum(1 for u in urls if u.startswith("http://"))
    f["ip_url_count"]        = sum(1 for u in urls if re.search(r'http://\d+\.\d+', u))
    f["short_url_count"]     = sum(1 for u in urls if any(s in u for s in ["bit.ly","tinyurl","goo.gl","t.co","ow.ly"]))
    f["url_entropy_avg"]     = (sum(_entropy(u) for u in urls) / len(urls)) if urls else 0.0

    # ── Social engineering composite ──────────────────────────────────────────
    f["social_eng_score"] = (
        min(f["urgency_count"],   5) * 0.25 +
        min(f["secrecy_count"],   3) * 0.30 +
        min(f["authority_count"], 3) * 0.20 +
        min(f["financial_count"], 5) * 0.25
    )

    return f


FEATURE_NAMES = list(extract_features({
    "subject": "", "body": "", "from_addr": "", "to_addr": "",
    "reply_to": "", "urls": []
}).keys())
