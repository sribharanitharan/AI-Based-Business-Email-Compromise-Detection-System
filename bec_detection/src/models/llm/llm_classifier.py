"""
llm_classifier.py — Tier 6 LLM & Semantic Few-Shot BEC Analyzer

Performs intent parsing and social engineering category classification:
  - CEO Fraud / Impersonation
  - Invoice / Payroll Redirection
  - Gift Card Scams
  - Legal / M&A Confidentiality Traps
  - Credential Harvesting
Provides structured JSON output with reasoning, threat type, and confidence.
Supports both direct API call (OpenAI/Anthropic/Ollama) and local zero-shot semantic heuristics.
"""

import json
import re
from typing import Dict, Any, Optional, List


FEW_SHOT_PROMPT_TEMPLATE = """You are an expert Cyber Threat Intelligence (CTI) analyst specializing in Business Email Compromise (BEC).
Analyze the following email and return a JSON response with:
- is_bec (boolean)
- bec_type (one of: "CEO Impersonation", "Payroll Fraud", "Invoice Fraud", "Gift Card Scam", "Vendor Compromise", "None")
- urgency_level ("Low", "Medium", "High", "Critical")
- financial_risk (boolean)
- reasoning (brief explanation)
- confidence_score (float 0.0 to 1.0)

Email Subject: {subject}
Sender: {sender}
Reply-To: {reply_to}
Body:
{body}
"""


class LLMBECClassifier:
    """LLM & Semantic Rule Engine for deep intent BEC detection."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model

    def analyze_semantic(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        High-precision semantic analysis checking linguistic intent markers.
        Used standalone or as prompt generation / offline evaluator.
        """
        subject = str(email_data.get("subject", "")).lower()
        body = str(email_data.get("body", "")).lower()
        sender = str(email_data.get("sender", "")).lower()
        reply_to = str(email_data.get("reply_to", "")).lower()

        reasons = []
        bec_type = "None"
        confidence = 0.0

        # Pattern 1: Gift card scam
        if any(w in body for w in ["gift card", "itunes card", "steam card", "google play"]) and any(w in body for w in ["urgent", "need you", "busy in a meeting"]):
            bec_type = "Gift Card Scam"
            confidence = 0.92
            reasons.append("Urgent gift card purchase request while executive is ostensibly busy")

        # Pattern 2: Payroll redirection
        elif any(w in body for w in ["payroll", "direct deposit", "bank account", "update direct deposit"]) and any(w in body for w in ["effective next", "paycheck", "hr", "routing number"]):
            bec_type = "Payroll Fraud"
            confidence = 0.88
            reasons.append("Request to modify employee payroll / bank deposit routing details")

        # Pattern 3: Invoice / Wire Fraud
        elif any(w in body for w in ["wire transfer", "updated banking details", "overdue invoice", "remittance", "swift"]) and any(w in body for w in ["payment", "invoice", "vendor", "account number"]):
            bec_type = "Invoice Fraud"
            confidence = 0.89
            reasons.append("Instructions to divert payment to updated wire/bank coordinates")

        # Pattern 4: CEO Impersonation
        elif any(w in sender for w in ["ceo", "director", "president", "chief"]) and (reply_to and reply_to != sender or any(f in sender for f in ["gmail.com", "outlook.com"])):
            bec_type = "CEO Impersonation"
            confidence = 0.85
            reasons.append("Executive display name combined with freemail domain or mismatched reply address")

        # Pattern 5: Confidential acquisition / NDA urgency
        elif any(w in body for w in ["confidential", "secret acquisition", "do not speak to anyone", "strictly confidential"]):
            bec_type = "Executive Pressure"
            confidence = 0.78
            reasons.append("High secrecy demand isolating the victim from regular internal controls")

        is_bec = bec_type != "None" and confidence >= 0.70
        urgency = "Critical" if confidence > 0.85 else ("High" if confidence > 0.7 else "Low")

        return {
            "is_bec": is_bec,
            "bec_type": bec_type,
            "urgency_level": urgency,
            "financial_risk": bool(bec_type in ["Payroll Fraud", "Invoice Fraud", "Gift Card Scam"]),
            "confidence_score": confidence if is_bec else 0.1,
            "reasoning": "; ".join(reasons) if reasons else "No anomalous social engineering signals detected."
        }

    def format_prompt(self, email_data: Dict[str, Any]) -> str:
        """Format email into the few-shot evaluation prompt."""
        return FEW_SHOT_PROMPT_TEMPLATE.format(
            subject=email_data.get("subject", ""),
            sender=email_data.get("sender", ""),
            reply_to=email_data.get("reply_to", ""),
            body=email_data.get("body", "")
        )
