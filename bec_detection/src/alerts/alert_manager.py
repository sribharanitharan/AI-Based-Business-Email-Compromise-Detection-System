"""
alert_manager.py — Alerting & Incident Response Dispatcher

Channels:
  - Webhooks (Slack incoming webhook, MS Teams, SIEM generic webhook)
  - SMTP Email alerts to SOC analysts
  - Local incident logging for SIEM ingestion (JSON lines)
"""

import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional
import httpx


class AlertManager:
    """Manages real-time incident alerting when high-confidence BEC is detected."""

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        smtp_config: Optional[Dict[str, Any]] = None,
        min_risk_level: str = "HIGH"
    ):
        self.webhook_url = webhook_url or os.getenv("ALERT_WEBHOOK_URL", "")
        self.smtp_config = smtp_config or {}
        self.min_risk_level = min_risk_level
        self._severity_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

    def should_alert(self, risk_level: str) -> bool:
        """Check if risk level meets alert dispatch threshold."""
        return self._severity_order.get(risk_level, 0) >= self._severity_order.get(self.min_risk_level, 2)

    async def send_webhook(self, incident: Dict[str, Any]) -> bool:
        """Send incident alert to Slack / Teams / SIEM webhook."""
        if not self.webhook_url:
            return False

        payload = {
            "title": f"🚨 [BEC ALERT] {incident.get('risk_level', 'HIGH')} Risk Email Detected",
            "text": (
                f"*Subject:* {incident.get('subject', 'N/A')}\n"
                f"*Sender:* {incident.get('sender', 'N/A')}\n"
                f"*Risk Score:* {incident.get('risk_score', 0):.2%}\n"
                f"*Threat Type:* {incident.get('bec_type', 'General BEC')}\n"
                f"*Reasoning:* {incident.get('reasoning', 'Suspicious indicators detected.')}"
            ),
            "severity": incident.get("risk_level", "HIGH"),
            "timestamp": incident.get("timestamp")
        }

        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                res = await client.post(self.webhook_url, json=payload)
                return res.status_code in [200, 204]
        except Exception:
            return False

    def send_smtp_alert(self, incident: Dict[str, Any], recipient: str) -> bool:
        """Send formatted alert email via SMTP."""
        if not self.smtp_config or not self.smtp_config.get("host"):
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[BEC Shield Alert] {incident.get('risk_level')} Risk: {incident.get('subject')}"
            msg["From"] = self.smtp_config.get("user", "bec-shield@corporate.internal")
            msg["To"] = recipient

            body = (
                f"BEC Shield detected an email with high risk score ({incident.get('risk_score'):.2%}).\n\n"
                f"Sender: {incident.get('sender')}\n"
                f"Subject: {incident.get('subject')}\n"
                f"BEC Category: {incident.get('bec_type')}\n"
                f"Action Taken: Flagged for SOC Review\n"
            )
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(self.smtp_config.get("host"), self.smtp_config.get("port", 587)) as server:
                if self.smtp_config.get("tls", True):
                    server.starttls()
                if self.smtp_config.get("pass"):
                    server.login(self.smtp_config["user"], self.smtp_config["pass"])
                server.sendmail(msg["From"], [recipient], msg.as_string())
            return True
        except Exception:
            return False
