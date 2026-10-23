"""
email_parser.py — Ingestion and normalization of raw email sources.

Supports:
  - Raw .eml file parsing (email.parser.BytesParser)
  - RFC 822 string parsing
  - Authentication header parsing (Authentication-Results, Received-SPF, DKIM-Signature)
  - URL and attachment extraction
  - Thread chain decomposition
"""

import email
from email import policy
from email.parser import BytesParser, Parser
from email.utils import parsedate_to_datetime
import re
from pathlib import Path
from typing import Dict, Any, List, Optional


class EmailParser:
    """Parses RFC 822 / MIME emails into normalized dicts for BEC analysis."""

    def __init__(self):
        self.url_pattern = re.compile(
            r'https?://(?:[a-zA-Z0-9-._~:/?#\[\]@!$&\'()*+,;=]|%[0-9a-fA-F]{2})+'
        )

    def parse_bytes(self, raw_bytes: bytes) -> Dict[str, Any]:
        """Parse raw email bytes."""
        msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)
        return self._normalize_message(msg)

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """Parse an .eml file on disk."""
        path = Path(file_path)
        with open(path, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)
        return self._normalize_message(msg)

    def parse_text(self, raw_text: str) -> Dict[str, Any]:
        """Parse raw RFC 822 text string."""
        msg = Parser(policy=policy.default).parsestr(raw_text)
        return self._normalize_message(msg)

    def _normalize_message(self, msg) -> Dict[str, Any]:
        """Normalize parsed email.message into standard dictionary."""
        subject = msg.get("Subject", "") or ""
        sender = msg.get("From", "") or ""
        to = msg.get("To", "") or ""
        reply_to = msg.get("Reply-To", "") or ""
        date_header = msg.get("Date", "") or ""

        # Hour sent
        hour_sent = 12
        if date_header:
            try:
                dt = parsedate_to_datetime(date_header)
                hour_sent = dt.hour
            except Exception:
                pass

        # Extract plain body and HTML
        body_parts = []
        html_parts = []
        attachments = []

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))

                if "attachment" in content_disposition:
                    filename = part.get_filename() or "unnamed_attachment"
                    attachments.append(filename)
                elif content_type == "text/plain":
                    try:
                        body_parts.append(part.get_content())
                    except Exception:
                        pass
                elif content_type == "text/html":
                    try:
                        html_parts.append(part.get_content())
                    except Exception:
                        pass
        else:
            try:
                body_parts.append(msg.get_content())
            except Exception:
                body_parts.append(str(msg.get_payload() or ""))

        full_body = "\n".join(body_parts)
        if not full_body.strip() and html_parts:
            # Strip tags fallback
            full_body = re.sub(r'<[^>]+>', ' ', "\n".join(html_parts))

        # URLs in body and html
        all_text = full_body + " " + " ".join(html_parts)
        urls = list(set(self.url_pattern.findall(all_text)))

        # Authentication-Results headers
        auth_results = msg.get("Authentication-Results", "") or ""
        spf = "pass" if "spf=pass" in auth_results.lower() else ("fail" if "spf=fail" in auth_results.lower() else "none")
        dkim = "pass" if "dkim=pass" in auth_results.lower() else ("fail" if "dkim=fail" in auth_results.lower() else "none")
        dmarc = "pass" if "dmarc=pass" in auth_results.lower() else ("fail" if "dmarc=fail" in auth_results.lower() else "none")

        received_spf = msg.get("Received-SPF", "") or ""
        if "pass" in received_spf.lower():
            spf = "pass"
        elif "fail" in received_spf.lower():
            spf = "fail"

        return {
            "subject": subject,
            "sender": sender,
            "to": to,
            "reply_to": reply_to,
            "body": full_body,
            "urls": urls,
            "attachments": attachments,
            "hour_sent": hour_sent,
            "headers": {
                "spf": spf,
                "dkim": dkim,
                "dmarc": dmarc,
                "raw_auth": auth_results
            }
        }
