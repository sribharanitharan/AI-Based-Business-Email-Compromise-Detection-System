"""
imap_connector.py — Real-time email stream ingestion via IMAP/SSL.

Connects to corporate mailboxes (Exchange, Gmail, Office 365, generic IMAP),
polls for unread messages, extracts RFC 822 bytes, and passes to the parser.
"""

import imaplib
import email
from typing import List, Dict, Any, Generator, Optional
from .email_parser import EmailParser


class IMAPConnector:
    """Connects to IMAP mail server and streams unread emails."""

    def __init__(self, host: str, port: int = 993, user: str = "", password: str = "", mailbox: str = "INBOX"):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.mailbox = mailbox
        self.client: Optional[imaplib.IMAP4_SSL] = None
        self.parser = EmailParser()

    def connect(self) -> bool:
        """Establish SSL connection and log in."""
        try:
            self.client = imaplib.IMAP4_SSL(self.host, self.port)
            self.client.login(self.user, self.password)
            self.client.select(self.mailbox)
            return True
        except Exception:
            return False

    def fetch_unread(self, mark_as_read: bool = False) -> List[Dict[str, Any]]:
        """Fetch all unread messages and parse them into structured email dictionaries."""
        if not self.client:
            if not self.connect():
                return []

        try:
            status, response = self.client.search(None, "UNSEEN")
            if status != "OK" or not response[0]:
                return []

            parsed_emails = []
            for num in response[0].split():
                fetch_cmd = "(RFC822)" if mark_as_read else "(BODY.PEEK[])"
                res, data = self.client.fetch(num, fetch_cmd)
                if res == "OK":
                    raw_bytes = data[0][1]
                    parsed = self.parser.parse_bytes(raw_bytes)
                    parsed["imap_id"] = num.decode()
                    parsed_emails.append(parsed)

            return parsed_emails
        except Exception:
            return []

    def disconnect(self):
        """Close mailbox and logout."""
        if self.client:
            try:
                self.client.close()
                self.client.logout()
            except Exception:
                pass
            self.client = None
