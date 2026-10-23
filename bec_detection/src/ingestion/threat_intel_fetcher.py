"""
threat_intel_fetcher.py — Automated fetching and caching of external IOC / threat intelligence feeds.

Sources:
  - OpenPhish public feed
  - URLhaus recent malware URLs
  - AbuseIPDB blacklist
  - Custom local IOC watchlists
"""

import os
from pathlib import Path
from typing import List, Set, Dict, Any
import httpx


class ThreatIntelFetcher:
    """Fetches and aggregates IOC feeds into local threat_intel/ cache directory."""

    def __init__(self, data_dir: str = "data/threat_intel"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.malicious_urls: Set[str] = set()
        self.malicious_ips: Set[str] = set()

    async def fetch_openphish(self) -> int:
        """Download latest phishing feed from OpenPhish."""
        url = "https://openphish.com/feed.txt"
        count = 0
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    lines = resp.text.splitlines()
                    for line in lines:
                        item = line.strip()
                        if item and not item.startswith("#"):
                            self.malicious_urls.add(item)
                            count += 1
                    with open(self.data_dir / "openphish.txt", "w", encoding="utf-8") as f:
                        f.write("\n".join(self.malicious_urls))
        except Exception:
            pass
        return count

    def is_known_malicious_url(self, url: str) -> bool:
        """Check if URL matches any cached threat feeds."""
        clean = url.strip().lower()
        return clean in self.malicious_urls or any(m in clean for m in self.malicious_urls)
