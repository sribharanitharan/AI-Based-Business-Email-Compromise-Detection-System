"""
schemas.py — Pydantic request and response schemas for BEC Detection API.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class EmailAnalysisRequest(BaseModel):
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email plain text or HTML body")
    from_addr: Optional[str] = Field("", description="Sender address")
    sender: Optional[str] = Field("", description="Sender display/address")
    to_addr: Optional[str] = Field("", description="Recipient address")
    reply_to: Optional[str] = Field("", description="Reply-To header")
    urls: Optional[List[str]] = Field(default_factory=list, description="Extracted URLs")
    attachments: Optional[List[str]] = Field(default_factory=list, description="Attachment filenames")
    headers: Optional[Dict[str, Any]] = Field(default_factory=dict, description="SPF/DKIM/DMARC headers")
    first_time_sender: Optional[bool] = False
    hour_sent: Optional[int] = 12

class ThreatIntelFeedStatus(BaseModel):
    name: str
    status: str
    last_updated: str
    entries_count: int
