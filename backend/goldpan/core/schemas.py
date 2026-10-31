from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FirewallReport(BaseModel):
    """Security telemetry emitted by AgentContextFirewall."""

    mode: str = "sanitize"
    risk_score: float = 0.0
    is_suspicious: bool = False
    sanitized_count: int = 0
    signals: List[str] = Field(default_factory=list)
    untrusted_content: bool = True


class DocumentInfo(BaseModel):
    source_url: Optional[str] = None
    file_path: Optional[str] = None
    title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RawContent(BaseModel):
    """Output from Extractors"""

    text: str
    doc_info: DocumentInfo
    original_format: str


class FilteredContent(BaseModel):
    """Output from Filters"""

    text: str
    doc_info: DocumentInfo
    stats: Dict[str, Any] = Field(default_factory=dict)


class FinalMarkdown(BaseModel):
    """Final Output"""

    text: str
    doc_info: DocumentInfo
    stats: Dict[str, Any] = Field(default_factory=dict)
    firewall: Optional[FirewallReport] = None
