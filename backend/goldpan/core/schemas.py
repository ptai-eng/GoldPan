from typing import Any

from pydantic import BaseModel, Field


class FirewallReport(BaseModel):
    """Security telemetry emitted by AgentContextFirewall."""

    mode: str = "sanitize"
    risk_score: float = 0.0
    is_suspicious: bool = False
    sanitized_count: int = 0
    signals: list[str] = Field(default_factory=list)
    untrusted_content: bool = True


class Finding(BaseModel):
    """An individual security finding detected during repo scanning."""

    path: str
    line: int = 1
    column: int = 1
    signal: str
    severity: str = "suspicious"  # "suspicious" | "info"
    risk_score: float = 0.0
    snippet: str = ""


class DocumentInfo(BaseModel):
    source_url: str | None = None
    file_path: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RawContent(BaseModel):
    """Output from Extractors"""

    text: str
    doc_info: DocumentInfo
    original_format: str


class FilteredContent(BaseModel):
    """Output from Filters"""

    text: str
    doc_info: DocumentInfo
    stats: dict[str, Any] = Field(default_factory=dict)


class FinalMarkdown(BaseModel):
    """Final Output"""

    text: str
    doc_info: DocumentInfo
    stats: dict[str, Any] = Field(default_factory=dict)
    firewall: FirewallReport | None = None
