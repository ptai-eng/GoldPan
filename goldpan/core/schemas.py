from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

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
    stats: Dict[str, Any] = Field(default_factory=dict) # E.g., noise_removed_bytes

class FinalMarkdown(BaseModel):
    """Final Output"""
    text: str
    doc_info: DocumentInfo
