"""GoldPan AI — Privacy-First Multimodal Data Refinery & Dual-Mode Local RAG Engine."""

__version__ = "2.0.0"

from .core.pipeline import GoldPanPipeline
from .core.schemas import DocumentInfo, RawContent, FilteredContent, FinalMarkdown
from .kb.chroma_manager import ChromaManager

__all__ = [
    "__version__",
    "GoldPanPipeline",
    "DocumentInfo",
    "RawContent",
    "FilteredContent",
    "FinalMarkdown",
    "ChromaManager",
]
