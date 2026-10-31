"""GoldPan AI v3.1 Offline Vietnamese & Multilingual OCR Module."""

from .engine import OCRResult, PageOCRResult, VietnameseOfflineOCR
from .image_preprocess import preprocess_ocr_image
from .pdf_scanner import is_scanned_pdf, render_pdf_page_to_image

__all__ = [
    "VietnameseOfflineOCR",
    "OCRResult",
    "PageOCRResult",
    "preprocess_ocr_image",
    "is_scanned_pdf",
    "render_pdf_page_to_image",
]
