"""GoldPan AI v3.1 Offline Vietnamese & Multilingual OCR Module."""

from .engine import OCRResult, PageOCRResult, VietnameseOfflineOCR
from .image_preprocess import preprocess_ocr_image
from .pdf_scanner import is_scanned_pdf, render_pdf_page_to_image

__all__ = [
    "OCRResult",
    "PageOCRResult",
    "VietnameseOfflineOCR",
    "is_scanned_pdf",
    "preprocess_ocr_image",
    "render_pdf_page_to_image",
]
