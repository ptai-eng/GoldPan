"""Scanned PDF detector and high-speed page rasterizer using pypdfium2 / pdfplumber."""

from pathlib import Path
from typing import Union

from PIL import Image

try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None  # type: ignore[assignment]

try:
    import pdfplumber
except ImportError:
    pdfplumber = None  # type: ignore[assignment]


def is_scanned_pdf(
    file_path: Union[str, Path],
    char_threshold_per_page: int = 35,
    max_pages_to_sample: int = 5,
) -> tuple[bool, list[int], int]:
    """
    Determines if a PDF is primarily a scanned image document.

    A page is considered 'scanned' if:
    - Text extracted is below `char_threshold_per_page` (e.g. < 35 chars).
    - Or page contains full-page raster images and minimal textual stream.

    Returns:
        (is_scanned, scanned_page_indices, total_pages)
    """
    path_str = str(file_path)
    if not Path(path_str).exists():
        raise FileNotFoundError(f"PDF file not found: {path_str}")

    scanned_pages: list[int] = []
    total_pages = 0

    if pdfplumber is not None:
        try:
            with pdfplumber.open(path_str) as pdf:
                total_pages = len(pdf.pages)
                sample_count = min(total_pages, max_pages_to_sample) if max_pages_to_sample > 0 else total_pages
                for idx in range(sample_count):
                    page = pdf.pages[idx]
                    txt = page.extract_text() or ""
                    clean_txt = txt.strip()
                    chars_count = len(clean_txt)
                    images = page.images or []
                    # Check if text is sparse or missing while page has images
                    if chars_count < char_threshold_per_page:
                        scanned_pages.append(idx)
                    elif len(images) >= 1 and chars_count < 100:
                        # Full page scan with trivial stamp or header
                        scanned_pages.append(idx)

                # If sampled pages have >= 50% scanned pages, mark whole document as scanned
                is_scanned = len(scanned_pages) >= max(1, sample_count // 2)
                return is_scanned, scanned_pages, total_pages
        except Exception:
            pass

    # Fallback using pypdfium2 page text if pdfplumber failed
    if pdfium is not None:
        try:
            with pdfium.PdfDocument(path_str) as pdf:
                total_pages = len(pdf)
                sample_count = min(total_pages, max_pages_to_sample) if max_pages_to_sample > 0 else total_pages
                for idx in range(sample_count):
                    page = pdf[idx]
                    textpage = page.get_textpage()
                    txt = textpage.get_text_range() or ""
                    try:
                        textpage.close()
                    except Exception:
                        pass
                    try:
                        page.close()
                    except Exception:
                        pass
                    if len(txt.strip()) < char_threshold_per_page:
                        scanned_pages.append(idx)
                is_scanned = len(scanned_pages) >= max(1, sample_count // 2)
                return is_scanned, scanned_pages, total_pages
        except Exception:
            pass

    return False, [], total_pages


def render_pdf_page_to_image(
    file_path: Union[str, Path],
    page_index: int = 0,
    dpi: int = 200,
) -> Image.Image:
    """
    Render a specific PDF page into a high-DPI PIL Image without external Ghostscript/Poppler.
    Uses native C++ `pypdfium2` rendering with fallback to `pdfplumber.to_image()`.
    """
    path_str = str(file_path)
    if not Path(path_str).exists():
        raise FileNotFoundError(f"PDF file not found: {path_str}")

    scale = float(dpi) / 72.0

    if pdfium is not None:
        try:
            with pdfium.PdfDocument(path_str) as pdf:
                if page_index < 0 or page_index >= len(pdf):
                    raise IndexError(f"Page index {page_index} out of range for PDF with {len(pdf)} pages.")
                page = pdf[page_index]
                pil_image = page.render(scale=scale).to_pil()
                try:
                    page.close()
                except Exception:
                    pass
                return pil_image
        except Exception:
            pass

    if pdfplumber is not None:
        try:
            with pdfplumber.open(path_str) as pdf:
                if page_index < 0 or page_index >= len(pdf.pages):
                    raise IndexError(f"Page index {page_index} out of range for PDF with {len(pdf.pages)} pages.")
                page = pdf.pages[page_index]
                page_img = page.to_image(resolution=dpi)
                return page_img.original
        except Exception as exc:
            raise RuntimeError(f"Failed to render PDF page {page_index} using pdfplumber: {exc}")

    raise RuntimeError("Neither pypdfium2 nor pdfplumber is available to render PDF pages.")


def render_all_pdf_pages(
    file_path: Union[str, Path],
    max_pages: int = 20,
    dpi: int = 200,
) -> list[tuple[int, Image.Image]]:
    """Render all (or up to max_pages) pages of a PDF into PIL Images."""
    path_str = str(file_path)
    images: list[tuple[int, Image.Image]] = []

    if pdfium is not None:
        try:
            with pdfium.PdfDocument(path_str) as pdf:
                count = min(len(pdf), max_pages) if max_pages > 0 else len(pdf)
                scale = float(dpi) / 72.0
                for idx in range(count):
                    page = pdf[idx]
                    img = page.render(scale=scale).to_pil()
                    images.append((idx, img))
                    try:
                        page.close()
                    except Exception:
                        pass
                return images
        except Exception:
            pass

    if pdfplumber is not None:
        with pdfplumber.open(path_str) as pdf:
            count = min(len(pdf.pages), max_pages) if max_pages > 0 else len(pdf.pages)
            for idx in range(count):
                img = pdf.pages[idx].to_image(resolution=dpi).original
                images.append((idx, img))
            return images

    return images
