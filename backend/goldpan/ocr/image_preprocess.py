"""Image preprocessing pipeline for offline Vietnamese & document OCR."""

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def load_image(image_input: Union[str, Path, Image.Image]) -> Image.Image:
    """Load image from file path or return existing PIL Image instance in RGB (closing file handles immediately)."""
    if isinstance(image_input, (str, Path)):
        with Image.open(str(image_input)) as raw_img:
            try:
                transposed = ImageOps.exif_transpose(raw_img)
            except Exception:
                transposed = raw_img
            img = transposed.convert("RGB") if transposed.mode not in ("RGB", "L") else transposed.copy()
        return img
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise TypeError(f"Expected str, Path, or Image.Image, got {type(image_input)}")

    # Handle EXIF orientation tags automatically
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img


def enhance_contrast(img: Image.Image, factor: float = 1.8) -> Image.Image:
    """Enhance document image contrast for crisp text glyph detection."""
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def sharpen_text(img: Image.Image) -> Image.Image:
    """Apply unsharp mask filter to sharpen glyph boundaries without adding ringing."""
    return img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=3))


def binarize_adaptive(gray_img: Image.Image, window_size: int = 25, k: float = 0.2) -> Image.Image:
    """
    Vectorized 2D Integral-Image Sauvola adaptive threshold binarization:
        T(x, y) = mean(x, y) * [1 + k * ((std(x, y) / R) - 1)], R = 128.0
    Handles uneven illumination, page curvature shadows, and low-contrast scans in O(1) per pixel.
    """
    if gray_img.mode != "L":
        gray_img = gray_img.convert("L")

    arr = np.array(gray_img, dtype=np.float64)
    h, w = arr.shape
    win = max(3, int(window_size) | 1)  # Ensure odd window size >= 3
    pad = win // 2

    padded = np.pad(arr, pad_width=pad, mode="reflect")
    # 2D integral images (padded with a leading 0 row/col for 4-corner box lookups)
    ii = np.pad(np.cumsum(np.cumsum(padded, axis=0), axis=1), ((1, 0), (1, 0)), mode="constant")
    ii_sq = np.pad(np.cumsum(np.cumsum(padded * padded, axis=0), axis=1), ((1, 0), (1, 0)), mode="constant")

    area = float(win * win)
    local_sum = ii[win : win + h, win : win + w] - ii[0:h, win : win + w] - ii[win : win + h, 0:w] + ii[0:h, 0:w]
    local_sq_sum = (
        ii_sq[win : win + h, win : win + w]
        - ii_sq[0:h, win : win + w]
        - ii_sq[win : win + h, 0:w]
        + ii_sq[0:h, 0:w]
    )

    mean = local_sum / area
    variance = np.maximum(0.0, (local_sq_sum / area) - (mean * mean))
    std = np.sqrt(variance)

    R = 128.0
    thresh = mean * (1.0 + float(k) * ((std / R) - 1.0))
    binary = np.where(arr > thresh, 255, 0).astype(np.uint8)
    return Image.fromarray(binary, mode="L")


def preprocess_ocr_image(
    image_input: Union[str, Path, Image.Image],
    target_dpi: int = 300,
    binarize: bool = False,
) -> Image.Image:
    """
    Standard preprocessing pipeline for scanned PDFs and photo captures:
    1. EXIF orientation correction
    2. Upscaling if image resolution is too low (< 1000px on long edge)
    3. Grayscale conversion
    4. Contrast boost and subtle text sharpening
    5. Optional adaptive binarization
    """
    img = load_image(image_input)

    # Upscale low-res images for improved character recognition
    w, h = img.size
    max_dim = max(w, h)
    if max_dim < 1200:
        scale = 1200.0 / float(max_dim)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Convert to grayscale
    gray = ImageOps.grayscale(img)

    # Enhance contrast
    contrast_img = enhance_contrast(gray, factor=1.6)

    # Sharpen glyphs
    sharpened = sharpen_text(contrast_img)

    if binarize:
        return binarize_adaptive(sharpened)
    return sharpened


def detect_text_bands(gray_img: Image.Image, min_line_height: int = 12) -> list[tuple[int, int, int, int]]:
    """
    Locates horizontal text line bounding boxes [x0, y0, x1, y1] via horizontal projection profile.
    Useful for layout reconstruction and table cell slicing.
    """
    if gray_img.mode != "L":
        gray_img = gray_img.convert("L")

    arr = np.array(gray_img, dtype=np.uint8)
    w = arr.shape[1]

    # Invert so text pixels are high (dark text on light background)
    inverted = 255 - arr
    row_sums = np.sum(inverted > 80, axis=1)

    threshold = max(5, int(w * 0.015))
    in_line = False
    start_y = 0
    bands: list[tuple[int, int, int, int]] = []

    for y, count in enumerate(row_sums):
        if count > threshold and not in_line:
            in_line = True
            start_y = y
        elif count <= threshold and in_line:
            in_line = False
            height = y - start_y
            if height >= min_line_height:
                bands.append((0, start_y, w, y))

    if in_line and (len(row_sums) - start_y) >= min_line_height:
        bands.append((0, start_y, w, len(row_sums)))

    return bands
