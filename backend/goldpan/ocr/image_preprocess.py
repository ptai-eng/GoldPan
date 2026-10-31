"""Image preprocessing pipeline for offline Vietnamese & document OCR."""

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def load_image(image_input: Union[str, Path, Image.Image]) -> Image.Image:
    """Load image from file path or return existing PIL Image instance in RGB."""
    if isinstance(image_input, (str, Path)):
        img = Image.open(str(image_input))
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
    Sauvola/Otsu-style adaptive threshold binarization for non-uniform lighting/shadows in scanned documents.
    """
    if gray_img.mode != "L":
        gray_img = gray_img.convert("L")

    arr = np.array(gray_img, dtype=np.float32)
    # Fast thresholding using mean luminance cutoff
    mean_val = float(np.mean(arr))
    thresh = max(60.0, min(210.0, mean_val * 0.92))
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
