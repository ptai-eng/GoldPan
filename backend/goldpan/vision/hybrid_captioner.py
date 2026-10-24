import os
from typing import Optional
from PIL import Image, ExifTags


class HybridVisionCaptioner:
    """Dual-mode image inspector: Local EXIF/geometry extraction with optional Gemini Vision captioning."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.local_mode = not bool(self.api_key)

    def suggest_api_if_needed(self, image_count: int, table_count: int) -> str:
        if image_count > 10 or table_count > 5:
            return "WARNING: Large number of complex images/tables detected. Suggest switching to API Vision Model for higher accuracy."
        return "Local Vision Model is sufficient."

    def process_image(self, image_path: str) -> str:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if not self.local_mode and self.api_key:
            try:
                from google import genai
                client = genai.Client(api_key=self.api_key)
                uploaded = client.files.upload(file=image_path)
                res = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        "Describe this image concisely, extract any visible text or tables in Markdown, and note key visual elements.",
                        uploaded,
                    ],
                )
                if res and res.text:
                    return res.text.strip()
            except Exception:
                pass

        # Local offline metadata & EXIF inspection via Pillow
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                fmt = img.format or "UNKNOWN"
                mode = img.mode
                exif_notes = []
                exif_data = img.getexif()
                if exif_data:
                    for tag_id, val in exif_data.items():
                        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                        if tag_name in ("Make", "Model", "DateTime", "Software", "ImageDescription"):
                            exif_notes.append(f"{tag_name}={val}")
                exif_str = f" | EXIF: {', '.join(exif_notes)}" if exif_notes else ""
                return f"[Image: {os.path.basename(image_path)} ({fmt}, {width}x{height}px, {mode}{exif_str})]"
        except Exception as e:
            return f"[Image: {os.path.basename(image_path)} (Local inspection fallback: {e})]"
