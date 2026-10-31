import os

from bs4 import BeautifulSoup
from markitdown import DocumentConverter, DocumentConverterResult, MarkItDown

from goldpan.core.schemas import DocumentInfo, RawContent
from goldpan.firewall.context_firewall import AgentContextFirewall
from goldpan.ocr.engine import VietnameseOfflineOCR
from goldpan.ocr.pdf_scanner import is_scanned_pdf
from goldpan.vision.hybrid_captioner import HybridVisionCaptioner


class GeminiMultimodalConverter(DocumentConverter):
    def __init__(
        self,
        api_key: str | None = None,
        firewall_mode: str = "sanitize",
    ) -> None:
        self.api_key = api_key
        if self.api_key:
            from google import genai

            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None
        self.local_captioner = HybridVisionCaptioner(api_key=self.api_key)
        self.ocr_engine = VietnameseOfflineOCR(firewall_mode=firewall_mode)

    def accepts(self, file_stream, stream_info, **kwargs) -> bool:
        if stream_info and stream_info.extension:
            ext = stream_info.extension.lower()
            if not ext.startswith("."):
                ext = "." + ext
            return ext in [".jpg", ".jpeg", ".png", ".webp", ".mp3", ".wav", ".ogg", ".m4a"]
        return False

    def convert(self, file_stream, stream_info, **kwargs) -> DocumentConverterResult:
        local_path = getattr(file_stream, "name", None)
        if not local_path or not isinstance(local_path, str) or not os.path.exists(local_path):
            raise ValueError("Multimodal converter requires a valid local file path.")
        source_str = str(local_path)
        ext = os.path.splitext(source_str)[1].lower()

        if not self.client:
            if ext in [".jpg", ".jpeg", ".png", ".webp"]:
                # Offline OCR text recognition
                ocr_page = self.ocr_engine.extract_from_image(source_str)
                exif_summary = self.local_captioner.process_image(source_str)

                if ocr_page.text and not ocr_page.text.startswith("[Trang scan không chứa"):
                    content = (
                        f"# {os.path.basename(source_str)}\n\n"
                        f"{ocr_page.text}\n\n"
                        f"---\n*{exif_summary} | OCR: {ocr_page.backend} (conf={ocr_page.confidence:.2f})*\n"
                    )
                else:
                    content = f"# {os.path.basename(source_str)}\n\n{exif_summary}\n"

                return DocumentConverterResult(
                    title=os.path.basename(source_str),
                    text_content=content,
                )
            raise ValueError(
                "Missing API Key for Audio transcription. Please configure your Gemini API Key in Settings."
            )

        prompt = (
            "Extract all text, transcriptions, tables, and relevant metadata (like EXIF data) "
            "from this file. Format the output cleanly in Markdown."
        )

        try:
            uploaded_file = self.client.files.upload(file=source_str)
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, uploaded_file],
            )
            return DocumentConverterResult(
                title=os.path.basename(source_str),
                text_content=response.text,
            )
        except Exception as e:
            if ext in [".jpg", ".jpeg", ".png", ".webp"]:
                ocr_page = self.ocr_engine.extract_from_image(source_str)
                exif_summary = self.local_captioner.process_image(source_str)
                content = f"# {os.path.basename(source_str)}\n\n{ocr_page.text or exif_summary}\n"
                return DocumentConverterResult(
                    title=os.path.basename(source_str),
                    text_content=content,
                )
            raise ValueError(f"Gemini multimodal processing failed: {str(e)}")


class MarkItDownExtractor:
    def __init__(
        self,
        api_key: str | None = None,
        firewall_mode: str = "sanitize",
        enable_ocr: bool = True,
    ) -> None:
        self.md_client = MarkItDown()
        self.api_key = api_key
        self.firewall_mode = firewall_mode
        self.enable_ocr = enable_ocr
        self.firewall = AgentContextFirewall(mode=firewall_mode)
        self.ocr_engine = VietnameseOfflineOCR(firewall_mode=firewall_mode)
        self.md_client.register_converter(
            GeminiMultimodalConverter(self.api_key, firewall_mode=firewall_mode),
            priority=-1.0,
        )

    def extract(self, source: str) -> RawContent:
        if not os.path.exists(source):
            raise FileNotFoundError(f"Source file not found: {source}")

        ext = os.path.splitext(source)[1].lower()
        fw_meta: dict = {}

        if ext == ".pdf":
            stealth_clean_text, fw_report = self.firewall.inspect_pdf_stealth(source)
            fw_meta = {"firewall": fw_report.model_dump()}

            # Check if this PDF is a scanned document needing Offline OCR
            if self.enable_ocr:
                is_scanned, scanned_indices, total_pages = is_scanned_pdf(source)
                if is_scanned:
                    ocr_res = self.ocr_engine.extract_from_pdf(source)
                    fw_meta["ocr"] = {
                        "backend": ocr_res.backend,
                        "avg_confidence": ocr_res.avg_confidence,
                        "is_scanned": True,
                        "pages_processed": len(ocr_res.pages),
                    }
                    if ocr_res.text:
                        doc_info = DocumentInfo(
                            file_path=source,
                            title=os.path.basename(source),
                            metadata=fw_meta,
                        )
                        return RawContent(
                            text=ocr_res.text,
                            doc_info=doc_info,
                            original_format="pdf_scanned_ocr",
                        )

            if stealth_clean_text is not None and self.firewall_mode == "sanitize":
                doc_info = DocumentInfo(
                    file_path=source,
                    title=os.path.basename(source),
                    metadata=fw_meta,
                )
                return RawContent(
                    text=stealth_clean_text,
                    doc_info=doc_info,
                    original_format="pdf",
                )
        elif ext == ".docx":
            vanish_clean_text, fw_report = self.firewall.inspect_docx_vanish(source)
            fw_meta = {"firewall": fw_report.model_dump()}
            if vanish_clean_text is not None and self.firewall_mode == "sanitize":
                doc_info = DocumentInfo(
                    file_path=source,
                    title=os.path.basename(source),
                    metadata=fw_meta,
                )
                return RawContent(
                    text=vanish_clean_text,
                    doc_info=doc_info,
                    original_format="docx",
                )
        elif ext in (".html", ".htm"):
            with open(source, "r", encoding="utf-8", errors="ignore") as f:
                raw_html = f.read()
            cleaned_html, fw_report = self.firewall.process_html(raw_html)
            fw_meta = {"firewall": fw_report.model_dump()}
            soup = BeautifulSoup(cleaned_html, "html.parser")
            doc_info = DocumentInfo(
                file_path=source,
                title=os.path.basename(source),
                metadata=fw_meta,
            )
            return RawContent(
                text=soup.get_text(separator="\n", strip=True),
                doc_info=doc_info,
                original_format="html",
            )

        result = self.md_client.convert(source)

        doc_info = DocumentInfo(
            file_path=source,
            title=os.path.basename(source),
            metadata=fw_meta,
        )

        return RawContent(
            text=result.text_content,
            doc_info=doc_info,
            original_format="markitdown",
        )
