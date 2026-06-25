import os
import mimetypes
from markitdown import MarkItDown, DocumentConverter, DocumentConverterResult
from goldpan.core.schemas import RawContent, DocumentInfo

class GeminiMultimodalConverter(DocumentConverter):
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        if self.api_key:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def accepts(self, file_stream, stream_info, **kwargs) -> bool:
        if stream_info and stream_info.extension:
            ext = stream_info.extension.lower()
            if not ext.startswith('.'):
                ext = '.' + ext
            return ext in ['.jpg', '.jpeg', '.png', '.webp', '.mp3', '.wav', '.ogg', '.m4a']
        return False

    def convert(self, file_stream, stream_info, **kwargs) -> DocumentConverterResult:
        local_path = getattr(file_stream, "name", None)
        if not local_path or not isinstance(local_path, str) or not os.path.exists(local_path):
            raise ValueError("Gemini API needs a local file path.")
        source_str = str(local_path)
        
        if not self.client:
            raise ValueError("Missing API Key for Image/Audio processing. Please configure your Gemini API Key in Settings.")
            
        prompt = "Extract all text, transcriptions, and relevant metadata (like EXIF data) from this file. Format the output cleanly in Markdown."
        
        try:
            # Upload file directly to Gemini API
            uploaded_file = self.client.files.upload(file=source_str)
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[prompt, uploaded_file]
            )
            return DocumentConverterResult(
                title=os.path.basename(source_str),
                text_content=response.text
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise ValueError(f"Gemini multimodal processing failed: {str(e)}")

class MarkItDownExtractor:
    def __init__(self, api_key: str = None):
        self.md_client = MarkItDown()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        # Đăng ký plugin đọc Ảnh/Audio bằng Gemini (ưu tiên cao hơn mặc định)
        self.md_client.register_converter(GeminiMultimodalConverter(self.api_key), priority=-1.0)

    def extract(self, source: str) -> RawContent:
        if not os.path.exists(source):
            raise FileNotFoundError(f"Source file not found: {source}")
            
        result = self.md_client.convert(source)
        
        doc_info = DocumentInfo(
            file_path=source,
            title=os.path.basename(source)
        )
        
        return RawContent(
            text=result.text_content,
            doc_info=doc_info,
            original_format="markitdown"
        )
