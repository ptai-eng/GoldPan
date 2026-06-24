import os
from goldpan.core.schemas import RawContent, DocumentInfo

class LiteParseExtractor:
    def __init__(self):
        # In a real setup, this would load the liteparse binary/WASM or Python package
        # e.g., import liteparse
        pass

    def extract(self, source: str) -> RawContent:
        if not os.path.exists(source):
            raise FileNotFoundError(f"Source file not found: {source}")
            
        # MOCK CALL to LiteParse
        # result = liteparse.parse(source, use_ocr=False)
        # text_content = result.get_markdown_with_bounding_boxes()
        
        # Simulated Output
        text_content = f"# [LiteParse] {os.path.basename(source)}\n\n(Extracted spatially in ~3ms. Bounding boxes preserved.)\n\nThis is a mock representation of the PDF content."
        
        doc_info = DocumentInfo(
            file_path=source,
            title=os.path.basename(source)
        )
        
        return RawContent(
            text=text_content,
            doc_info=doc_info,
            original_format="liteparse_pdf"
        )
