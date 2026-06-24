import os
from markitdown import MarkItDown
from goldpan.core.schemas import RawContent, DocumentInfo

class MarkItDownExtractor:
    def __init__(self):
        self.md_client = MarkItDown()

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
