import re
from goldpan.core.schemas import FilteredContent, FinalMarkdown

class AIFriendlyMarkdownFormatter:
    def format(self, content: FilteredContent) -> FinalMarkdown:
        text = content.text
        
        # 1. Ensure headings have space after them (e.g., #Heading -> # Heading)
        text = re.sub(r'^(#+)(?! )', r'\1 ', text, flags=re.MULTILINE)
        
        # 2. Consolidate multiple blank lines into max 2 blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 3. Add title if missing from text but present in doc_info
        if content.doc_info.title and content.doc_info.title.lower() not in text[:500].lower():
            text = f"# {content.doc_info.title}\n\n{text}"
            
        return FinalMarkdown(
            text=text.strip(),
            doc_info=content.doc_info
        )
