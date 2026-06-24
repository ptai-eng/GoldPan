import os
import csv
import json
from goldpan.core.schemas import RawContent, DocumentInfo

class LiteParseExtractor:
    def __init__(self):
        pass

    def extract(self, source: str) -> RawContent:
        if not os.path.exists(source):
            raise FileNotFoundError(f"Source file not found: {source}")
            
        ext = os.path.splitext(source)[1].lower()
        title = os.path.basename(source)
        text_content = ""

        try:
            if ext == '.json':
                with open(source, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    text_content = f"# {title}\n\n```json\n{json.dumps(data, indent=2, ensure_ascii=False)}\n```"
            elif ext == '.csv':
                with open(source, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if rows:
                        # Convert to Markdown Table
                        header = rows[0]
                        table = f"| {' | '.join(header)} |\n"
                        table += f"|{'|'.join(['---'] * len(header))}|\n"
                        for row in rows[1:]:
                            # Pad row if necessary to match header length
                            row = row + [''] * (len(header) - len(row))
                            table += f"| {' | '.join(row)} |\n"
                        text_content = f"# {title}\n\n{table}"
                    else:
                        text_content = f"# {title}\n\n(Empty CSV)"
            else:
                # Default to plain text
                with open(source, 'r', encoding='utf-8') as f:
                    content = f.read()
                    text_content = f"# {title}\n\n{content}"
        except Exception as e:
            text_content = f"# {title}\n\nError reading file: {str(e)}"

        doc_info = DocumentInfo(
            file_path=source,
            title=title
        )
        
        return RawContent(
            text=text_content,
            doc_info=doc_info,
            original_format="liteparse_" + ext.replace('.', '')
        )
