import os
import io
import csv
import json
from goldpan.core.schemas import RawContent, DocumentInfo


def _read_text_with_fallback(path: str) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to decode text file: {path}")


def _escape_md_cell(cell: str) -> str:
    cleaned = str(cell).replace("\r\n", "<br>").replace("\n", "<br>").replace("|", "\\|")
    return cleaned.strip()


class LiteParseExtractor:
    def __init__(self):
        pass

    def extract(self, source: str) -> RawContent:
        if not os.path.exists(source):
            raise FileNotFoundError(f"Source file not found: {source}")

        ext = os.path.splitext(source)[1].lower()
        title = os.path.basename(source)

        if ext == ".json":
            raw_str = _read_text_with_fallback(source)
            data = json.loads(raw_str)
            text_content = f"# {title}\n\n```json\n{json.dumps(data, indent=2, ensure_ascii=False)}\n```"
        elif ext == ".csv":
            raw_str = _read_text_with_fallback(source)
            reader = csv.reader(io.StringIO(raw_str))
            rows = [r for r in reader if any(cell.strip() for cell in r)]
            if rows:
                header = [_escape_md_cell(c) or f"Col_{i+1}" for i, c in enumerate(rows[0])]
                col_count = len(header)
                table_lines = [
                    f"| {' | '.join(header)} |",
                    f"| {' | '.join(['---'] * col_count)} |",
                ]
                for row in rows[1:]:
                    cells = [_escape_md_cell(c) for c in row[:col_count]]
                    if len(cells) < col_count:
                        cells.extend([""] * (col_count - len(cells)))
                    table_lines.append(f"| {' | '.join(cells)} |")
                text_content = f"# {title}\n\n" + "\n".join(table_lines)
            else:
                text_content = f"# {title}\n\n(Empty CSV)"
        elif ext == ".xml":
            content = _read_text_with_fallback(source)
            text_content = f"# {title}\n\n```xml\n{content.strip()}\n```"
        else:
            content = _read_text_with_fallback(source)
            text_content = f"# {title}\n\n{content}"

        doc_info = DocumentInfo(
            file_path=source,
            title=title,
        )

        return RawContent(
            text=text_content,
            doc_info=doc_info,
            original_format="liteparse_" + ext.replace(".", ""),
        )
