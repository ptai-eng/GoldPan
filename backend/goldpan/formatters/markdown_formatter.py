import re
from goldpan.core.schemas import FilteredContent, FinalMarkdown


class AIFriendlyMarkdownFormatter:
    """
    Normalizes ATX headings (outside fenced code blocks), collapses excessive
    blank lines, injects missing document title, and forwards pipeline telemetry.
    """

    def format(self, content: FilteredContent) -> FinalMarkdown:
        text = content.text
        lines = text.split("\n")
        formatted_lines = []
        in_code_fence = False

        # 1. Ensure valid ATX headings (1-6 '#') have a space after '#', but NEVER touch lines inside code fences
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_code_fence = not in_code_fence
                formatted_lines.append(line)
                continue

            if not in_code_fence:
                line = re.sub(r"^(#{1,6})([^\s#])", r"\1 \2", line)
            formatted_lines.append(line)

        text = "\n".join(formatted_lines)

        # 2. Consolidate 3+ consecutive newlines into 2 newlines (1 blank line)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 3. Add title if missing from text but present in doc_info
        if content.doc_info.title and content.doc_info.title.lower() not in text[:500].lower():
            text = f"# {content.doc_info.title}\n\n{text}"

        return FinalMarkdown(
            text=text.strip(),
            doc_info=content.doc_info,
            stats=content.stats,
        )
