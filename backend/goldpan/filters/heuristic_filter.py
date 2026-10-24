import re
from goldpan.core.schemas import RawContent, FilteredContent


class HeuristicFilter:
    """
    Deterministic line-level boilerplate & navigation noise filter.
    Preserves fenced code blocks and long prose paragraphs while stripping
    short cookie banners, newsletter prompts, copyright footers, and standalone nav links.
    """

    def __init__(self):
        # Boilerplate phrases (applied only to short non-code lines <= 80 chars)
        self.short_line_noise_patterns = [
            re.compile(r"(?i)\bcookie policy\b"),
            re.compile(r"(?i)\bsubscribe to (?:our )?newsletter\b"),
            re.compile(r"(?i)\ball rights reserved\b"),
            re.compile(r"(?i)^\s*(?:accept all cookies|manage cookies|privacy preferences)\s*$"),
        ]
        # Non-greedy standalone navigation link on its own line: [Label](url)
        self.standalone_link_pattern = re.compile(
            r"^\s*\[[^\[\]]{1,80}\]\([^()\s]{1,300}\)\s*$"
        )

    def apply(self, content: RawContent | FilteredContent) -> FilteredContent:
        text = content.text
        lines = text.split("\n")

        filtered_lines = []
        noise_removed_bytes = 0
        in_code_fence = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_code_fence = not in_code_fence
                filtered_lines.append(line)
                continue

            if not in_code_fence:
                if self.standalone_link_pattern.match(line):
                    noise_removed_bytes += len(line.encode("utf-8"))
                    continue
                if len(stripped) <= 80 and any(p.search(stripped) for p in self.short_line_noise_patterns):
                    noise_removed_bytes += len(line.encode("utf-8"))
                    continue

            filtered_lines.append(line)

        final_text = "\n".join(filtered_lines)

        stats = content.stats.copy() if isinstance(content, FilteredContent) else {}
        stats["heuristic_noise_removed_bytes"] = noise_removed_bytes

        return FilteredContent(
            text=final_text,
            doc_info=content.doc_info,
            stats=stats,
        )
