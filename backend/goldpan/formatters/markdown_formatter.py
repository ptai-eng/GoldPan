import re
import unicodedata

from goldpan.core.schemas import FilteredContent, FinalMarkdown, FirewallReport
from goldpan.filters.vn_encoding import normalize_vietnamese_text
from goldpan.firewall.context_firewall import AgentContextFirewall


class AIFriendlyMarkdownFormatter:
    """
    Normalizes Vietnamese legacy encodings (TCVN3, VNI-Windows, NFD) to composed Unicode NFC,
    runs the Agent Context Firewall text layer (Unicode Tags, Bidi, fake delimiters, exfil beacons),
    normalizes ATX headings (outside fenced code blocks), collapses excessive blank lines,
    injects missing document title, and forwards pipeline + firewall telemetry.
    """

    def __init__(self, firewall_mode: str = "sanitize") -> None:
        self.firewall_mode = firewall_mode
        self.firewall = AgentContextFirewall(mode=firewall_mode)

    def format(self, content: FilteredContent) -> FinalMarkdown:
        # 0. Normalize Vietnamese encodings (VNI-Windows, TCVN3, NFD -> NFC)
        text, detected_encoding = normalize_vietnamese_text(content.text)

        # 0.5. Run Agent Context Firewall text inspection & sanitization
        prior_fw_dict = (content.doc_info.metadata or {}).get("firewall")
        prior_report = FirewallReport(**prior_fw_dict) if isinstance(prior_fw_dict, dict) else None
        text, fw_report = self.firewall.process_text(text, prior_report=prior_report)

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
        if content.doc_info.title:
            norm_title = unicodedata.normalize("NFC", content.doc_info.title)
            if norm_title.lower() not in text[:500].lower():
                text = f"# {norm_title}\n\n{text}"

        stats = dict(content.stats)
        stats["detected_encoding"] = detected_encoding
        stats["firewall"] = fw_report.model_dump()
        content.doc_info.metadata["firewall"] = fw_report.model_dump()
        content.doc_info.metadata["detected_encoding"] = detected_encoding

        return FinalMarkdown(
            text=text.strip(),
            doc_info=content.doc_info,
            stats=stats,
            firewall=fw_report,
        )
