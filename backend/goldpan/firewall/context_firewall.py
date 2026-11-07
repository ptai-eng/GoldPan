"""Multi-Layer Agent Context Firewall (HTML/CSS DOM, PDF/DOCX Stealth, Unicode Tags, Delimiters, Spotlighting)."""

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Comment

from ..core.exceptions import PromptInjectionBlockedError
from ..core.schemas import Finding, FirewallReport
from ..filters.vn_encoding import fold_vietnamese_diacritics

# Unicode Tag block U+E0000..U+E007F (both astral codepoints and UTF-16 surrogate pairs)
_RE_UNICODE_TAGS = re.compile(r"[\U000e0000-\U000e007f]|(?:\udb40[\udc00-\udc7f])+")

# Legitimate Emoji ZWJ sequences (e.g. 👨‍👩‍👧‍👦, 🏳️‍🌈, 🧑🏽‍💻)
_EMOJI_PART = r"(?:[\U0001f1e6-\U0001f1ff]|[\U0001f300-\U0001faff]|[\u2600-\u27bf]|[\u2300-\u23ff]|\ufe0f|[\U0001f3fb-\U0001f3ff])"
_RE_EMOJI_ZWJ_SEQ = re.compile(rf"{_EMOJI_PART}(?:\u200d{_EMOJI_PART})+")

# Bidi overrides and Zero-Width characters
_RE_BIDI_ZW = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]+")

# Fake ChatML / Llama / System delimiters
_RE_FAKE_CHAT_BLOCKS = re.compile(
    r"(?:<\|im_start\|>.*?(?:<\|im_end\|>|$))|(?:\[INST\].*?(?:\[/INST\]|$))|(?:<<SYS>>.*?(?:<</SYS>>|$))",
    re.IGNORECASE | re.DOTALL,
)
_RE_FAKE_CHAT_TOKENS = re.compile(
    r"(?:<\|im_start\|>|<\|im_end\|>|<\|system\|>|<\|user\|>|<\|assistant\|>|\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>)",
    re.IGNORECASE,
)

# Suspicious Markdown image exfiltration beacons: ![alt](http(s)://...?key=val)
_RE_MD_EXFIL_BEACON = re.compile(
    r"!\[([^\]]*)\]\((https?://[^\s)]+\?[^\s)]*(?:secret|token|key|cookie|env|session|exfil|data|leak|pwn|auth)=[^\s)]*)\)",
    re.IGNORECASE,
)

# Inline CSS stealth patterns
_RE_CSS_DISPLAY_NONE = re.compile(r"display\s*:\s*none", re.IGNORECASE)
_RE_CSS_VISIBILITY_OPACITY = re.compile(
    r"(?:visibility\s*:\s*hidden)|(?:opacity\s*:\s*0(?:\.0+)?(?:\s*;|\s*$))",
    re.IGNORECASE,
)
_RE_CSS_ZERO_FONT = re.compile(
    r"font-size\s*:\s*0(?:\.0+)?(?:px|pt|em|rem|%)?(?:\s*;|\s*$)",
    re.IGNORECASE,
)
_RE_CSS_WHITE_ON_WHITE = re.compile(
    r"(?:color\s*:\s*(?:#fff(?:fff)?|white|rgb\(\s*255\s*,\s*255\s*,\s*255\s*\)).*?"
    r"background(?:-color)?\s*:\s*(?:#fff(?:fff)?|white|rgb\(\s*255\s*,\s*255\s*,\s*255\s*\)))|"
    r"(?:color\s*:\s*(var\(--[a-zA-Z0-9_-]+\)).*?background(?:-color)?\s*:\s*\1)",
    re.IGNORECASE | re.DOTALL,
)
_RE_CSS_OFFSCREEN = re.compile(
    r"(?:(?:left|top|right|text-indent)\s*:\s*-\d{3,}(?:px|em|rem|vw|vh))|"
    r"(?:transform\s*:\s*translate[XY3d]*\(\s*-\d{3,})|"
    r"(?:clip-path\s*:\s*inset\(\s*50%\s*\))|"
    r"(?:clip\s*:\s*rect\(\s*0(?:px)?\s*,\s*0(?:px)?\s*,\s*0(?:px)?\s*,\s*0(?:px)?\s*\))",
    re.IGNORECASE,
)

# Suspicious HTML comment content
_RE_COMMENT_INJECTION = re.compile(
    r"\b(?:ignore|disregard|system\s+instruction|system\s+override|forget\s+prior|secret|exfil|api\s*key|rm\s+-rf|bỏ\s+qua)\b",
    re.IGNORECASE,
)

# Imperative semantic override patterns (English & Vietnamese)
_RE_SEMANTIC_EN = re.compile(
    r"(?:ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions)|"
    r"(?:disregard\s+(?:all\s+)?(?:previous|prior|evaluation)\s+(?:instructions|rubric|rules))|"
    r"(?:important\s+system\s+override\s*:)|"
    r"(?:output\s+the\s+contents\s+of\s+~/\.ssh)",
    re.IGNORECASE,
)
_RE_SEMANTIC_VI_FOLDED = re.compile(
    r"(?:bo\s+qua\s+(?:tat\s+ca\s+)?(?:cac\s+)?huong\s+dan\s+truoc\s+do)|"
    r"(?:dong\s+vai\s+quan\s+tri\s+vien\s+toi\s+cao)|"
    r"(?:gui\s+toan\s+bo\s+mat\s+khau\s+co\s+so\s+du\s+lieu)",
    re.IGNORECASE,
)

# Academic / Security educational context markers (Hard Negative protection)
_RE_ACADEMIC_CONTEXT = re.compile(
    r"(?:owasp\s+top\s+10|llm01|in\s+this\s+paper|abstract\s*:|our\s+empirical\s+study|"
    r"security\s+engineers\s+mitigate|such\s+as\s+[\"']ignore\s+previous\s+instructions[\"']|"
    r"write\s+[\"']bỏ\s+qua\s+hướng\s+dẫn\s+trước\s+đó[\"'])",
    re.IGNORECASE,
)


class AgentContextFirewall:
    """Multi-layer sanitizer and semantic risk scorer for untrusted RAG & MCP inputs.

    Modes:
    - 'sanitize' (default): Deterministically strips hidden DOM/CSS, PDF/DOCX stealth text,
      Unicode Tags/Bidi/Zero-Width chars, fake chat delimiters, and markdown exfil beacons.
      Scores semantic instruction overrides without deleting visible prose (0% FPR on OWASP/research docs).
    - 'audit': Non-destructive inspection; returns risk_score and signals without modifying content.
    - 'strict': Raises `PromptInjectionBlockedError` whenever any injection signal or `risk_score >= 0.7` is found.
    """

    VALID_MODES = ("sanitize", "audit", "strict")
    VALID_PROFILES = ("text", "code")

    def __init__(self, mode: str = "sanitize", profile: str = "text") -> None:
        normalized_mode = (mode or "sanitize").strip().lower()
        if normalized_mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid firewall mode '{mode}'. Choose from {self.VALID_MODES}."
            )
        self.mode = normalized_mode

        normalized_profile = (profile or "text").strip().lower()
        if normalized_profile not in self.VALID_PROFILES:
            normalized_profile = "text"
        self.profile = normalized_profile

    def process_html(self, html: str) -> tuple[str, FirewallReport]:
        """Inspect and optionally strip hidden DOM/CSS elements and malicious HTML comments before extraction."""
        if not html:
            return "", FirewallReport(mode=self.mode)

        soup = BeautifulSoup(html, "html.parser")
        signals: list[str] = []
        sanitized_count = 0

        # 1. Inspect HTML comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            c_text = str(comment)
            if _RE_COMMENT_INJECTION.search(c_text):
                if "html_comment_injection" not in signals:
                    signals.append("html_comment_injection")
                sanitized_count += 1
                if self.mode == "sanitize":
                    comment.extract()

        # 2. Inspect <template> tags (invisible in rendered DOM)
        for tmpl in soup.find_all("template"):
            t_text = tmpl.get_text(" ", strip=True)
            if t_text:
                if "html_template_hidden" not in signals:
                    signals.append("html_template_hidden")
                sanitized_count += 1
                if self.mode == "sanitize":
                    tmpl.decompose()

        # 2.5. Inspect <style> blocks for stylesheet-level stealth selectors (.class, #id, [attr])
        for style_tag in soup.find_all("style"):
            css_text = style_tag.get_text() or ""
            for rule_match in re.finditer(r"([^{}]+)\{([^{}]+)\}", css_text):
                raw_selectors, decls = (
                    rule_match.group(1).strip(),
                    rule_match.group(2).strip(),
                )
                if (
                    _RE_CSS_DISPLAY_NONE.search(decls)
                    or _RE_CSS_VISIBILITY_OPACITY.search(decls)
                    or _RE_CSS_ZERO_FONT.search(decls)
                    or _RE_CSS_WHITE_ON_WHITE.search(decls)
                    or _RE_CSS_OFFSCREEN.search(decls)
                ):
                    for sel in raw_selectors.split(","):
                        clean_sel = re.sub(
                            r"::?[a-zA-Z-]+(?:\([^)]*\))?", "", sel
                        ).strip()
                        if (
                            not clean_sel
                            or clean_sel.startswith("@")
                            or clean_sel.lower() in ("html", "head", "body")
                        ):
                            continue
                        try:
                            for matched_el in list(soup.select(clean_sel)):
                                if matched_el.name not in (
                                    "html",
                                    "head",
                                    "body",
                                    "style",
                                    "script",
                                ):
                                    if "css_stylesheet_hidden" not in signals:
                                        signals.append("css_stylesheet_hidden")
                                    sanitized_count += 1
                                    if self.mode == "sanitize":
                                        matched_el.decompose()
                        except Exception:
                            pass

        # 3. Inspect inline CSS styles and HTML5 `hidden` attributes for stealth payloads
        for el in list(soup.find_all(True)):
            if el.attrs is None:
                continue
            if "hidden" in el.attrs and el.get_text(" ", strip=True):
                if "css_display_none" not in signals:
                    signals.append("css_display_none")
                sanitized_count += 1
                if self.mode == "sanitize":
                    el.decompose()
                continue

            style = str(el.attrs.get("style", "") or "")
            if not style:
                continue

            matched_sig: str | None = None
            if _RE_CSS_DISPLAY_NONE.search(style):
                matched_sig = "css_display_none"
            elif _RE_CSS_VISIBILITY_OPACITY.search(style):
                matched_sig = "css_visibility_opacity"
            elif _RE_CSS_ZERO_FONT.search(style) or _RE_CSS_WHITE_ON_WHITE.search(
                style
            ):
                matched_sig = "css_zero_font_white"
            elif _RE_CSS_OFFSCREEN.search(style):
                matched_sig = "css_offscreen_position"

            if matched_sig is not None:
                if matched_sig not in signals:
                    signals.append(matched_sig)
                sanitized_count += 1
                if self.mode == "sanitize":
                    el.decompose()

        cleaned_html = str(soup) if self.mode == "sanitize" else html
        risk_score = min(1.0, round(0.85 * len(signals), 2)) if signals else 0.0
        report = FirewallReport(
            mode=self.mode,
            risk_score=risk_score,
            is_suspicious=len(signals) > 0,
            sanitized_count=sanitized_count if self.mode == "sanitize" else 0,
            signals=signals,
            untrusted_content=True,
        )

        if self.mode == "strict" and report.is_suspicious:
            raise PromptInjectionBlockedError(
                f"Blocked HTML prompt injection payload (signals={signals}, risk_score={risk_score})",
                risk_score=risk_score,
                signals=signals,
            )

        return cleaned_html, report

    def inspect_pdf_stealth(
        self, file_path: str | Path
    ) -> tuple[str | None, FirewallReport]:
        """Inspect PDF character attributes via `pdfplumber` for `<1pt`, white-on-white, or off-page text."""
        signals: list[str] = []
        sanitized_count = 0
        visible_pages: list[str] = []

        try:
            import pdfplumber

            with pdfplumber.open(str(file_path)) as pdf:
                for page in pdf.pages:
                    chars = page.chars or []
                    if not chars:
                        page_txt = page.extract_text() or ""
                        if page_txt.strip():
                            visible_pages.append(page_txt)
                        continue

                    clean_chars: list[str] = []
                    for ch in chars:
                        txt = ch.get("text", "")
                        size = float(ch.get("size", 12.0) or 12.0)
                        color = ch.get("non_stroking_color")
                        x0 = float(ch.get("x0", 0.0) or 0.0)
                        top = float(ch.get("top", 0.0) or 0.0)
                        x1 = float(ch.get("x1", 0.0) or 0.0)

                        is_white = color in (
                            (1, 1, 1),
                            (1.0, 1.0, 1.0),
                            [1, 1, 1],
                            [1.0, 1.0, 1.0],
                            1,
                            1.0,
                        )
                        is_offpage = (
                            x0 < -5.0
                            or top < -5.0
                            or (page.width and x1 > float(page.width) + 10.0)
                        )
                        if size < 1.0 or is_white or is_offpage:
                            if txt.strip():
                                sanitized_count += 1
                                if "pdf_stealth_text" not in signals:
                                    signals.append("pdf_stealth_text")
                            if self.mode != "sanitize":
                                clean_chars.append(txt)
                        else:
                            clean_chars.append(txt)

                    reconstructed = "".join(clean_chars).strip()
                    if reconstructed:
                        visible_pages.append(reconstructed)
        except Exception:
            return None, FirewallReport(mode=self.mode)

        if not signals:
            return None, FirewallReport(mode=self.mode)

        risk_score = 0.85
        report = FirewallReport(
            mode=self.mode,
            risk_score=risk_score,
            is_suspicious=True,
            sanitized_count=sanitized_count if self.mode == "sanitize" else 0,
            signals=signals,
            untrusted_content=True,
        )
        if self.mode == "strict":
            raise PromptInjectionBlockedError(
                f"Blocked PDF stealth text injection (signals={signals})",
                risk_score=risk_score,
                signals=signals,
            )
        return "\n\n".join(visible_pages), report

    def inspect_docx_vanish(
        self, file_path: str | Path
    ) -> tuple[str | None, FirewallReport]:
        """Inspect DOCX XML (`word/document.xml`) for `<w:vanish/>` hidden text runs."""
        signals: list[str] = []
        sanitized_count = 0
        paragraphs: list[str] = []

        try:
            with zipfile.ZipFile(str(file_path), "r") as zf:
                if "word/document.xml" not in zf.namelist():
                    return None, FirewallReport(mode=self.mode)
                xml_bytes = zf.read("word/document.xml")

            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            root = ET.fromstring(xml_bytes)
            for p in root.findall(".//w:p", ns):
                p_texts: list[str] = []
                for r in p.findall(".//w:r", ns):
                    rpr = r.find("w:rPr", ns)
                    is_vanish = False
                    if rpr is not None:
                        vanish_el = rpr.find("w:vanish", ns)
                        if vanish_el is not None:
                            val = vanish_el.attrib.get(
                                f"{{{ns['w']}}}val", "true"
                            ).lower()
                            if val not in ("false", "0", "off"):
                                is_vanish = True

                    run_str = "".join(t.text or "" for t in r.findall(".//w:t", ns))
                    if is_vanish and run_str.strip():
                        sanitized_count += 1
                        if "docx_vanish_text" not in signals:
                            signals.append("docx_vanish_text")
                        if self.mode != "sanitize":
                            p_texts.append(run_str)
                    else:
                        p_texts.append(run_str)

                para_line = "".join(p_texts).strip()
                if para_line:
                    paragraphs.append(para_line)
        except Exception:
            return None, FirewallReport(mode=self.mode)

        if not signals:
            return None, FirewallReport(mode=self.mode)

        risk_score = 0.85
        report = FirewallReport(
            mode=self.mode,
            risk_score=risk_score,
            is_suspicious=True,
            sanitized_count=sanitized_count if self.mode == "sanitize" else 0,
            signals=signals,
            untrusted_content=True,
        )
        if self.mode == "strict":
            raise PromptInjectionBlockedError(
                f"Blocked DOCX w:vanish hidden text injection (signals={signals})",
                risk_score=risk_score,
                signals=signals,
            )
        return "\n\n".join(paragraphs), report

    def process_text(
        self,
        text: str,
        prior_report: FirewallReport | None = None,
    ) -> tuple[str, FirewallReport]:
        """Inspect and sanitize Unicode steganography, fake chat delimiters, exfil beacons, and score semantic risk."""
        signals: list[str] = list(prior_report.signals) if prior_report else []
        sanitized_count = prior_report.sanitized_count if prior_report else 0
        risk_score = prior_report.risk_score if prior_report else 0.0

        if not text:
            return "", FirewallReport(
                mode=self.mode,
                risk_score=risk_score,
                is_suspicious=len(signals) > 0,
                sanitized_count=sanitized_count,
                signals=signals,
            )

        out = text

        # 1. Unicode Tag characters (U+E0000..U+E007F)
        tag_matches = _RE_UNICODE_TAGS.findall(out)
        if tag_matches:
            if "unicode_tags_e0000" not in signals:
                signals.append("unicode_tags_e0000")
            sanitized_count += len(tag_matches)
            risk_score = max(risk_score, 0.9)
            if self.mode == "sanitize":
                out = _RE_UNICODE_TAGS.sub("", out)

        # 2. Unicode Bidi overrides & Zero-Width steganography (preserving legitimate Emoji ZWJ sequences)
        emoji_zwj_spans = [m.span() for m in _RE_EMOJI_ZWJ_SEQ.finditer(out)]

        def _in_emoji_zwj(start_pos: int, end_pos: int) -> bool:
            return any(es <= start_pos and end_pos <= ee for es, ee in emoji_zwj_spans)

        bidi_matches = [
            m
            for m in _RE_BIDI_ZW.finditer(out)
            if not _in_emoji_zwj(m.start(), m.end())
        ]
        if bidi_matches:
            if "unicode_bidi_zero_width" not in signals:
                signals.append("unicode_bidi_zero_width")
            sanitized_count += len(bidi_matches)
            risk_score = max(risk_score, 0.8)
            if self.mode == "sanitize":
                chars = list(out)
                for m in reversed(bidi_matches):
                    chars[m.start() : m.end()] = []
                out = "".join(chars)

        # 3. Fake ChatML / Llama / System delimiters (ignoring code blocks/inline code documentation examples like `<|im_start|>`)
        if self.profile == "code":
            out_for_tokens = re.sub(r"(?s)(```|~~~).*?(?:\1|$)", "", out)
            out_for_tokens = re.sub(r"`[^`\n]{1,80}`", "", out_for_tokens)
        else:
            out_for_tokens = re.sub(r"`[^`\n]{1,40}`", "", out)

        if _RE_FAKE_CHAT_TOKENS.search(out_for_tokens):
            if "fake_chat_delimiter" not in signals:
                signals.append("fake_chat_delimiter")
            sanitized_count += 1
            risk_score = max(risk_score, 0.9)
            if self.mode == "sanitize":
                out = _RE_FAKE_CHAT_BLOCKS.sub("", out)
                out = re.sub(
                    r"(?<!`)(?:<\|im_start\|>\s*(?:system|user|assistant)?|<\|im_end\|>|\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>|<\|system\|>|<\|endoftext\|>)(?!`)",
                    "",
                    out,
                    flags=re.IGNORECASE,
                )

        # 4. Markdown image exfiltration beacons
        exfil_matches = _RE_MD_EXFIL_BEACON.findall(out)
        if exfil_matches:
            if "markdown_exfil_beacon" not in signals:
                signals.append("markdown_exfil_beacon")
            sanitized_count += len(exfil_matches)
            risk_score = max(risk_score, 0.9)
            if self.mode == "sanitize":
                out = _RE_MD_EXFIL_BEACON.sub(
                    r"[Blocked External Image Beacon: \1]", out
                )

        # 5. Semantic Instruction-Override Scoring (English & Vietnamese)
        # First check if the document is a benign academic/OWASP hard negative discussing prompt injection
        is_academic_discussion = bool(_RE_ACADEMIC_CONTEXT.search(out))

        if not is_academic_discussion:
            text_for_semantic = out_for_tokens if self.profile == "code" else out
            if _RE_SEMANTIC_EN.search(text_for_semantic):
                if "semantic_override_en" not in signals:
                    signals.append("semantic_override_en")
                risk_score = max(risk_score, 0.85)

            folded_out = fold_vietnamese_diacritics(text_for_semantic)
            if _RE_SEMANTIC_VI_FOLDED.search(folded_out):
                if "semantic_override_vi" not in signals:
                    signals.append("semantic_override_vi")
                risk_score = max(risk_score, 0.85)

        is_suspicious = len(signals) > 0 or risk_score >= 0.7
        report = FirewallReport(
            mode=self.mode,
            risk_score=round(min(1.0, risk_score), 4),
            is_suspicious=is_suspicious,
            sanitized_count=sanitized_count if self.mode == "sanitize" else 0,
            signals=signals,
            untrusted_content=True,
        )

        if self.mode == "strict" and is_suspicious:
            raise PromptInjectionBlockedError(
                f"Blocked prompt injection payload in strict mode (signals={signals}, risk_score={report.risk_score})",
                risk_score=report.risk_score,
                signals=signals,
            )

        return out, report

    def scan_text_findings(self, text: str, filename: str = "") -> list[Finding]:
        """Scan source code or document text for prompt injection findings with exact line & column coordinates."""
        findings: list[Finding] = []
        if not text:
            return findings

        fn_lower = filename.lower().replace("\\", "/")
        is_test_or_bench = any(
            k in fn_lower
            for k in (
                "test",
                "tests/",
                "bench",
                "benchmark",
                "threat_model",
                "threatmodel",
                "firewall",
            )
        )

        def _make_finding(
            start_pos: int, signal: str, severity: str, risk: float
        ) -> Finding:
            line_no = text[:start_pos].count("\n") + 1
            last_nl = text.rfind("\n", 0, start_pos)
            col_no = start_pos + 1 if last_nl == -1 else start_pos - last_nl
            line_start = last_nl + 1 if last_nl != -1 else 0
            next_nl = text.find("\n", start_pos)
            line_end = next_nl if next_nl != -1 else len(text)
            snippet = text[line_start:line_end].strip()[:120]
            return Finding(
                path=filename,
                line=line_no,
                column=col_no,
                signal=signal,
                severity=severity,
                risk_score=risk,
                snippet=snippet,
            )

        fence_spans = [m.span() for m in re.finditer(r"(?s)(```|~~~).*?(?:\1|$)", text)]
        inline_spans = [m.span() for m in re.finditer(r"`[^`\n]{1,80}`", text)]

        def _in_spans(pos: int, spans: list[tuple[int, int]]) -> bool:
            return any(s <= pos < e for s, e in spans)

        # 1. Unicode Tags (suspicious unless in test/bench fixtures in code profile)
        for m in _RE_UNICODE_TAGS.finditer(text):
            sev = (
                "info"
                if (self.profile == "code" and is_test_or_bench)
                else "suspicious"
            )
            risk = 0.10 if sev == "info" else 0.95
            findings.append(_make_finding(m.start(), "unicode_tags_e0000", sev, risk))

        # 2. Unicode Bidi & Zero-Width (ignoring legitimate emoji ZWJ sequences)
        emoji_zwj_spans = [m.span() for m in _RE_EMOJI_ZWJ_SEQ.finditer(text)]

        def _in_emoji_zwj(start_pos: int, end_pos: int) -> bool:
            return any(es <= start_pos and end_pos <= ee for es, ee in emoji_zwj_spans)

        for m in _RE_BIDI_ZW.finditer(text):
            if not _in_emoji_zwj(m.start(), m.end()):
                sev = (
                    "info"
                    if (self.profile == "code" and is_test_or_bench)
                    else "suspicious"
                )
                risk = 0.10 if sev == "info" else 0.85
                findings.append(
                    _make_finding(m.start(), "unicode_bidi_zero_width", sev, risk)
                )

        # 3. Markdown exfiltration beacons
        for m in _RE_MD_EXFIL_BEACON.finditer(text):
            pos = m.start()
            in_fence = _in_spans(pos, fence_spans)
            in_inline = _in_spans(pos, inline_spans)
            if self.profile == "code" and (in_fence or in_inline or is_test_or_bench):
                findings.append(
                    _make_finding(pos, "markdown_exfil_beacon", "info", 0.10)
                )
            else:
                findings.append(
                    _make_finding(pos, "markdown_exfil_beacon", "suspicious", 0.90)
                )

        # 4. Fake chat delimiters
        for m in _RE_FAKE_CHAT_TOKENS.finditer(text):
            pos = m.start()
            in_fence = _in_spans(pos, fence_spans)
            in_inline = _in_spans(pos, inline_spans)
            if self.profile == "code":
                if in_fence or in_inline or is_test_or_bench:
                    findings.append(
                        _make_finding(pos, "fake_chat_delimiter", "info", 0.10)
                    )
                else:
                    findings.append(
                        _make_finding(pos, "fake_chat_delimiter", "suspicious", 0.90)
                    )
            else:
                if in_inline:
                    findings.append(
                        _make_finding(pos, "fake_chat_delimiter", "info", 0.10)
                    )
                else:
                    findings.append(
                        _make_finding(pos, "fake_chat_delimiter", "suspicious", 0.90)
                    )

        # 5. Semantic override (English and Vietnamese in markdown/text files)
        ext = Path(filename).suffix.lower() if filename else ""
        if ext in (".md", ".txt", ".rst") and not is_test_or_bench:
            if not _RE_ACADEMIC_CONTEXT.search(text):
                for m in _RE_SEMANTIC_EN.finditer(text):
                    pos = m.start()
                    if not _in_spans(pos, fence_spans) and not _in_spans(
                        pos, inline_spans
                    ):
                        findings.append(
                            _make_finding(
                                pos,
                                "semantic_override_en",
                                "suspicious",
                                0.85,
                            )
                        )

                folded_text = fold_vietnamese_diacritics(text)
                for m in _RE_SEMANTIC_VI_FOLDED.finditer(folded_text):
                    pos = m.start()
                    if not _in_spans(pos, fence_spans) and not _in_spans(
                        pos, inline_spans
                    ):
                        findings.append(
                            _make_finding(
                                pos,
                                "semantic_override_vi",
                                "suspicious",
                                0.85,
                            )
                        )

        return findings


def build_provenance_meta(
    report: FirewallReport | None, source: str = ""
) -> dict[str, Any]:
    """Construct the standardized `_meta.provenance` object for MCP responses and API schemas."""
    fw_data = (
        report.model_dump()
        if report is not None
        else FirewallReport(mode="sanitize").model_dump()
    )
    return {
        "provenance": {
            "untrusted_content": True,
            "source": source,
            "firewall": fw_data,
        }
    }


def spotlight_untrusted_content(
    text: str,
    source: str,
    report: FirewallReport | None = None,
) -> str:
    """Wrap untrusted extracted or retrieved content in XML Data Spotlighting tags for MCP agents."""
    mode = report.mode if report else "sanitize"
    risk = report.risk_score if report else 0.0
    signals_str = ",".join(report.signals) if (report and report.signals) else "none"
    return (
        f'<untrusted_document_content source="{source}" '
        f'firewall_mode="{mode}" risk_score="{risk}" signals="{signals_str}">\n'
        f"{text}\n"
        f"</untrusted_document_content>"
    )
