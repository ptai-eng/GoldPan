import re
from urllib.parse import urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup

from goldpan.core.schemas import DocumentInfo, RawContent
from goldpan.firewall.context_firewall import AgentContextFirewall

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36 GoldPanAI/3.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

YOUTUBE_ID_PATTERN = re.compile(
    r"(?:v=|youtu\.be/|/embed/|/shorts/|/v/)([0-9A-Za-z_-]{11})(?:[^0-9A-Za-z_-]|$)"
)


class HybridWebExtractor:
    def __init__(
        self,
        use_playwright_fallback: bool = True,
        firewall_mode: str = "sanitize",
    ) -> None:
        self.use_playwright_fallback = use_playwright_fallback
        self.firewall_mode = firewall_mode
        self.firewall = AgentContextFirewall(mode=firewall_mode)

    def _validate_url(self, url: str) -> str:
        parsed = urlparse(url.strip())
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"Invalid URL scheme or host: {url}")
        host = (parsed.hostname or "").lower()
        if host in ("169.254.169.254", "metadata.google.internal"):
            raise ValueError("Blocked cloud metadata endpoint.")
        return url.strip()

    def _extract_youtube_id(self, url: str) -> str | None:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if not ("youtube.com" in host or "youtu.be" in host):
            return None
        match = YOUTUBE_ID_PATTERN.search(url)
        return match.group(1) if match else None

    def _extract_youtube(self, url: str, video_id: str) -> RawContent:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi

            ytt_api = YouTubeTranscriptApi()
            transcript_list = ytt_api.list(video_id)
            try:
                transcript = transcript_list.find_transcript(["en", "vi"])
            except Exception:
                transcript = next(iter(transcript_list))
            data = transcript.fetch()
            text = "\n".join([getattr(t, "text", str(t)) for t in data])
            doc_info = DocumentInfo(source_url=url, title=f"YouTube Video - {video_id}")
            return RawContent(text=text, doc_info=doc_info, original_format="youtube")
        except Exception as e:
            raise ValueError(f"Could not extract YouTube transcript: {str(e)}")

    def extract(self, url: str) -> RawContent:
        url = self._validate_url(url)

        yt_id = self._extract_youtube_id(url)
        if yt_id:
            return self._extract_youtube(url, yt_id)

        if "github.com" in url and "/blob/" in url:
            url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

        html = ""
        soup = None
        computed_hidden_stripped = 0

        try:
            response = requests.get(url, headers=DEFAULT_HEADERS, timeout=12)
            response.raise_for_status()
            html = response.text
            soup = BeautifulSoup(html, "html.parser")
            text_len = len(soup.get_text(strip=True))

            if text_len < 500 and self.use_playwright_fallback:
                pw_html, stripped = self._extract_with_playwright(url)
                if pw_html:
                    html = pw_html
                    computed_hidden_stripped += stripped
                    soup = BeautifulSoup(html, "html.parser")
        except Exception as req_err:
            if self.use_playwright_fallback:
                pw_html, stripped = self._extract_with_playwright(url)
                if not pw_html:
                    raise req_err
                html = pw_html
                computed_hidden_stripped += stripped
                soup = BeautifulSoup(html, "html.parser")
            else:
                raise req_err

        # Step 1.5: Agent Context Firewall HTML/CSS DOM sanitization BEFORE Trafilatura
        cleaned_html, fw_report = self.firewall.process_html(html)
        if computed_hidden_stripped > 0:
            fw_report.sanitized_count += computed_hidden_stripped
            if "playwright_computed_css_hidden" not in fw_report.signals:
                fw_report.signals.append("playwright_computed_css_hidden")
            fw_report.is_suspicious = True
            fw_report.risk_score = max(fw_report.risk_score, 0.85)
            if self.firewall_mode == "strict":
                from ..core.exceptions import PromptInjectionBlockedError
                raise PromptInjectionBlockedError(
                    f"Prompt injection blocked by Tier-2 Playwright Computed-Style Firewall: {computed_hidden_stripped} elements hidden via external CSS.",
                    risk_score=fw_report.risk_score,
                    signals=fw_report.signals,
                )

        soup = BeautifulSoup(cleaned_html, "html.parser")

        # Step 2: Extract main content via Trafilatura
        extracted_text = trafilatura.extract(
            cleaned_html,
            include_links=True,
            include_images=True,
            include_tables=True,
        )

        if not extracted_text:
            extracted_text = (
                soup.get_text(separator="\n", strip=True) if soup is not None else cleaned_html
            )

        if not extracted_text or not extracted_text.strip():
            raise ValueError(f"No readable text content could be extracted from {url}")

        title = url
        if soup is not None and soup.title and soup.title.string:
            title = soup.title.string.strip()

        doc_info = DocumentInfo(
            source_url=url,
            title=title,
            metadata={"firewall": fw_report.model_dump()},
        )
        return RawContent(
            text=extracted_text,
            doc_info=doc_info,
            original_format="html",
        )

    def _extract_with_playwright(self, url: str) -> tuple[str | None, int]:
        """
        Tier-2 Computed-Style DOM Firewall:
        Uses headless Chromium via Playwright to evaluate rendered computed styles
        (window.getComputedStyle), stripping elements hidden via external CSS stylesheets,
        dynamic stylesheets, or transform off-screen positioning before serializing HTML.
        """
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(user_agent=DEFAULT_HEADERS["User-Agent"])
                page.goto(url, wait_until="domcontentloaded", timeout=15000)

                stripped_count = page.evaluate("""
                    () => {
                        let count = 0;
                        const elements = Array.from(document.querySelectorAll('*'));
                        for (const el of elements) {
                            try {
                                if (['HTML', 'HEAD', 'BODY', 'META', 'TITLE', 'SCRIPT', 'STYLE', 'LINK'].includes(el.tagName)) continue;
                                const style = window.getComputedStyle(el);
                                const isHidden = 
                                    style.display === 'none' ||
                                    style.visibility === 'hidden' ||
                                    parseFloat(style.opacity) === 0 ||
                                    parseFloat(style.fontSize) <= 1 ||
                                    style.clip === 'rect(0px, 0px, 0px, 0px)' ||
                                    style.clipPath === 'inset(50%)' ||
                                    (style.transform && style.transform.includes('-9999'));

                                if (isHidden) {
                                    el.remove();
                                    count++;
                                }
                            } catch (e) {}
                        }
                        return count;
                    }
                """)
                html = page.content()
                browser.close()
                return html, int(stripped_count or 0)
        except Exception as e:
            print(f"[Playwright Fallback] Warning: {e}")
            return None, 0
