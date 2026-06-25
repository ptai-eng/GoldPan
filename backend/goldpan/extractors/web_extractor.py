import requests
import trafilatura
from bs4 import BeautifulSoup
import re
from goldpan.core.schemas import RawContent, DocumentInfo

class HybridWebExtractor:
    def __init__(self, use_playwright_fallback: bool = True):
        self.use_playwright_fallback = use_playwright_fallback

    def _extract_youtube_id(self, url: str) -> str:
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
        return match.group(1) if match else None

    def _extract_youtube(self, url: str, video_id: str) -> RawContent:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            ytt_api = YouTubeTranscriptApi()
            transcript_list = ytt_api.list(video_id)
            transcript = transcript_list.find_transcript(['en', 'vi'])
            data = transcript.fetch()
            text = "\n".join([getattr(t, 'text', str(t)) for t in data])
            doc_info = DocumentInfo(source_url=url, title=f"YouTube Video - {video_id}")
            return RawContent(text=text, doc_info=doc_info, original_format="youtube")
        except Exception as e:
            raise ValueError(f"Could not extract YouTube transcript: {str(e)}")

    def extract(self, url: str) -> RawContent:
        yt_id = self._extract_youtube_id(url)
        if yt_id and "youtube.com" in url or "youtu.be" in url:
            return self._extract_youtube(url, yt_id)

        # Basic Github blob to raw conversion
        if "github.com" in url and "/blob/" in url:
            url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

        # Step 1: Try fast requests
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            html = response.text
            
            # Simple heuristic to detect if page is mostly empty / requires JS
            soup = BeautifulSoup(html, 'html.parser')
            text_len = len(soup.get_text(strip=True))
            
            if text_len < 500 and self.use_playwright_fallback:
                html = self._extract_with_playwright(url)
                
        except Exception as e:
            if self.use_playwright_fallback:
                html = self._extract_with_playwright(url)
            else:
                raise e
                
        # Trafilatura can extract main content from HTML natively
        extracted_text = trafilatura.extract(html, include_links=True, include_images=True, include_tables=True)
        
        # Fallback if trafilatura fails
        if not extracted_text:
            extracted_text = soup.get_text(separator="\n", strip=True) if 'soup' in locals() else html
            
        doc_info = DocumentInfo(source_url=url, title=soup.title.string if 'soup' in locals() and soup.title else url)
        return RawContent(
            text=extracted_text,
            doc_info=doc_info,
            original_format="html"
        )

    def _extract_with_playwright(self, url: str) -> str:
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                # Use domcontentloaded to avoid hanging on GitHub websockets
                page.goto(url, wait_until="domcontentloaded", timeout=15000)
                html = page.content()
                browser.close()
                return html
        except Exception as e:
            print(f"[Playwright Fallback] Failed: {e}")
            return f"<html><body><h1>Extraction Failed</h1><p>{str(e)}</p></body></html>"
