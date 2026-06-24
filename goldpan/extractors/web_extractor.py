import requests
import trafilatura
from bs4 import BeautifulSoup
from goldpan.core.schemas import RawContent, DocumentInfo

class HybridWebExtractor:
    def __init__(self, use_playwright_fallback: bool = True):
        self.use_playwright_fallback = use_playwright_fallback

    def extract(self, url: str) -> RawContent:
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
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle")
            html = page.content()
            browser.close()
            return html
