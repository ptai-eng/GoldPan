import os
import time
from pprint import pprint

from goldpan.core.pipeline import GoldPanPipeline
from goldpan.extractors.web_extractor import HybridWebExtractor
from goldpan.extractors.markitdown_wrapper import MarkItDownExtractor
from goldpan.extractors.liteparse_wrapper import LiteParseExtractor
from goldpan.filters.heuristic_filter import HeuristicFilter
from goldpan.filters.semantic_router import SemanticRouterFilter
from goldpan.formatters.markdown_formatter import AIFriendlyMarkdownFormatter

def test_web_extractor():
    print("=== Testing Web Extractor (URL) ===")
    url = "https://example.com"
    print(f"URL: {url}")
    
    # using requests only for fast test
    extractor = HybridWebExtractor(use_playwright_fallback=False) 
    pipeline = GoldPanPipeline(extractor=extractor)
    pipeline.add_filter(HeuristicFilter())
    pipeline.add_filter(SemanticRouterFilter())
    pipeline.set_formatter(AIFriendlyMarkdownFormatter())
    
    start_time = time.time()
    try:
        result = pipeline.process(url)
        elapsed = time.time() - start_time
        print(f"Time taken: {elapsed:.2f}s")
        print("--- Output Snippet ---")
        print(result.text[:500])
        print("----------------------\n")
    except Exception as e:
        print(f"Error: {e}")

def test_file_extractor():
    print("=== Testing File Extractor (MarkItDown) ===")
    test_file = "test_document.txt"
    # Create dummy file with noise
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("Header: Example Company\nCookie policy\nSubscribe to newsletter\n\nThis is the core knowledge of the document that should be preserved. It has a lot of important facts and data points.\n\nAll rights reserved.\n")
    
    print(f"File: {test_file}")
    
    extractor = MarkItDownExtractor()
    pipeline = GoldPanPipeline(extractor=extractor)
    pipeline.add_filter(HeuristicFilter())
    pipeline.add_filter(SemanticRouterFilter())
    pipeline.set_formatter(AIFriendlyMarkdownFormatter())
    
    start_time = time.time()
    try:
        result = pipeline.process(test_file)
        elapsed = time.time() - start_time
        print(f"Time taken: {elapsed:.2f}s")
        print("--- Cleaned Output ---")
        print(result.text)
        print("----------------------\n")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

def test_liteparse_extractor():
    print("=== Testing LiteParse Extractor (PDF) ===")
    test_file = "test_document.pdf"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("%PDF-1.4 mock content")
    
    print(f"File: {test_file}")
    
    extractor = LiteParseExtractor()
    pipeline = GoldPanPipeline(extractor=extractor)
    pipeline.add_filter(HeuristicFilter())
    pipeline.add_filter(SemanticRouterFilter())
    pipeline.set_formatter(AIFriendlyMarkdownFormatter())
    
    start_time = time.time()
    try:
        result = pipeline.process(test_file)
        elapsed = time.time() - start_time
        print(f"Time taken: {elapsed:.2f}s")
        print("--- Cleaned Output ---")
        print(result.text)
        print("----------------------\n")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    test_web_extractor()
    test_file_extractor()
    test_liteparse_extractor()
