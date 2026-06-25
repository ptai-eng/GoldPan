import sys
import os
import logging
logging.basicConfig(level=logging.DEBUG)

# Fix path to allow importing from backend/goldpan
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from goldpan.extractors.web_extractor import HybridWebExtractor

try:
    extractor = HybridWebExtractor(use_playwright_fallback=True)
    print("Extracting...")
    res = extractor.extract("https://github.com/ptai-eng/GoldPan")
    print("Done!")
    print(f"Extracted {len(res.text)} characters")
except Exception as e:
    print(f"Error: {e}")
