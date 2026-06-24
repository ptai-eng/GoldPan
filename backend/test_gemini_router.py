from goldpan.filters.semantic_router import SemanticRouterFilter
from goldpan.core.schemas import RawContent
import os
from dotenv import load_dotenv

load_dotenv()

def test_gemini_router():
    router = SemanticRouterFilter(use_api=True)
    
    if not router.client:
        print("API model not initialized. Check API key.")
        return

    # Text contains some useful information and some obvious noise (menu/ads)
    test_text = """
Home | About Us | Contact | Login

Welcome to the ultimate guide on Machine Learning. Machine learning is a subfield of artificial intelligence (AI) that involves the development of algorithms and statistical models that enable computers to improve their performance on a specific task through experience. 

Click here to buy our premium shoes! Only $19.99 for a limited time! Subscribe to our newsletter.

Supervised learning algorithms build a mathematical model of a set of data that contains both the inputs and the desired outputs. The data is known as training data, and consists of a set of training examples.

Copyright 2026 AI Corp. All rights reserved. Follow us on Facebook and Twitter.
""" * 3

    content = RawContent(text=test_text, doc_info={"source": "test.txt"}, original_format="txt")
    filtered = router.apply(content)
    
    print("\n=== ORIGINAL TEXT ===")
    print(test_text.strip())
    
    print("\n=== FILTERED TEXT ===")
    print(filtered.text.strip())
    
    print("\n=== STATS ===")
    print(filtered.stats)

if __name__ == "__main__":
    test_gemini_router()
