import os
import requests
import json

BASE_URL = "http://localhost:8000"
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")

def print_result(name, res):
    if res.status_code == 200:
        data = res.json()
        print(f"[OK] {name}: SUCCESS (Extracted {len(data.get('markdown', ''))} chars)")
    else:
        print(f"[FAIL] {name}: FAILED - {res.status_code} {res.text}")

def test_urls():
    print("\n--- Testing URL Extractions ---")
    
    # 1. YouTube
    print("Testing YouTube URL...")
    res = requests.post(f"{BASE_URL}/api/extract/url", json={"url": "https://www.youtube.com/watch?v=jNQXAC9IVRw"})
    print_result("YouTube", res)
    
    # 2. GitHub Repo (Trafilatura/Playwright)
    print("Testing GitHub Repo URL...")
    res = requests.post(f"{BASE_URL}/api/extract/url", json={"url": "https://github.com/ptai-eng/GoldPan"})
    print_result("GitHub Repo", res)
    
    # 3. Standard Article
    print("Testing Standard Article URL...")
    res = requests.post(f"{BASE_URL}/api/extract/url", json={"url": "https://en.wikipedia.org/wiki/Data_extraction"})
    print_result("Standard Article", res)

def test_files():
    print("\n--- Testing File Extractions ---")
    files_to_test = [
        "sample.txt", "sample.csv", "sample.json", "sample.md", "sample.xml",
        "sample.pdf", "sample.docx", "sample.jpg", "sample.mp3", "heavy_sample.txt"
    ]
    
    for filename in files_to_test:
        path = os.path.join(SAMPLES_DIR, filename)
        if not os.path.exists(path):
            print(f"⚠️ Skipping {filename}: File not found")
            continue
            
        print(f"Testing {filename}...")
        with open(path, "rb") as f:
            res = requests.post(f"{BASE_URL}/api/extract/file", files={"file": (filename, f)})
        print_result(filename, res)

if __name__ == "__main__":
    try:
        # Check if server is running
        requests.get(f"{BASE_URL}/health")
        print("Server is up. Running tests...")
        test_urls()
        test_files()
    except requests.exceptions.ConnectionError:
        print("Backend server is not running on localhost:8000. Please start it first.")
