import os
import requests

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

# 1. Text-based files
texts = {
    "sample.txt": "This is a simple text file.\nHello World!",
    "sample.csv": "id,name,age\n1,Alice,30\n2,Bob,25",
    "sample.json": '{"name": "GoldPan", "version": "1.2"}',
    "sample.md": "# Markdown\n\nThis is a **markdown** file.",
    "sample.xml": "<note><to>User</to><from>GoldPan</from><body>Hello!</body></note>"
}

for filename, content in texts.items():
    with open(os.path.join(SAMPLES_DIR, filename), "w", encoding="utf-8") as f:
        f.write(content)

# 2. Binary files (Download from public URLs)
urls = {
    "sample.pdf": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
    "sample.docx": "https://calibre-ebook.com/downloads/demos/demo.docx",
    "sample.jpg": "https://via.placeholder.com/150",
    # MP3, XLSX might be too large or hard to find stable URLs, let's use tiny equivalents or skip binary if not needed.
}

for filename, url in urls.items():
    path = os.path.join(SAMPLES_DIR, filename)
    if not os.path.exists(path):
        print(f"Downloading {filename}...")
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

print("Sample generation complete.")
