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

# 2. Heavy files (Simulate large documents > 1MB)
heavy_text = "This is a heavy paragraph containing useless noise and some valuable information. " * 50000 # ~4MB file
with open(os.path.join(SAMPLES_DIR, "heavy_sample.txt"), "w", encoding="utf-8") as f:
    f.write(heavy_text)

# 3. Binary files (Download from public URLs)
urls = {
    "sample.pdf": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
    "sample.docx": "https://calibre-ebook.com/downloads/demos/demo.docx",
    "sample.jpg": "https://picsum.photos/800/600.jpg",
    "sample.mp3": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
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
