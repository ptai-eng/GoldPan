import os
import sys
from fastapi.testclient import TestClient

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from main import app

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")
client = TestClient(app)


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["version"] == "2.0.0"


def test_sample_files_extraction():
    offline_samples = [
        "sample.txt",
        "sample.csv",
        "sample.json",
        "sample.md",
        "sample.xml",
        "sample.pdf",
        "sample.docx",
        "sample.jpg",
    ]
    for filename in offline_samples:
        path = os.path.join(SAMPLES_DIR, filename)
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            res = client.post("/api/extract/file", files={"file": (filename, f)})
        assert res.status_code == 200, f"Failed on {filename}: {res.text}"
        body = res.json()
        assert len(body.get("markdown", "")) > 0
        assert body["metadata"]["file_path"] == filename


if __name__ == "__main__":
    test_health_endpoint()
    test_sample_files_extraction()
    print("[PASS] All sample file API extractions verified via TestClient!")
