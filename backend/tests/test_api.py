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
    assert data["version"] == "3.2.0"


def test_doctor_and_audit_endpoints():
    doc_res = client.get("/api/doctor")
    assert doc_res.status_code == 200
    doc_data = doc_res.json()
    assert doc_data["embedding_dimension"] == 384
    assert "active_provider" in doc_data

    aud_res = client.get("/api/kb/audit")
    assert aud_res.status_code == 200
    aud_data = aud_res.json()
    assert "status" in aud_data


def test_ocr_endpoint():
    import io
    from PIL import Image

    # Create synthetic test image
    img = Image.new("RGB", (300, 100), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    res = client.post("/api/ocr", files={"file": ("test_scan.png", buf, "image/png")})
    assert res.status_code == 200, f"OCR API failed: {res.text}"
    data = res.json()
    assert "markdown" in data
    assert "backend" in data
    assert "confidence" in data


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
        assert body["_meta"]["provenance"]["untrusted_content"] is True
        assert "firewall" in body["_meta"]["provenance"]


if __name__ == "__main__":
    test_health_endpoint()
    test_doctor_and_audit_endpoints()
    test_ocr_endpoint()
    test_sample_files_extraction()
    print("[PASS] All sample file API extractions, OCR, doctor & audit endpoints verified via TestClient!")
