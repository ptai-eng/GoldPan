import os
import io
import sys
import json
import math
import time
import tempfile
import shutil
import unicodedata
from unittest.mock import patch, MagicMock
import pytest

# Allow running directly from repository root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from typer.testing import CliRunner
from fastapi.testclient import TestClient
from goldpan.core.pipeline import GoldPanPipeline
from goldpan.core.schemas import RawContent, DocumentInfo
from goldpan.core.exceptions import ExtractionError
from goldpan.extractors.web_extractor import HybridWebExtractor
from goldpan.extractors.markitdown_wrapper import MarkItDownExtractor
from goldpan.extractors.liteparse_wrapper import LiteParseExtractor
from goldpan.filters.heuristic_filter import HeuristicFilter
from goldpan.filters.semantic_router import SemanticRouterFilter
from goldpan.formatters.markdown_formatter import AIFriendlyMarkdownFormatter
from goldpan.kb.chroma_manager import ChromaManager, compute_document_id, GEMINI_EMBEDDING_DIM
from goldpan.mcp_server import GoldPanMCPServer
from goldpan.cli import app as cli_app
import main as backend_main


def test_web_extractor():
    print("=== [1/10] Testing Hybrid Web Extractor (URL) ===")
    url = "https://example.com"
    extractor = HybridWebExtractor(use_playwright_fallback=False)
    pipeline = GoldPanPipeline(extractor=extractor)
    pipeline.add_filter(HeuristicFilter())
    pipeline.add_filter(SemanticRouterFilter(use_api=False))
    pipeline.set_formatter(AIFriendlyMarkdownFormatter())

    start_time = time.time()
    result = pipeline.process(url)
    elapsed = time.time() - start_time
    assert result.text, "Expected non-empty Markdown output from URL extraction"
    assert "Example Domain" in result.text
    print(f"[PASS] Web Extractor ({elapsed:.2f}s) -> {len(result.text)} chars\n")


def test_file_extractor():
    print("=== [2/10] Testing MarkItDown + Heuristic & Semantic Filters ===")
    test_file = os.path.join(ROOT_DIR, "test_document_tmp.txt")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(
            "Header: Example Company\n"
            "Cookie policy\n"
            "Subscribe to newsletter\n\n"
            "This is the core knowledge of the document that should be preserved. "
            "It has important technical facts and structured data points.\n\n"
            "All rights reserved.\n"
        )

    try:
        extractor = MarkItDownExtractor()
        pipeline = GoldPanPipeline(extractor=extractor)
        pipeline.add_filter(HeuristicFilter())
        pipeline.add_filter(SemanticRouterFilter(use_api=False))
        pipeline.set_formatter(AIFriendlyMarkdownFormatter())

        start_time = time.time()
        result = pipeline.process(test_file)
        elapsed = time.time() - start_time

        assert "core knowledge of the document" in result.text
        assert "Cookie policy" not in result.text
        assert "Subscribe to newsletter" not in result.text
        assert result.stats.get("heuristic_noise_removed_bytes", 0) > 0
        print(f"[PASS] MarkItDown + Filters ({elapsed:.2f}s) | Stats: {result.stats}\n")
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)


def test_liteparse_extractor():
    print("=== [3/10] Testing LiteParse Extractor (CSV -> Markdown Table) ===")
    test_file = os.path.join(ROOT_DIR, "test_table_tmp.csv")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("Model,Dimension,Mode\nall-MiniLM-L6-v2,384,Local ONNX\ngemini-embedding-001,768,Cloud API\n")

    try:
        extractor = LiteParseExtractor()
        pipeline = GoldPanPipeline(extractor=extractor)
        pipeline.add_filter(HeuristicFilter())
        pipeline.add_filter(SemanticRouterFilter(use_api=False))
        pipeline.set_formatter(AIFriendlyMarkdownFormatter())

        result = pipeline.process(test_file)
        assert "| Model | Dimension | Mode |" in result.text
        assert "all-MiniLM-L6-v2" in result.text
        print("[PASS] LiteParse CSV -> Markdown Table verified\n")
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)


def test_local_onnx_kb():
    print("=== [4/10] Testing Dual-Mode ChromaManager (100% Offline Local ONNX) ===")
    tmp_dir = tempfile.mkdtemp(prefix="goldpan_kb_test_")
    try:
        kb = ChromaManager(persist_directory=tmp_dir, prefer_local=True)
        mode = kb.ingest_document(
            document_id="doc-001",
            text="# GoldPan Architecture\n\nGoldPan uses ChromaDB with local ONNX all-MiniLM-L6-v2 embeddings for offline privacy-first vector retrieval.",
            metadata={"title": "GoldPan Architecture Spec", "file_path": "architecture.md", "source_url": None},
        )
        assert mode == "local_onnx", f"Expected local_onnx mode, got {mode}"

        docs = kb.get_all_documents()
        assert len(docs) == 1 and docs[0]["id"] == "doc-001"

        hits = kb.search_with_citations("What embedding model does GoldPan use offline?", n_results=1)
        assert len(hits) == 1
        assert "all-MiniLM-L6-v2" in hits[0]["text"]
        assert hits[0]["embedding_mode"] == "local_onnx"
        assert hits[0]["rrf_score"] > 0

        kb.delete_document("doc-001")
        assert len(kb.get_all_documents()) == 0
        print("[PASS] Local ONNX Knowledge Base (Ingest -> Semantic Search -> Citations -> Delete) verified\n")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_mcp_server_contracts():
    print("=== [5/10] Testing GoldPan MCP Server (JSON-RPC 2.0 + rea-tool-design Contracts) ===")
    tmp_chroma = tempfile.mkdtemp(prefix="goldpan_mcp_chroma_")
    tmp_kb = tempfile.mkdtemp(prefix="goldpan_mcp_kb_")
    sample_md = os.path.join(tmp_kb, "sample_input.md")
    with open(sample_md, "w", encoding="utf-8") as f:
        f.write("# GoldPan MCP Test\n\n" + ("Pure signal extracted for MCP agent consumption. " * 25))

    try:
        server = GoldPanMCPServer(persist_directory=tmp_chroma, kb_dir=tmp_kb, api_key="")

        # 1. initialize
        init_res = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        assert init_res["result"]["serverInfo"]["name"] == "goldpan-mcp"
        assert init_res["result"]["serverInfo"]["version"] == "2.1.0"

        # 2. tools/list
        tools_res = server.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tool_names = {t["name"] for t in tools_res["result"]["tools"]}
        assert {
            "goldpan_extract",
            "goldpan_kb_ingest",
            "goldpan_kb_search",
            "goldpan_kb_list",
            "goldpan_kb_delete",
        }.issubset(tool_names)

        # 3. tools/call -> goldpan_extract with summary truncation & rea-tool-design contracts
        ext_res = server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "goldpan_extract",
                    "arguments": {
                        "source": sample_md,
                        "detail": "summary",
                        "max_chars": 300,
                        "ingest_to_kb": True,
                    },
                },
            }
        )
        assert ext_res["result"]["_meta"]["provenance"]["untrusted_content"] is True
        ext_payload = json.loads(ext_res["result"]["content"][0]["text"])
        assert ext_payload["truncated"] is True
        assert ext_payload["coverage"]["completeness"] == "partial"
        assert len(ext_payload["residual_unknowns"]) >= 1
        doc_id = ext_payload["ingested"]["document_id"]

        # 4. tools/call -> goldpan_kb_search
        search_res = server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "goldpan_kb_search",
                    "arguments": {"query": "MCP agent consumption", "n_results": 2, "detail": "summary"},
                },
            }
        )
        search_payload = json.loads(search_res["result"]["content"][0]["text"])
        assert search_payload["results_count"] >= 1
        assert search_payload["results"][0]["rrf_score"] > 0
        assert search_payload["_meta"]["provenance"]["untrusted_content"] is True
        assert "coverage" in search_payload

        # 5. resources/list & resources/read
        res_list = server.handle_request({"jsonrpc": "2.0", "id": 5, "method": "resources/list"})
        uris = [r["uri"] for r in res_list["result"]["resources"]]
        target_uri = f"goldpan://kb/{doc_id}"
        assert target_uri in uris

        res_read = server.handle_request(
            {"jsonrpc": "2.0", "id": 6, "method": "resources/read", "params": {"uri": target_uri}}
        )
        assert "Pure signal extracted" in res_read["result"]["contents"][0]["text"]
        assert res_read["result"]["_meta"]["provenance"]["untrusted_content"] is True

        # 6. tools/call -> goldpan_kb_delete
        server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {"name": "goldpan_kb_delete", "arguments": {"document_id": doc_id}},
            }
        )
        list_after = server.handle_request(
            {"jsonrpc": "2.0", "id": 8, "method": "tools/call", "params": {"name": "goldpan_kb_list", "arguments": {}}}
        )
        assert json.loads(list_after["result"]["content"][0]["text"])["total_documents"] == 0
        print("[PASS] GoldPan MCP Server (initialize, tools, resources, provenance & coverage contracts) verified\n")
    finally:
        shutil.rmtree(tmp_chroma, ignore_errors=True)
        shutil.rmtree(tmp_kb, ignore_errors=True)


def test_cli_commands():
    print("=== [6/10] Testing GoldPan CLI (Typer + Rich Entrypoint) ===")
    runner = CliRunner()
    ver_res = runner.invoke(cli_app, ["version"])
    assert ver_res.exit_code == 0
    assert "GoldPan AI v2.1.0" in ver_res.stdout

    sample_file = os.path.join(ROOT_DIR, "README.md")
    ext_res = runner.invoke(cli_app, ["extract", sample_file, "--detail", "summary", "--max-chars", "250", "--json"])
    assert ext_res.exit_code == 0
    data = json.loads(ext_res.stdout)
    assert data["detail"] == "summary"
    assert data["_meta"]["provenance"]["untrusted_content"] is True
    print("[PASS] GoldPan CLI (`goldpan version` & `goldpan extract --json`) verified\n")


def test_algorithmic_edge_cases_and_mutations():
    print("=== [7/10] Testing Algorithmic Edge Cases & Mutation Guards (rea-ci-tests-audit) ===")
    raw_md = (
        "#Architecture Overview\n\n"
        "[Home](https://example.com/nav)\n"
        "Cookie policy\n\n"
        "[1] Vaswani et al. (2017) introduced the Transformer architecture in [Attention Is All You Need](https://arxiv.org/abs/1706.03762).\n\n"
        "In enterprise compliance, analyzing how a cookie policy interacts with GDPR data retention is critical for legal audits and cannot be stripped as boilerplate.\n\n"
        "## Code Example\n\n"
        "```python\n"
        "#!/usr/bin/env python3\n"
        "#include_guard\n"
        "x = 1\n\n"
        "y = 2\n"
        "```\n\n"
        "| Key | Value |\n"
        "| --- | --- |\n"
        "| A | B |\n\n"
        "- Bullet item one\n"
        "- Bullet item two\n\n"
        "Sign Up Now\n"
    )
    raw = RawContent(text=raw_md, doc_info=DocumentInfo(title="Architecture Overview"), original_format="md")
    h_filtered = HeuristicFilter().apply(raw)
    s_filtered = SemanticRouterFilter(use_api=False).apply(h_filtered)
    final_md = AIFriendlyMarkdownFormatter().format(s_filtered)

    assert "# Architecture Overview" in final_md.text
    assert "## Code Example" in final_md.text
    assert "[Home](https://example.com/nav)" not in final_md.text
    assert "\nCookie policy\n" not in final_md.text
    assert "Sign Up Now" not in final_md.text
    assert "[1] Vaswani et al. (2017)" in final_md.text
    assert "[Attention Is All You Need](https://arxiv.org/abs/1706.03762)" in final_md.text
    assert "analyzing how a cookie policy interacts with GDPR" in final_md.text
    assert "#!/usr/bin/env python3" in final_md.text
    assert "#include_guard" in final_md.text
    assert "x = 1\n\ny = 2" in final_md.text
    assert "| Key | Value |" in final_md.text
    assert "- Bullet item one" in final_md.text

    tmp_dir = tempfile.mkdtemp(prefix="goldpan_edge_")
    try:
        csv_path = os.path.join(tmp_dir, "edge.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write('Name,Formula\n"Alpha|Beta","Line1\nLine2"\n')
        csv_res = LiteParseExtractor().extract(csv_path)
        assert "Alpha\\|Beta" in csv_res.text
        assert "Line1<br>Line2" in csv_res.text

        bad_json = os.path.join(tmp_dir, "corrupt.json")
        with open(bad_json, "w", encoding="utf-8") as f:
            f.write("{corrupt_json: ")
        pipeline = GoldPanPipeline(extractor=LiteParseExtractor())
        pipeline.set_formatter(AIFriendlyMarkdownFormatter())
        with pytest.raises(ExtractionError):
            pipeline.process(bad_json)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    web_ext = HybridWebExtractor(use_playwright_fallback=False)
    assert web_ext._extract_youtube_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert web_ext._extract_youtube_id("https://youtu.be/dQw4w9WgXcQ?t=10") == "dQw4w9WgXcQ"
    assert web_ext._extract_youtube_id("https://youtu.be/") is None
    assert web_ext._extract_youtube_id("https://example.com/watch?v=dQw4w9WgXcQ") is None
    with pytest.raises(ValueError):
        web_ext.extract("http://169.254.169.254/latest/meta-data/")
    with pytest.raises(ValueError):
        web_ext.extract("file:///C:/Windows/win.ini")

    print("[PASS] Algorithmic Edge Cases & Mutation Guards verified\n")


def test_gemini_768d_and_rrf_fusion():
    print("=== [8/10] Testing Gemini 768d MRL Embeddings, 429 Retry & Cross-Collection RRF Fusion ===")
    tmp_dir = tempfile.mkdtemp(prefix="goldpan_rrf_test_")
    try:
        kb = ChromaManager(persist_directory=tmp_dir, prefer_local=False)
        kb.set_api_key("AIzaSyFakeKeyForUnitTestVerification12345")

        # 1. Ingest a local document into goldpan_kb_local (384d)
        kb.prefer_local = True
        kb.ingest_document(
            document_id="doc-local",
            text="# Offline Engine\n\nGoldPan runs all-MiniLM-L6-v2 ONNX locally in 384 dimensions.",
            metadata={"title": "Offline Engine Spec", "file_path": "offline.md"},
        )

        # 2. Mock Gemini embedding endpoint: first call returns 429, second call returns 768d vector
        kb.prefer_local = False
        captured_payloads = []
        call_count = {"n": 0}

        def fake_post(url, json=None, timeout=20):
            captured_payloads.append(json)
            call_count["n"] += 1
            resp = MagicMock()
            if call_count["n"] == 1:
                resp.ok = False
                resp.status_code = 429
                resp.text = "RESOURCE_EXHAUSTED"
                return resp
            resp.ok = True
            resp.status_code = 200
            raw_vec = [float((i % 7) + 1) for i in range(GEMINI_EMBEDDING_DIM)]
            resp.json.return_value = {"embedding": {"values": raw_vec}}
            return resp

        with patch("requests.post", side_effect=fake_post), patch("time.sleep", return_value=None):
            mode = kb.ingest_document(
                document_id="doc-cloud",
                text="# Cloud Engine\n\nGoldPan supports Gemini 768d Matryoshka embeddings with RRF fusion.",
                metadata={"title": "Cloud Engine Spec", "file_path": "cloud.md"},
            )
            assert mode == "gemini_api"
            assert captured_payloads[0]["outputDimensionality"] == 768

            # Verify L2 normalization on stored vector
            stored = kb.gemini_collection.get(include=["embeddings"])
            vec = stored["embeddings"][0]
            assert len(vec) == 768
            l2_norm = math.sqrt(sum(float(v) * float(v) for v in vec))
            assert math.isclose(l2_norm, 1.0, rel_tol=1e-4)

            # 3. Search across both 384d local_collection and 768d gemini_collection via RRF
            hits = kb.search_with_citations("Gemini 768d Matryoshka embeddings", n_results=2)
            assert len(hits) == 2
            modes_returned = {h["embedding_mode"] for h in hits}
            assert modes_returned == {"local_onnx", "gemini_api"}
            assert hits[0]["title"] == "Cloud Engine Spec"
            assert hits[0]["rrf_score"] > hits[1]["rrf_score"]

        print("[PASS] Gemini 768d MRL outputDimensionality, L2 Norm, 429 Backoff & RRF Fusion verified\n")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_idempotent_ingest_and_vietnamese_retrieval():
    print("=== [9/10] Testing Content-Hash Deduplication & Vietnamese NFD/Diacritic Retrieval ===")
    tmp_dir = tempfile.mkdtemp(prefix="goldpan_vi_test_")
    try:
        kb = ChromaManager(persist_directory=tmp_dir, prefer_local=True)

        # 1. Idempotent content-hash deduplication when re-ingesting same content + source
        vn_text_nfd = unicodedata.normalize(
            "NFD",
            "# Quy chế tính điểm Ban Tổ Chức\n\n"
            "Thí sinh nộp bài qua hệ thống chấm điểm tự động sẽ được đánh giá theo độ chính xác "
            "và thời gian phản hồi của thuật toán.",
        )
        doc_id_1 = compute_document_id(vn_text_nfd, "tinhdiem_bantochuc.md")
        doc_id_2 = compute_document_id(unicodedata.normalize("NFC", vn_text_nfd), "tinhdiem_bantochuc.md")
        assert doc_id_1 == doc_id_2, "NFC and NFD forms of the same Vietnamese text must hash to the same document_id"

        kb.ingest_document(
            document_id="auto",
            text=vn_text_nfd,
            metadata={"title": "Quy chế tính điểm", "file_path": "tinhdiem_bantochuc.md"},
        )
        kb.ingest_document(
            document_id="auto",
            text=vn_text_nfd,
            metadata={"title": "Quy chế tính điểm", "file_path": "tinhdiem_bantochuc.md"},
        )
        assert len(kb.get_all_documents()) == 1, "Re-ingesting identical document must not create duplicate entries"

        # 2. Add English distractor documents
        kb.ingest_document(
            document_id="auto",
            text="# Docker Deployment Guide\n\nConfigure container port bindings and healthchecks in docker-compose.yml.",
            metadata={"title": "Docker Guide", "file_path": "docker.md"},
        )

        # 3. Query in accented Vietnamese (NFC) AND unaccented Vietnamese (ASCII-folded)
        hits_accented = kb.search_with_citations("cách tính điểm của ban tổ chức", n_results=2)
        assert hits_accented[0]["title"] == "Quy chế tính điểm"

        hits_unaccented = kb.search_with_citations("quy che tinh diem ban to chuc", n_results=2)
        assert hits_unaccented[0]["title"] == "Quy chế tính điểm"

        print("[PASS] Content-Hash Deduplication & Vietnamese (NFC/NFD + Unaccented) Retrieval verified\n")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_fastapi_endpoints_and_security():
    print("=== [10/10] Testing FastAPI Endpoints, Path Traversal & Idempotent KB Ingest ===")
    tmp_chroma = tempfile.mkdtemp(prefix="goldpan_api_chroma_")
    orig_chroma = backend_main.chroma_manager
    backend_main.chroma_manager = ChromaManager(persist_directory=tmp_chroma, prefer_local=True)
    client = TestClient(backend_main.app)

    try:
        malicious_name = "../../etc/traversal_test.txt"
        content_bytes = b"Important engineering specification for GoldPan API security verification."
        res = client.post(
            "/api/extract/file",
            files={"file": (malicious_name, io.BytesIO(content_bytes), "text/plain")},
        )
        assert res.status_code == 200
        body = res.json()
        assert body["metadata"]["file_path"] == "traversal_test.txt"
        assert ".." not in body["metadata"]["file_path"]

        # Ingest twice -> verify deterministic document_id and zero duplicates
        payload = {
            "markdown": "# Security Spec\n\nGoldPan enforces 384-dim ONNX vectors in goldpan_kb_local.",
            "metadata": {"title": "Security Spec", "file_path": "sec.md"},
            "prefer_local": True,
        }
        ing_res_1 = client.post("/api/kb/ingest", json=payload).json()
        ing_res_2 = client.post("/api/kb/ingest", json=payload).json()
        assert ing_res_1["document_id"] == ing_res_2["document_id"]
        doc_id = ing_res_1["document_id"]

        docs_now = client.get("/api/kb/documents").json()["documents"]
        assert len(docs_now) == 1

        chat_res = client.post(
            "/api/kb/chat",
            json={"message": "What vector dimension does GoldPan enforce?", "history": []},
        )
        assert chat_res.status_code == 200
        chat_data = chat_res.json()
        assert chat_data["mode"] == "local_extractive"
        assert len(chat_data["citations"]) >= 1

        del_res = client.delete(f"/api/kb/documents/{doc_id}")
        assert del_res.status_code == 200
        docs_after = client.get("/api/kb/documents").json()["documents"]
        assert len(docs_after) == 0
        print("[PASS] FastAPI Endpoints, Path Traversal Guard & Idempotent KB Ingest verified\n")
    finally:
        backend_main.chroma_manager = orig_chroma
        shutil.rmtree(tmp_chroma, ignore_errors=True)


if __name__ == "__main__":
    test_web_extractor()
    test_file_extractor()
    test_liteparse_extractor()
    test_local_onnx_kb()
    test_mcp_server_contracts()
    test_cli_commands()
    test_algorithmic_edge_cases_and_mutations()
    test_gemini_768d_and_rrf_fusion()
    test_idempotent_ingest_and_vietnamese_retrieval()
    test_fastapi_endpoints_and_security()
    print("All 10 GoldPan v2.1 verification & audit suites passed!")
