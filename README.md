<div align="center">
  <img src="assets/goldpan-hero-banner.png" alt="GoldPan AI — Privacy-First Multimodal Data Refinery & Local RAG Workspace" width="100%"/>
  <br/>
  <br/>

  <p>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/></a>
    <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/></a>
    <a href="https://nextjs.org/"><img src="https://img.shields.io/badge/Next.js-16-000000?style=for-the-badge&logo=next.js&logoColor=white" alt="Next.js 16"/></a>
    <a href="https://www.trychroma.com/"><img src="https://img.shields.io/badge/ChromaDB_%2B_SQLite_FTS5-Multilingual_E5_ONNX-FACC15?style=for-the-badge&logoColor=black" alt="ChromaDB + SQLite FTS5"/></a>
    <a href="https://modelcontextprotocol.io/"><img src="https://img.shields.io/badge/MCP_Server-Context_Firewall-EAB308?style=for-the-badge&logoColor=black" alt="MCP Server"/></a>
    <a href="#-dual-mode-operation"><img src="https://img.shields.io/badge/AI_Integration-Supported-8B5CF6?style=for-the-badge" alt="AI Integration Supported"/></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge" alt="MIT License"/></a>
  </p>

  <p>
    <a href="#-why-goldpan-ai-v30">Why GoldPan v3.0</a> •
    <a href="#-three-core-pillars-of-v30">v3.0 Core Pillars</a> •
    <a href="#-3-tier-hardware-aware-onnx-acceleration">GPU Tiers</a> •
    <a href="#-goldpan-bench-empirical-evaluation">Benchmarks</a> •
    <a href="#-cli--model-context-protocol-mcp-server">CLI & MCP Server</a> •
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-roadmap-v31--v32">Roadmap</a>
  </p>
</div>

---

## ⚡ Why GoldPan AI v3.0?

Modern RAG pipelines and MCP tool servers fail before embedding even begins: PDFs arrive with broken tables, web pages carry 60% navigation boilerplate or hidden **Indirect Prompt Injection** payloads, and English-centric embedding models fail on Vietnamese legal/technical documents (especially legacy **TCVN3 / VNI-Windows** encodings and unaccented queries).

**GoldPan AI v3.0** is a local-first multimodal data refinery, **Multilingual E5 ONNX + SQLite FTS5** hybrid RAG workspace, **Agent Context Firewall**, and **Model Context Protocol (MCP) server** built to **sift pure signal ("gold") from noisy and untrusted inputs**:

- **Pillar B — Multilingual & Vietnamese First-Class Core:** Upgrades local dense retrieval to **`intfloat/multilingual-e5-small` (384-dim ONNX)** with automatic `"query: "` / `"passage: "` prefixing, backed by a **persistent SQLite FTS5 (`goldpan_fts.db`)** lexical index (`bm25()` with `đ/Đ -> d` diacritic folding and compound bigrams), automatic **TCVN3 (ABC) & VNI-Windows -> Unicode NFC** conversion, and a **Hierarchical Breadcrumb Chunker** (`Phần > Chương > Mục > Điều > Khoản`).
- **Pillar A — Agent Context Firewall (`sanitize` | `audit` | `strict`):** Defends downstream LLM agents against Indirect Prompt Injection (`OWASP LLM01`) across HTML/CSS DOM (`display:none`, `opacity:0`, `font-size:0`, white-on-white, off-screen positioning, HTML comments), PDF character attributes (`<1pt` / white fill via `pdfplumber`), DOCX vanished runs (`<w:vanish/>`), Unicode Tags (`U+E0000`–`U+E007F`), Bidi overrides, fake ChatML/Llama delimiters (`<|im_start|>`, `[INST]`), and Markdown image exfiltration beacons—while preserving **0% False Positive Rate (FPR)** on legitimate OWASP/cybersecurity papers. See [`THREAT_MODEL.md`](THREAT_MODEL.md).
- **Pillar C — Reproducible `GoldPan Bench` & CI Gate:** Built-in dual-track benchmark (`goldpan bench --ci-gate`) evaluating retrieval (`Recall@1`, `Recall@3`, `MRR@5`) and firewall security (`TPR`, `Sanitization Rate`, `Hard-Negative FPR`) with **95% Wilson Score Confidence Intervals**.

---

## 🏛️ Three Core Pillars of v3.0

<div align="center">
  <img src="assets/goldpan-architecture.png" alt="GoldPan AI System Architecture" width="100%"/>
</div>

### 1. Multilingual & Vietnamese Hybrid RAG Engine (`backend/goldpan/kb/`)
- **Multilingual E5 ONNX (`onnx_embedder.py`):** Enforces the asymmetric E5 contract (`passage: ...` on indexing, `query: ...` on search) with L2 normalization in isolated collection `goldpan_kb_e5_v3`.
- **Persistent SQLite FTS5 (`fts_index.py`):** Eliminates `O(N)` in-memory corpus rebuilds by persisting `text_nfc` and `text_folded` in `chroma_data/goldpan_fts.db` (`WAL` mode) and fusing FTS5 `bm25()` with dense vectors via Reciprocal Rank Fusion (`RRF, k=60`).
- **Legacy Vietnamese Encoding Decoder (`vn_encoding.py`):** Detects and converts **VNI-Windows**, **TCVN3 (ABC)**, and decomposed **Unicode NFD** into composed **Unicode NFC** at the token level.
- **Hierarchical Breadcrumb Chunker (`chunker.py`):** Tracks both Markdown (`# H1 > ## H2 > ### H3`) and Vietnamese statutory structure (`Phần > Chương > Mục > Điều > Khoản`), prepending `[Ngữ cảnh: ...]` to every chunk and extracting `so_hieu`, `ngay_ban_hanh`, and `loai_van_ban`.

### 2. Agent Context Firewall (`backend/goldpan/firewall/context_firewall.py`)
- **Deterministic Structural & Steganographic Stripping (`sanitize` default):** Strips hidden DOM nodes *before* Trafilatura text extraction, filters `<1pt`/white PDF characters and DOCX `<w:vanish/>` runs, removes Unicode Tag (`U+E0000`–`U+E007F`) and Bidi/Zero-Width characters, neutralizes `<|im_start|>`/`[INST]` delimiters, and blocks Markdown image exfiltration beacons.
- **Context-Aware Semantic Risk Scoring:** Flags imperative override payloads (`risk_score` `0.0–1.0`, `signals`) in English and Vietnamese while distinguishing academic/educational security discussions (Hard Negatives) so visible prose is never blindly corrupted.
- **MCP Data Spotlighting:** Optional `<untrusted_document_content source="..." firewall_mode="..." risk_score="...">` wrapping + `_meta.provenance.firewall` telemetry.

### 3. 3-Tier Hardware-Aware ONNX Acceleration (`goldpan doctor`)

| Tier | Target Hardware | Package / Extra | ONNX Provider & Config | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1 (Default)** | Any x86_64 / ARM64 CPU | `pip install goldpan` | `CPUExecutionProvider` (`INT8`, `batch_size=32`) | **CI-Verified** (Linux & Windows) |
| **Tier 2 (Windows GPU)** | **AMD Radeon RX 6600 (RDNA 2)**, **RX 5000 (RDNA 1)**, RX 7000, Intel Arc, NVIDIA DX12 | `pip install "goldpan[directml]"` | `DmlExecutionProvider` (`FP32`, bounded `batch_size=12` + warmup fallback) | **Hardware-Verified** on Windows + AMD RX 6600 (8GB VRAM) |
| **Tier 3 (Community Extras)** | NVIDIA CUDA / Intel OpenVINO / Apple Silicon | `goldpan[cuda]` / `goldpan[openvino]` / `goldpan[coreml]` | Respective provider with automatic fallback to `CPUExecutionProvider` | **Community-Tested** (CI promotion in `v3.1`) |

---

## 📊 `GoldPan Bench` Empirical Evaluation

Reproduce at any time with `goldpan bench --ci-gate` (dataset SHA-256 hashes verified automatically; see [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md)):

### Track B: Multilingual & Vietnamese Hybrid Retrieval (`16` Held-Out Queries across `4` Categories)

| System | Recall@1 (95% Wilson CI) | Recall@3 (95% Wilson CI) | MRR@5 | Avg Latency |
| :--- | :---: | :---: | :---: | :---: |
| **BM25-only (Surface Lexical)** | `50.0%` (`[28.0%, 72.0%]`) | `62.5%` (`[38.6%, 81.5%]`) | `0.5677` | `0.06 ms` |
| **v2.1 (MiniLM-L6 + In-Memory BM25 RRF)** | `68.8%` (`[44.4%, 85.8%]`) | `81.2%` (`[57.0%, 93.4%]`) | `0.7417` | `3.56 ms` |
| **v3.0 (E5-small ONNX + SQLite FTS5 + Breadcrumb + VNI/TCVN3)** | **`87.5%`** (`[64.0%, 96.5%]`) | **`93.8%`** (`[71.7%, 98.9%]`) | **`0.8958`** | `5.62 ms` |

- **`vi_accented` / `vi_unaccented` / `vi_paraphrase`:** `100.0%` Recall@1 & Recall@3 (`MRR@5 = 1.0000`), including legacy VNI-Windows (`Boä Luaät Lao Ñoäng`) and TCVN3 (`Thông tư 80/2021/TT-BTC`) documents.
- **`cross_lingual_en_vi`:** `75.0%` Recall@3 (`[30.1%, 95.4%]` 95% Wilson CI).

### Track A: Agent Context Firewall (`13` Multi-Vector Attacks + `4` Benign Hard Negatives)

| Metric | Score | 95% Wilson Confidence Interval | Target Gate |
| :--- | :---: | :---: | :---: |
| **Attack Detection Rate (TPR)** | **`100.0%`** | `[77.2%, 100.0%]` | `>= 90.0%` |
| **Payload Sanitization Rate** | **`100.0%`** | `[77.2%, 100.0%]` | `>= 90.0%` |
| **Hard-Negative False Positive Rate (FPR)** | **`0.0%`** | `[0.0%, 49.0%]` | `<= 5.0%` (0% ideal) |
| **Benign Content Preservation Rate** | **`100.0%`** | — | `100.0%` |
| **Firewall Overhead Latency** | **`0.21 ms/doc`** | — | `< 25.0 ms/doc` |

---

## 🧰 CLI & Model Context Protocol (MCP) Server

### 1. Command-Line Interface (`goldpan`)

```powershell
# Install GoldPan v3.0 (add [directml] on Windows with AMD RX 6600 / RX 5000 / DX12 GPUs)
pip install -e .
# pip install -e ".[directml]"

# 1. Diagnose hardware execution provider, warmup latency, SQLite FTS5 & v2->v3 migration status
goldpan doctor

# 2. Extract clean Markdown protected by the Agent Context Firewall (sanitize | audit | strict)
goldpan extract https://example.com --firewall-mode sanitize --spotlight

# 3. Ingest documents into the v3 Multilingual E5 + SQLite FTS5 Knowledge Base
goldpan ingest ./luat_so_huu_tri_tue.pdf --local

# 4. Hybrid RRF search (supports accented, unaccented, paraphrased & cross-lingual queries)
goldpan search "dieu 15 luat so huu tri tue khong bao ho quyen tac gia" --top-k 4

# 5. Re-index / migrate existing KB/*.md or v2 chunks into v3 E5 + SQLite FTS5
goldpan reindex

# 6. Audit all indexed Knowledge Base chunks for prompt injection signatures
goldpan audit

# 7. Run the reproducible Dual-Track Benchmark & CI Regression Gate
goldpan bench --ci-gate --output-md BENCHMARK_REPORT.md
```

### 2. MCP Server (`goldpan-mcp`) for AI Coding Agents

Register `goldpan-mcp` in **Claude Desktop**, **Claude Code**, **Cursor**, or **Antigravity**:

```json
{
  "mcpServers": {
    "goldpan": {
      "command": "C:/path/to/GoldPan/venv/Scripts/goldpan-mcp.exe",
      "args": [],
      "env": {
        "GEMINI_API_KEY": ""
      }
    }
  }
}
```

| MCP Tool | Arguments | Description |
| :--- | :--- | :--- |
| `goldpan_extract` | `source`, `detail="summary"`, `max_chars=4000`, `ingest_to_kb=false`, `firewall_mode="sanitize"`, `spotlight=false` | Extracts clean Markdown protected by the Agent Context Firewall with `_meta.provenance.firewall` telemetry. |
| `goldpan_kb_ingest` | `markdown`, `title`, `source`, `prefer_local=true` | Chunks with `HierarchicalBreadcrumbChunker`, indexes into `goldpan_kb_e5_v3` + SQLite FTS5, and saves `KB/*.md`. |
| `goldpan_kb_search` | `query`, `n_results=4`, `detail="summary"`, `spotlight=false` | Runs Hybrid RRF search across Multilingual E5 (384d), SQLite FTS5 BM25, and Gemini (768d). |
| `goldpan_kb_audit` | *(none)* | Scans all indexed chunks in the Knowledge Base using `AgentContextFirewall(mode="audit")`. |
| `goldpan_doctor` | *(none)* | Returns hardware execution provider tier, warmup latency, FTS5 status, and v2->v3 migration health. |
| `goldpan_kb_list` | *(none)* | Lists all indexed documents, titles, sources, and chunk counts. |
| `goldpan_kb_delete` | `document_id` | Deletes a document's vectors and removes its physical `KB/*.md` file. |

---

## 🖥️ Workspace Tour

Built with **Next.js 16**, **React 19**, **Tailwind CSS v4**, and **Framer Motion** in a high-contrast **Dark Stone & Gold (`#0C0A09` / `#FACC15`)** theme.

![GoldPan Extractor Engine](screenshot1.png)

![GoldPan Knowledge Base Chat](screenshot.png)

---

## 🚀 Quick Start

```powershell
git clone https://github.com/ptai-eng/GoldPan.git
cd GoldPan

python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# Run all 14 verification, security & benchmark suites
.\venv\Scripts\python.exe run_tests.py
.\venv\Scripts\python.exe backend/tests/test_api.py
```

---

## 🗺️ Roadmap (`v3.1` & `v3.2`)

- **`v3.1` (Deferred Hardware CI & Offline Vietnamese OCR):**
  - `goldpan[ocr]`: Offline Vietnamese scanned PDF/image OCR (`VietOCR` + `PaddleOCR-VL` / `Tesseract vie`) when `pdfplumber` detects `< 30` chars/page.
  - Official GitHub Actions CI matrix runners for Apple Silicon (`CoreMLExecutionProvider`) and Intel (`OpenVINOExecutionProvider`).
- **`v3.2` (Autonomous Verification & MCP-Native Engineering):**
  - **Playwright MCP + Chrome DevTools MCP:** Automated E2E verification and runtime network/console profiling of the Next.js 16 RAG & Firewall UI (`http://localhost:3000` <-> `http://localhost:8000`) plus adversarial SPA DOM extraction tests.
  - **Context7 MCP:** Version-pinned syntax verification for Next.js 16, React 19, Tailwind CSS v4, FastAPI, ChromaDB, and ONNX Runtime.
  - **GitHub MCP Server:** Automated issue triage, branch creation, PR benchmark comment gating (`BENCHMARK_REPORT.md`), and release tagging.

---

## 📜 License

Released under the [MIT License](LICENSE). Crafted by **[Hoang Kim Phat Tai (ptai-eng)](https://github.com/ptai-eng)**.
