# GoldPan AI v3.2 — Dual-Track Benchmark Report (`GoldPan Bench`)

## Hardware & Execution Telemetry
- **Active Execution Provider:** `CPUExecutionProvider` (Tier 1 (Default & CI-Tested CPU SIMD))
- **Model Precision & Batch Size:** `INT8` (`batch_size=32`)
- **Track B Dataset SHA-256:** `91c34534d0dec74a...` (10 docs, 16 queries)
- **Track A Dataset SHA-256:** `63223031148ecd6d...` (13 attacks, 4 hard negatives)

---

## Track B: Multilingual & Vietnamese Hybrid Retrieval

| System | Recall@1 (95% Wilson CI) | Recall@3 (95% Wilson CI) | MRR@5 | Avg Latency (ms/query) |
| :--- | :---: | :---: | :---: | :---: |
| **BM25-only (Surface Lexical)** | `50.0%` (`[28.0%, 72.0%]`) | `62.5%` (`[38.6%, 81.5%]`) | `0.5677` | `0.07 ms` |
| **v2.1 (MiniLM-L6 + In-Memory BM25 RRF)** | `62.5%` (`[38.6%, 81.5%]`) | `81.2%` (`[57.0%, 93.4%]`) | `0.7104` | `2.92 ms` |
| **v3.0 (E5-small ONNX + SQLite FTS5 + Breadcrumb + VNI/TCVN3)** | `87.5%` (`[64.0%, 96.5%]`) | `93.8%` (`[71.7%, 98.9%]`) | `0.8958` | `5.33 ms` |

### Track B Breakdown by Query Category (`v3.0 E5-small + SQLite FTS5`)

| Category | Queries | Recall@1 | Recall@3 (95% CI) | MRR@5 |
| :--- | :---: | :---: | :---: | :---: |
| `vi_accented` | 4 | `100.0%` | `100.0%` (`[51.0%, 100.0%]`) | `1.0000` |
| `vi_unaccented` | 4 | `100.0%` | `100.0%` (`[51.0%, 100.0%]`) | `1.0000` |
| `vi_paraphrase` | 4 | `100.0%` | `100.0%` (`[51.0%, 100.0%]`) | `1.0000` |
| `cross_lingual_en_vi` | 4 | `50.0%` | `75.0%` (`[30.1%, 95.4%]`) | `0.5833` |

---

## Track A: Agent Context Firewall (Indirect Prompt Injection Defense)

| Metric | Score | 95% Wilson Confidence Interval | Target Gate |
| :--- | :---: | :---: | :---: |
| **Attack Detection Rate (TPR)** | `100.0%` | `[77.2%, 100.0%]` | `>= 90.0%` |
| **Payload Sanitization Rate** | `100.0%` | `[77.2%, 100.0%]` | `>= 90.0%` |
| **Hard-Negative False Positive Rate (FPR)** | `0.0%` | `[0.0%, 49.0%]` | `<= 5.0%` (0% ideal) |
| **Benign Content Preservation Rate** | `100.0%` | — | `100.0%` |
| **Firewall Overhead Latency** | `0.21 ms/doc` | — | `< 25.0 ms/doc` |
