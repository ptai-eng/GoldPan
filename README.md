# ⛏️ GoldPan (Kẻ Đào Vàng)

**GoldPan** is a high-performance Data Cleansing pipeline designed specifically for RAG (Retrieval-Augmented Generation) and AI Agents. It extracts raw data from various sources (Web, PDF, DOCX) and transforms it into ultra-clean, AI-Friendly Markdown by aggressively filtering out UI noise, boilerplate, and disjointed text.

## 🚀 Core Features

- **Multi-Format Extraction:** Seamlessly reads Web URLs (with SPA/JS support via Playwright), PDFs, and Office documents.
- **Two-Pass Filtering System:**
  - **Heuristic Filter:** Rapidly strips out headers, footers, cookie banners, and ads.
  - **AI Semantic Router:** Intelligently drops orphan paragraphs and non-contextual noise.
- **Ultra-Fast PDF Parsing:** Powered by `liteparse` to ensure lightning-fast extraction while preserving spatial bounding boxes.
- **AI-Friendly Output:** Outputs a single, contiguous Markdown file with clean H1/H2/H3 headers—perfect for LLM consumption.

## 📦 Installation

```bash
git clone https://github.com/ptai-eng/GoldPan.git
cd GoldPan
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
```

## 💻 Usage

### Web UI (Gradio)
To launch the interactive Web UI:
```bash
python app.py
```
Then open `http://127.0.0.1:7860` in your browser.

## 🙏 Acknowledgements & Credits

This project stands on the shoulders of giants. We proudly integrate and build upon the following incredible open-source projects:

- **[LiteParse](https://github.com/run-llama/liteparse) (by LlamaIndex):** Powering our ultra-fast, spatial-aware PDF extraction engine.
- **[MarkItDown](https://github.com/microsoft/markitdown) (by Microsoft):** Providing robust parsing for Office formats (.docx, .xlsx, .pptx).
- **[Trafilatura](https://github.com/adbar/trafilatura):** Core heuristic engine for web content extraction.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
