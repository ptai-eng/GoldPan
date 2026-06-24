<div align="center">
  <img src="https://raw.githubusercontent.com/lucide-icons/lucide/main/icons/sparkles.svg" width="80" height="80" alt="GoldPan Logo">
  <h1>GoldPan AI v1.1.3</h1>
  <p><strong>A powerful, multimodal data extraction and RAG pipeline for AI applications.</strong></p>
  
  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
  [![Next.js 14](https://img.shields.io/badge/Next.js-14-black)](https://nextjs.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)](https://fastapi.tiangolo.com/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

---

**GoldPan AI** is an open-source utility designed to "pan for gold" in your noisy data. It converts complex, unstructured documents (PDFs, Office files, Images, Audio) and web URLs (YouTube, GitHub, Twitter) into clean, AI-ready Markdown. But it doesn't stop there — it features a built-in **Local Knowledge Base** (powered by ChromaDB) and a **RAG Chatbot**, allowing you to seamlessly index your extracted content and chat with your documents using Google's Gemini 2.5 Flash.

## ✨ Key Features (New in v1.1.3)

### 📄 Smart Routing (LiteParse vs MarkItDown)
- **LiteParse (New):** Native, ultra-fast parsing for simple text-based files (`.txt`, `.csv`, `.json`, `.md`, `.xml`). Bypasses heavy engines to give you millisecond extractions formatted cleanly for LLMs (e.g., CSV to Markdown Tables).
- **MarkItDown (Heavy Lifting):** Natively converts PDF, DOCX, XLSX, PPTX, and HTML into Markdown using Microsoft's `MarkItDown`.
- **Multimodal (Vision & Audio):** Seamlessly processes Images (`.jpg`, `.png`) and Audio (`.mp3`, `.wav`) using Google Gemini's Multimodal capabilities to extract text, transcriptions, and EXIF metadata.

### 🌐 Smart Web Routing
- **YouTube Transcripts:** Automatically bypasses HTML scraping and fetches the exact video transcript in milliseconds.
- **GitHub Raw Fallback:** Intelligently converts GitHub blob URLs into RAW format to extract clean source code.
- **JavaScript-Heavy Sites:** Uses a headless `Playwright` browser as a fallback to scrape dynamic content from platforms like Twitter and Facebook.

### 🧠 Local Knowledge Base & RAG Chatbot
- **100% Offline Vector Storage:** Embeds and stores your documents locally using `ChromaDB` and HuggingFace's `all-MiniLM-L6-v2` model. No cloud database required.
- **One-Click Ingest:** Save any extracted document directly to your Knowledge Base with a single click.
- **Interactive Chat:** Ask questions and retrieve answers grounded strictly in your ingested documents using Semantic Search and Gemini 2.5 Flash.

---

## 🏗 Architecture

GoldPan AI is composed of two main services:
1. **Backend (Python & FastAPI):** Handles file parsing, web scraping, chunking, Vector DB (ChromaDB) operations, and LLM orchestration (`google-genai`).
2. **Frontend (Next.js & Tailwind CSS):** Provides a sleek, dark-themed, highly interactive UI to manage your extractions and chat with your Knowledge Base.

---

## 🚀 Getting Started

### Prerequisites
- [Node.js](https://nodejs.org/en/) (v18+)
- [Python](https://www.python.org/downloads/) (3.10+)
- A [Google Gemini API Key](https://aistudio.google.com/) (Required for Vision/Audio extraction and RAG Chat).

### 0. Clone the Repository

To get the latest version (v1.1), open your terminal and run:

```bash
git clone https://github.com/ptai-eng/GoldPan.git
cd GoldPan
```

### 1. Backend Setup

Open a terminal and navigate to the `backend` directory:

```bash
cd backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers (for web scraping)
playwright install
```

Create a `.env` file in the `backend` folder and add your Gemini API Key (Optional, you can also input it directly via the Web UI):
```env
GEMINI_API_KEY=your_api_key_here
```

Start the FastAPI server:
```bash
uvicorn main:app --reload --port 8000
```

### 2. Frontend Setup

Open a new terminal and navigate to the `frontend` directory:

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

Visit `http://localhost:3000` in your browser.

---

## 🛠 Usage Guide

### 1. Extract Content
- In the **Extractor Engine** tab, drag and drop any file (PDF, PPTX, JPG, MP3) or paste a Web Link (YouTube, GitHub, etc.).
- Input your `Gemini API Key` in the bottom left corner if you are processing Media files.
- Click **Start Extraction**. The system will output clean Markdown.

### 2. Build your Knowledge Base
- Once the extraction is successful, click the **Save to KB** button.
- The backend will automatically chunk the text, generate embeddings using a local HuggingFace model, and store it in ChromaDB.

### 3. Chat with your Data
- Switch to the **Knowledge Base** tab using the left sidebar.
- You will see a list of your ingested documents on the left.
- Use the Chat interface on the right to ask questions. The AI will answer based *only* on the facts present in your extracted documents.

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/goldpan/issues).

## 📄 License
This project is licensed under the [MIT License](LICENSE).
