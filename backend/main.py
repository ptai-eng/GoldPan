import sys
import os
import re
import uuid
import shutil
from typing import Optional

# Fix path to allow importing from backend/goldpan
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from goldpan.core.pipeline import GoldPanPipeline
from goldpan.extractors.markitdown_wrapper import MarkItDownExtractor
from goldpan.extractors.web_extractor import HybridWebExtractor
from goldpan.extractors.liteparse_wrapper import LiteParseExtractor
from goldpan.filters.heuristic_filter import HeuristicFilter
from goldpan.filters.semantic_router import SemanticRouterFilter
from goldpan.formatters.markdown_formatter import AIFriendlyMarkdownFormatter
from goldpan.kb.chroma_manager import ChromaManager, compute_document_id
from google import genai

app = FastAPI(
    title="GoldPan AI API",
    description="Privacy-First Multimodal Data Extraction & Dual-Mode Local RAG Pipeline",
    version="2.1.0",
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)

MAX_OFFLINE_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB

# Initialize ChromaManager globally (supports both 100% offline Local ONNX and Gemini Cloud embeddings)
chroma_manager = ChromaManager()


class UrlRequest(BaseModel):
    url: str


class IngestRequest(BaseModel):
    markdown: str
    metadata: dict
    prefer_local: bool = False


class ChatRequest(BaseModel):
    message: str
    history: list = []


def get_pipeline(source: str, is_file: bool = False, api_key: Optional[str] = None) -> GoldPanPipeline:
    if not is_file:
        extractor = HybridWebExtractor(use_playwright_fallback=True)
    else:
        ext = os.path.splitext(source)[1].lower()
        if ext in [".txt", ".csv", ".json", ".md", ".xml"]:
            extractor = LiteParseExtractor()
        else:
            extractor = MarkItDownExtractor(api_key=api_key)

    pipeline = GoldPanPipeline(extractor=extractor)
    pipeline.add_filter(HeuristicFilter())
    pipeline.add_filter(SemanticRouterFilter(api_key=api_key))
    pipeline.set_formatter(AIFriendlyMarkdownFormatter())
    return pipeline


@app.post("/api/kb/ingest")
def ingest_document(req: IngestRequest, x_gemini_api_key: Optional[str] = Header(None)):
    """
    Ingests cleaned Markdown into the Knowledge Base.
    Works 100% offline using Local ONNX embeddings (`all-MiniLM-L6-v2`) when no API key is passed,
    or uses Gemini Cloud embeddings when `X-Gemini-Api-Key` is provided.
    """
    if not req.markdown or not req.markdown.strip():
        raise HTTPException(status_code=400, detail="Markdown content cannot be empty.")

    try:
        chroma_manager.set_api_key(x_gemini_api_key)
        chroma_manager.prefer_local = req.prefer_local

        meta = dict(req.metadata or {})
        source_name = (
            meta.get("file_path")
            or meta.get("source_url")
            or meta.get("title")
            or "document"
        )
        doc_id = compute_document_id(req.markdown, str(source_name))
        meta["document_id"] = doc_id

        # Save physical Markdown mirror in KB/ (overwriting any previous mirror for this deterministic doc_id)
        kb_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "KB")
        os.makedirs(kb_dir, exist_ok=True)

        safe_stem = re.sub(r'[\\/*?:"<>|]', "", os.path.basename(str(source_name)))[:50] or "document"
        safe_filename = f"{safe_stem}_{doc_id[:8]}.md"
        physical_path = os.path.join(kb_dir, safe_filename)

        with open(physical_path, "w", encoding="utf-8") as f:
            f.write(req.markdown)

        meta["physical_file"] = physical_path

        embedding_mode = chroma_manager.ingest_document(
            document_id=doc_id,
            text=req.markdown,
            metadata=meta,
        )
        return {
            "status": "success",
            "document_id": doc_id,
            "physical_file": physical_path,
            "embedding_mode": embedding_mode,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/kb/documents/{document_id}")
def delete_document(document_id: str):
    try:
        all_docs = chroma_manager.get_all_documents()
        for doc in all_docs:
            if doc.get("id") == document_id and doc.get("physical_file"):
                p = doc["physical_file"]
                if os.path.exists(p):
                    os.remove(p)
                break
        chroma_manager.delete_document(document_id)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/kb/documents")
def get_documents():
    try:
        docs = chroma_manager.get_all_documents()
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/kb/chat")
def chat_with_kb(req: ChatRequest, x_gemini_api_key: Optional[str] = Header(None)):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        chroma_manager.set_api_key(x_gemini_api_key)

        # 1. Semantic Search in ChromaDB with Citations
        hits = chroma_manager.search_with_citations(req.message, n_results=4)
        relevant_chunks = [h["text"] for h in hits]
        citations = [
            {
                "title": h["title"],
                "source": h["source"],
                "embedding_mode": h["embedding_mode"],
            }
            for h in hits
        ]

        if not hits:
            return {
                "response": "No documents are currently indexed in the Knowledge Base, or no relevant passages matched your query.",
                "context_used": [],
                "citations": [],
                "mode": "empty_kb",
            }

        # 2. Offline Extractive Fallback when no Gemini API Key is configured in request header
        effective_key = x_gemini_api_key
        if not effective_key:
            passages_md = "\n\n".join(
                [f"**[{i+1}] {h['title']}** (`{h['source']}`):\n> {h['text'][:600]}" for i, h in enumerate(hits)]
            )
            offline_reply = (
                "### 🔒 Local Offline Retrieval (ONNX `all-MiniLM-L6-v2`)\n\n"
                "Top matching passages retrieved strictly from your local Knowledge Base:\n\n"
                f"{passages_md}\n\n"
                "*Tip: Add a Gemini API Key in Settings to enable generative multi-turn synthesis over these passages.*"
            )
            return {
                "response": offline_reply,
                "context_used": relevant_chunks,
                "citations": citations,
                "mode": "local_extractive",
            }

        # 3. Cloud Generative RAG with Multi-Turn Conversation History
        context = "\n\n---\n\n".join(
            [f"[Source {i+1}: {h['title']}]\n{h['text']}" for i, h in enumerate(hits)]
        )

        history_lines = []
        for turn in (req.history or [])[-6:]:
            role = "User" if turn.get("role") == "user" else "Assistant"
            content_str = str(turn.get("content", "")).strip()
            if content_str:
                history_lines.append(f"{role}: {content_str}")
        history_block = "\n".join(history_lines)

        system_prompt = (
            "You are the GoldPan AI Knowledge Base Assistant. "
            "Answer the user's question strictly using the provided CONTEXT passages. "
            "Cite the source title when referencing specific facts. "
            "If the context does not contain the answer, state clearly: "
            "'I don't know based on the provided documents.'\n\n"
            f"CONTEXT:\n{context}"
        )

        full_prompt = system_prompt
        if history_block:
            full_prompt += f"\n\nCONVERSATION HISTORY:\n{history_block}"
        full_prompt += f"\n\nUser Question: {req.message}"

        client = genai.Client(api_key=effective_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
        )

        return {
            "response": response.text,
            "context_used": relevant_chunks,
            "citations": citations,
            "mode": "gemini_rag",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/extract/url")
def extract_url(req: UrlRequest, x_gemini_api_key: Optional[str] = Header(None)):
    if not req.url or not req.url.strip():
        raise HTTPException(status_code=400, detail="URL is required")

    pipeline = get_pipeline(req.url, is_file=False, api_key=x_gemini_api_key)
    try:
        result = pipeline.process(req.url)
        return {
            "markdown": result.text,
            "metadata": {
                "file_path": getattr(result.doc_info, "file_path", None),
                "source_url": getattr(result.doc_info, "source_url", None),
                "title": result.doc_info.title,
                "stats": result.stats,
            },
        }
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            raise HTTPException(
                status_code=429,
                detail="API miễn phí đã vượt quá hạn mức. Vui lòng nhập API trả phí của bạn vào mục Settings (Cài Đặt) ở giao diện chính để tiếp tục.",
            )
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/api/extract/file")
def extract_file(file: UploadFile = File(...), x_gemini_api_key: Optional[str] = Header(None)):
    if not x_gemini_api_key and file.size and file.size > MAX_OFFLINE_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Dung lượng file vượt quá giới hạn 5MB ở chế độ mặc định không khóa API. Vui lòng cấu hình Gemini API Key trong mục Settings để xử lý file lớn.",
        )

    # Sanitize filename against path traversal and prefix with UUID to prevent concurrent collisions
    raw_filename = os.path.basename(file.filename or "upload.bin")
    safe_basename = re.sub(r'[\\/*?:"<>|\x00]', "_", raw_filename) or "upload.bin"
    unique_prefix = uuid.uuid4().hex[:8]
    file_path = os.path.join(TEMP_DIR, f"{unique_prefix}_{safe_basename}")

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Enforce 5MB limit on actual bytes written even if Content-Length was omitted
        if not x_gemini_api_key and os.path.getsize(file_path) > MAX_OFFLINE_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="Dung lượng file vượt quá giới hạn 5MB ở chế độ mặc định không khóa API. Vui lòng cấu hình Gemini API Key trong mục Settings để xử lý file lớn.",
            )

        pipeline = get_pipeline(file_path, is_file=True, api_key=x_gemini_api_key)
        result = pipeline.process(file_path)

        # Restore original clean filename in title if formatter/extractor used prefixed temp name
        clean_title = result.doc_info.title
        if clean_title == os.path.basename(file_path):
            clean_title = safe_basename
            result.text = result.text.replace(os.path.basename(file_path), safe_basename)

        return {
            "markdown": result.text,
            "metadata": {
                "file_path": safe_basename,
                "source_url": getattr(result.doc_info, "source_url", None),
                "title": clean_title,
                "stats": result.stats,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            raise HTTPException(
                status_code=429,
                detail="API miễn phí đã vượt quá hạn mức. Vui lòng nhập API trả phí của bạn vào mục Settings (Cài Đặt) ở giao diện chính để tiếp tục.",
            )
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "2.1.0"}
