import sys
import os
import shutil

# Fix path to allow importing from backend/goldpan
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from goldpan.core.pipeline import GoldPanPipeline
from goldpan.extractors.markitdown_wrapper import MarkItDownExtractor
from goldpan.extractors.web_extractor import HybridWebExtractor
from goldpan.extractors.liteparse_wrapper import LiteParseExtractor
from goldpan.filters.heuristic_filter import HeuristicFilter
from goldpan.filters.semantic_router import SemanticRouterFilter
from goldpan.formatters.markdown_formatter import AIFriendlyMarkdownFormatter

app = FastAPI(title="GoldPan API", description="RAG Data Cleansing API")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "./temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

class UrlRequest(BaseModel):
    url: str

def get_pipeline(source: str, is_file: bool = False, api_key: Optional[str] = None):
    if not is_file:
        extractor = HybridWebExtractor(use_playwright_fallback=True)
    else:
        ext = os.path.splitext(source)[1].lower()
        if ext in ['.txt', '.csv', '.json', '.md', '.xml']:
            extractor = LiteParseExtractor()
        else:
            extractor = MarkItDownExtractor(api_key=api_key)
            
    pipeline = GoldPanPipeline(extractor=extractor)
    pipeline.add_filter(HeuristicFilter())
    pipeline.add_filter(SemanticRouterFilter(api_key=api_key))
    pipeline.set_formatter(AIFriendlyMarkdownFormatter())
    return pipeline

from goldpan.kb.chroma_manager import ChromaManager
from google import genai
import uuid

# Initialize ChromaManager globally
chroma_manager = ChromaManager()

class IngestRequest(BaseModel):
    markdown: str
    metadata: dict

class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.post("/api/kb/ingest")
def ingest_document(req: IngestRequest, x_gemini_api_key: Optional[str] = Header(None)):
    if not x_gemini_api_key:
        raise HTTPException(status_code=400, detail="Missing API Key for Ingest. Please configure it in Settings.")
    
    try:
        chroma_manager.set_api_key(x_gemini_api_key)
        doc_id = str(uuid.uuid4())
        # Attach doc_id to metadata so we can retrieve it
        req.metadata['document_id'] = doc_id
        chroma_manager.ingest_document(document_id=doc_id, text=req.markdown, metadata=req.metadata)
        return {"status": "success", "document_id": doc_id}
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
    if not x_gemini_api_key:
        raise HTTPException(status_code=400, detail="Missing API Key for Chat. Please configure it in Settings.")
        
    try:
        chroma_manager.set_api_key(x_gemini_api_key)
        # 1. Semantic Search in ChromaDB
        relevant_chunks = chroma_manager.search(req.message, n_results=3)
        context = "\n\n---\n\n".join(relevant_chunks)
        
        # 2. Build Prompt for Gemini
        system_prompt = (
            "You are a helpful assistant for the GoldPan AI system. "
            "Your task is to answer the user's question based strictly on the provided context. "
            "If the context does not contain the answer, say 'I don't know based on the provided documents.'\n\n"
            f"CONTEXT:\n{context}"
        )
        
        # 3. Call Gemini
        client = genai.Client(api_key=x_gemini_api_key)
        
        # We can format history if needed, but for simplicity we'll just pass a combined prompt
        full_prompt = f"{system_prompt}\n\nUser Question: {req.message}"
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt
        )
        
        return {"response": response.text, "context_used": relevant_chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/extract/url")
def extract_url(req: UrlRequest, x_gemini_api_key: Optional[str] = Header(None)):
    if not req.url:
        raise HTTPException(status_code=400, detail="URL is required")
        
    pipeline = get_pipeline(req.url, is_file=False, api_key=x_gemini_api_key)
    try:
        result = pipeline.process(req.url)
        # return dict, Pydantic dict() is deprecated in v2, use model_dump() if available
        # But we'll just return raw fields to be safe
        return {"markdown": result.text, "metadata": {"file_path": result.doc_info.file_path, "title": result.doc_info.title}}
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            raise HTTPException(status_code=429, detail="API miễn phí đã vượt quá hạn mức. Vui lòng nhập API trả phí của bạn vào mục Settings (Cài Đặt) ở giao diện chính để tiếp tục.")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/extract/file")
def extract_file(file: UploadFile = File(...), x_gemini_api_key: Optional[str] = Header(None)):
    if not x_gemini_api_key and file.size and file.size > 1 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Dung lượng file vượt quá giới hạn 1MB của API miễn phí. Vui lòng nhập API trả phí của bạn vào mục Settings (Cài Đặt) ở giao diện chính để xử lý file nặng này.")

    file_path = os.path.join(TEMP_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        pipeline = get_pipeline(file_path, is_file=True, api_key=x_gemini_api_key)
        result = pipeline.process(file_path)
        
        return {"markdown": result.text, "metadata": {"file_path": result.doc_info.file_path, "title": result.doc_info.title}}
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            raise HTTPException(status_code=429, detail="API miễn phí đã vượt quá hạn mức. Vui lòng nhập API trả phí của bạn vào mục Settings (Cài Đặt) ở giao diện chính để tiếp tục.")
        raise HTTPException(status_code=500, detail=error_msg)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/health")
def health_check():
    return {"status": "ok"}
