import os
import re
import time
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from goldpan.filters.semantic_router import _split_preserving_fences


def _sanitize_metadata(metadata: Optional[dict], document_id: str) -> dict:
    """
    Ensures all metadata values passed to ChromaDB are non-null primitives
    (str, int, float, bool) to prevent ChromaDB Rust binding TypeError/ValueError.
    """
    clean: dict[str, str | int | float | bool] = {"document_id": document_id}
    for k, v in (metadata or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[str(k)] = v
    return clean


class ChromaManager:
    """
    Dual-Mode Vector Knowledge Base Manager:
    - Local Mode (100% Offline Default): Uses ChromaDB's local ONNX `all-MiniLM-L6-v2` (384-dim)
      in collection `goldpan_kb_local`. Requires zero API keys.
    - Gemini Cloud Mode (Optional): Uses `gemini-embedding-001` with Exponential Backoff retry
      against HTTP 429 rate limits in collection `goldpan_kb_gemini`, automatically falling back
      to Local ONNX if no API key is configured or quota is exhausted.
    """

    def __init__(self, persist_directory: Optional[str] = None, prefer_local: bool = False):
        if persist_directory is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            persist_directory = os.path.join(base_dir, "chroma_data")
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        self.local_ef = DefaultEmbeddingFunction()
        self.local_collection = self.client.get_or_create_collection(
            name="goldpan_kb_local",
            embedding_function=self.local_ef,
        )
        self.gemini_collection = self.client.get_or_create_collection(
            name="goldpan_kb_gemini"
        )
        # Keep reference to legacy collection if it exists on disk
        try:
            self.legacy_collection = self.client.get_collection(name="goldpan_kb")
        except Exception:
            self.legacy_collection = None

        self.api_key: Optional[str] = None
        self.prefer_local: bool = prefer_local

    @property
    def collection(self):
        """Backward-compatible active collection accessor."""
        if self.api_key and not self.prefer_local:
            return self.gemini_collection
        return self.local_collection

    def set_api_key(self, api_key: Optional[str]):
        self.api_key = api_key.strip() if api_key else None

    def _get_gemini_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise ValueError("Gemini API Key is required for Cloud Embeddings.")

        import requests

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={self.api_key}"

        def embed_single(text: str, index: int):
            payload = {
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": text}]},
            }
            max_retries = 5
            backoff = 1.5
            for attempt in range(max_retries):
                try:
                    res = requests.post(url, json=payload, timeout=20)
                    if res.ok:
                        data = res.json()
                        return index, data["embedding"]["values"]

                    error_data = res.text
                    if res.status_code == 429 or "RESOURCE_EXHAUSTED" in error_data:
                        if attempt < max_retries - 1:
                            time.sleep(backoff)
                            backoff *= 2
                            continue
                    raise ValueError(f"Gemini API Error ({res.status_code}): {error_data}")
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(backoff)
                    backoff *= 2

            raise ValueError("Failed to obtain Gemini embedding after retries.")

        embeddings = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_idx = {executor.submit(embed_single, text, i): i for i, text in enumerate(texts)}
            for future in as_completed(future_to_idx):
                idx, emb = future.result()
                embeddings[idx] = emb

        return embeddings

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
        """
        Markdown-aware semantic chunking:
        Splits along Markdown section headers and paragraph boundaries while keeping
        fenced code blocks and tables intact, falling back to sliding window only for
        oversized blocks.
        """
        cleaned = text.strip()
        if not cleaned:
            return []

        blocks = _split_preserving_fences(cleaned)
        chunks: list[str] = []
        current_chunk = ""

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            if len(current_chunk) + len(block) + 2 <= chunk_size:
                current_chunk = f"{current_chunk}\n\n{block}".strip() if current_chunk else block
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    tail = current_chunk[-overlap:] if overlap > 0 and len(current_chunk) > overlap else ""
                    current_chunk = (
                        f"{tail}\n\n{block}".strip()
                        if tail and len(tail) + len(block) + 2 <= chunk_size
                        else ""
                    )

                if not current_chunk:
                    if len(block) <= chunk_size:
                        current_chunk = block
                    else:
                        start = 0
                        step = max(1, chunk_size - overlap)
                        while start < len(block):
                            chunks.append(block[start : start + chunk_size])
                            start += step

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def ingest_document(self, document_id: str, text: str, metadata: Optional[dict] = None) -> str:
        """
        Splits Markdown into semantic chunks and stores them in ChromaDB.
        Returns the embedding mode used ('local_onnx' or 'gemini_api').
        """
        if not text or not text.strip():
            return "none"

        chunks = self._chunk_text(text)
        if not chunks:
            return "none"

        # Remove any stale chunks if re-ingesting the same document_id
        self.delete_document(document_id)

        ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
        base_meta = _sanitize_metadata(metadata, document_id)

        # Try Gemini Cloud Embedding when API Key is provided and prefer_local is False
        if self.api_key and not self.prefer_local:
            try:
                embeddings = self._get_gemini_embeddings(chunks)
                meta_with_mode = [{**base_meta, "embedding_mode": "gemini_api"} for _ in chunks]
                self.gemini_collection.upsert(
                    documents=chunks,
                    ids=ids,
                    metadatas=meta_with_mode,
                    embeddings=embeddings,
                )
                return "gemini_api"
            except Exception as e:
                print(f"[ChromaManager] Gemini embedding unavailable ({e}), falling back to Local ONNX.")

        # Default / Fallback: 100% Local Offline ONNX Embedding (all-MiniLM-L6-v2)
        meta_with_mode = [{**base_meta, "embedding_mode": "local_onnx"} for _ in chunks]
        self.local_collection.upsert(
            documents=chunks,
            ids=ids,
            metadatas=meta_with_mode,
        )
        return "local_onnx"

    def search_with_citations(self, query: str, n_results: int = 3) -> list[dict]:
        """
        Searches across active Local ONNX and Gemini collections and returns ranked
        chunks with source metadata and distance scores.
        """
        if not query or not query.strip():
            return []

        candidates: list[dict] = []

        # 1. Query Local ONNX collection (always available offline)
        try:
            local_cnt = self.local_collection.count()
            if local_cnt > 0:
                res = self.local_collection.query(
                    query_texts=[query],
                    n_results=min(max(1, n_results), local_cnt),
                    include=["documents", "metadatas", "distances"],
                )
                docs = (res.get("documents") or [[]])[0]
                metas = (res.get("metadatas") or [[]])[0]
                dists = (res.get("distances") or [[]])[0]
                for doc, meta, dist in zip(docs, metas, dists):
                    candidates.append(
                        {
                            "document_id": (meta or {}).get("document_id"),
                            "text": doc,
                            "title": (meta or {}).get("title", "Unknown Document"),
                            "source": (meta or {}).get("source_url") or (meta or {}).get("file_path") or "Local KB",
                            "embedding_mode": (meta or {}).get("embedding_mode", "local_onnx"),
                            "distance": float(dist) if dist is not None else 1.0,
                        }
                    )
        except Exception as e:
            print(f"[ChromaManager] Local search warning: {e}")

        # 2. Query Gemini collection (and legacy collection) if API Key is available
        if self.api_key:
            for col in (self.gemini_collection, self.legacy_collection):
                if col is None:
                    continue
                try:
                    cnt = col.count()
                    if cnt > 0:
                        q_emb = self._get_gemini_embeddings([query])
                        res = col.query(
                            query_embeddings=q_emb,
                            n_results=min(max(1, n_results), cnt),
                            include=["documents", "metadatas", "distances"],
                        )
                        docs = (res.get("documents") or [[]])[0]
                        metas = (res.get("metadatas") or [[]])[0]
                        dists = (res.get("distances") or [[]])[0]
                        for doc, meta, dist in zip(docs, metas, dists):
                            candidates.append(
                                {
                                    "document_id": (meta or {}).get("document_id"),
                                    "text": doc,
                                    "title": (meta or {}).get("title", "Unknown Document"),
                                    "source": (meta or {}).get("source_url") or (meta or {}).get("file_path") or "Local KB",
                                    "embedding_mode": (meta or {}).get("embedding_mode", "gemini_api"),
                                    "distance": float(dist) if dist is not None else 1.0,
                                }
                            )
                except Exception as e:
                    print(f"[ChromaManager] Cloud collection search warning: {e}")

        candidates.sort(key=lambda x: x["distance"])
        return candidates[:n_results]

    def search(self, query: str, n_results: int = 3) -> list[str]:
        """Backward-compatible search returning top chunk strings."""
        hits = self.search_with_citations(query, n_results=n_results)
        return [h["text"] for h in hits]

    def get_all_documents(self) -> list[dict]:
        """Lists all unique documents across Local ONNX and Gemini collections."""
        docs: dict[str, dict] = {}
        for col in (self.local_collection, self.gemini_collection, self.legacy_collection):
            if col is None:
                continue
            try:
                results = col.get(include=["metadatas"])
                for meta in results.get("metadatas") or []:
                    if meta and "document_id" in meta:
                        doc_id = meta["document_id"]
                        if doc_id not in docs:
                            docs[doc_id] = {
                                "id": doc_id,
                                "title": meta.get("title", "Unknown Document"),
                                "source": meta.get("source_url") or meta.get("file_path") or "Unknown",
                                "embedding_mode": meta.get("embedding_mode", "local_onnx"),
                                "physical_file": meta.get("physical_file"),
                            }
            except Exception:
                continue
        return list(docs.values())

    def delete_document(self, document_id: str):
        """Deletes a document by ID across all collections."""
        for col in (self.local_collection, self.gemini_collection, self.legacy_collection):
            if col is None:
                continue
            try:
                col.delete(where={"document_id": document_id})
            except Exception:
                continue
