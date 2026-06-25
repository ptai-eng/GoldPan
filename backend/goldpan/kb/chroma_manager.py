import os
import uuid
import chromadb
from google import genai

class ChromaManager:
    def __init__(self, persist_directory: str = None):
        if persist_directory is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            persist_directory = os.path.join(base_dir, "chroma_data")
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Không dùng default embedding function nữa, ta tự nhúng bằng Gemini
        self.collection = self.client.get_or_create_collection(name="goldpan_kb")
        self.api_key = None
        
    def set_api_key(self, api_key: str):
        self.api_key = api_key
        
    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise ValueError("Gemini API Key is required for Knowledge Base operations.")
        
        import requests
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={self.api_key}"
        
        def embed_single(text: str, index: int):
            payload = {
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": text}]}
            }
            res = requests.post(url, json=payload)
            if not res.ok:
                raise ValueError(f"Gemini API Error: {res.text}")
            data = res.json()
            return index, data["embedding"]["values"]
            
        embeddings = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_idx = {executor.submit(embed_single, text, i): i for i, text in enumerate(texts)}
            for future in as_completed(future_to_idx):
                idx, emb = future.result()
                embeddings[idx] = emb
                
        return embeddings

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def ingest_document(self, document_id: str, text: str, metadata: dict = None):
        """Chia nhỏ văn bản và lưu vào ChromaDB."""
        if not text.strip():
            return
            
        chunks = self._chunk_text(text)
        if not chunks:
            return
            
        ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [metadata or {} for _ in chunks]
        
        # Tự lấy embeddings từ Gemini
        embeddings = self._get_embeddings(chunks)
        
        self.collection.upsert(
            documents=chunks,
            ids=ids,
            metadatas=metadatas,
            embeddings=embeddings
        )

    def search(self, query: str, n_results: int = 3) -> list[str]:
        """Tìm kiếm các chunk liên quan nhất đến câu hỏi."""
        query_embeddings = self._get_embeddings([query])
        
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results
        )
        if not results['documents'] or not results['documents'][0]:
            return []
        return results['documents'][0]
        
    def get_all_documents(self):
        """Lấy danh sách các tài liệu đã lưu."""
        results = self.collection.get(include=['metadatas'])
        
        docs = {}
        for meta in results['metadatas']:
            if meta and 'document_id' in meta:
                doc_id = meta['document_id']
                if doc_id not in docs:
                    docs[doc_id] = {
                        "id": doc_id,
                        "title": meta.get('title', 'Unknown Document'),
                        "source": meta.get('source_url', meta.get('file_path', 'Unknown'))
                    }
        return list(docs.values())
        
    def delete_document(self, document_id: str):
        self.collection.delete(where={"document_id": document_id})
