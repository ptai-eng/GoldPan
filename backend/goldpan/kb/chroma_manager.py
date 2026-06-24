import os
import uuid
import chromadb
from chromadb.utils import embedding_functions

class ChromaManager:
    def __init__(self, persist_directory: str = "chroma_data"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Use default local embedding function (all-MiniLM-L6-v2) for 100% free offline usage
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="goldpan_kb",
            embedding_function=self.embedding_fn
        )

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
        
        self.collection.upsert(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )

    def search(self, query: str, n_results: int = 3) -> list[str]:
        """Tìm kiếm các chunk liên quan nhất đến câu hỏi."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        if not results['documents'] or not results['documents'][0]:
            return []
        return results['documents'][0]
        
    def get_all_documents(self):
        """Lấy danh sách các tài liệu đã lưu (dựa trên metadata)."""
        results = self.collection.get(include=['metadatas'])
        
        # Lọc để lấy danh sách document unique (mỗi chunk chung document_id)
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
        """Xóa toàn bộ chunk của một tài liệu."""
        self.collection.delete(where={"document_id": document_id})
