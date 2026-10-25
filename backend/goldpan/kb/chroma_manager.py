import math
import os
import re
import time
import unicodedata
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from goldpan.filters.semantic_router import _split_preserving_fences

GEMINI_EMBEDDING_DIM = 768
RRF_K = 60


def compute_document_id(text: str, source: Optional[str] = None) -> str:
    """
    Computes a deterministic, content-addressed UUID5 for idempotent ingestion.
    Re-ingesting the same document content and source yields the exact same
    document_id so duplicate vectors and KB/*.md files never accumulate.
    """
    norm_text = unicodedata.normalize("NFC", (text or "").strip())
    norm_source = unicodedata.normalize("NFC", (source or "").strip())
    seed = f"{norm_source}\n---\n{norm_text}"
    return str(uuid.uuid5(uuid.NAMESPACE_OID, seed))


def _l2_normalize(vec: list[float]) -> list[float]:
    """L2-normalizes truncated Matryoshka embeddings (e.g. 768d slice of gemini-embedding-001)."""
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 1e-12:
        return [v / norm for v in vec]
    return vec


def _fold_vietnamese_diacritics(text: str) -> str:
    """
    Converts Vietnamese & Latin text to ASCII-folded form (including đ/Đ -> d)
    so both accented and unaccented queries match accurately.
    """
    s = text.replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def _tokenize_multilingual(text: str) -> list[str]:
    """
    Produces both NFC-accented unigrams and ASCII-folded unigrams + bigrams
    for high-recall Vietnamese & multilingual BM25 lexical scoring.
    """
    nfc_lower = unicodedata.normalize("NFC", text).lower()
    raw_words = re.findall(r"\w+", nfc_lower, flags=re.UNICODE)
    if not raw_words:
        return []

    folded_words = [_fold_vietnamese_diacritics(w) for w in raw_words]
    tokens = list(raw_words)
    for fw in folded_words:
        if fw and fw not in raw_words:
            tokens.append(fw)

    # Add ASCII-folded bigrams to capture 2-syllable Vietnamese compound words (e.g. "tinh_diem", "ban_to_chuc")
    for i in range(len(folded_words) - 1):
        if folded_words[i] and folded_words[i + 1]:
            tokens.append(f"{folded_words[i]}_{folded_words[i + 1]}")

    return tokens


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
    Dual-Mode Hybrid Knowledge Base Manager:
    - Local Mode (100% Offline Default): Uses ChromaDB's local ONNX `all-MiniLM-L6-v2` (384-dim)
      in `goldpan_kb_local` fused with a Vietnamese-aware BM25 + Diacritic-Folded Lexical Ranker.
    - Gemini Cloud Mode (Optional): Uses `gemini-embedding-001` with explicit `outputDimensionality=768`
      and L2 normalization in `goldpan_kb_gemini`.
    - Reciprocal Rank Fusion (RRF): Combines rankings across Local ONNX, Gemini Cloud, and
      Multilingual BM25 without comparing incompatible raw vector distances across spaces.
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

    def _get_gemini_embeddings(
        self,
        texts: list[str],
        output_dimensionality: int = GEMINI_EMBEDDING_DIM,
    ) -> list[list[float]]:
        if not self.api_key:
            raise ValueError("Gemini API Key is required for Cloud Embeddings.")

        import requests

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={self.api_key}"

        def embed_single(text: str, index: int):
            payload = {
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": text}]},
                "outputDimensionality": output_dimensionality,
            }
            max_retries = 5
            backoff = 1.5
            for attempt in range(max_retries):
                try:
                    res = requests.post(url, json=payload, timeout=20)
                    if res.ok:
                        data = res.json()
                        raw_values = data["embedding"]["values"]
                        return index, _l2_normalize(raw_values)

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
        Normalizes Unicode to NFC and splits along Markdown section headers and
        paragraph boundaries while keeping fenced code blocks and tables intact.
        """
        cleaned = unicodedata.normalize("NFC", text).strip()
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

    def ingest_document(
        self,
        document_id: Optional[str],
        text: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Splits Markdown into semantic chunks and stores them in ChromaDB.
        If document_id is None or 'auto', computes a deterministic content-hash ID.
        Returns the embedding mode used ('local_onnx' or 'gemini_api').
        """
        if not text or not text.strip():
            return "none"

        norm_text = unicodedata.normalize("NFC", text)
        chunks = self._chunk_text(norm_text)
        if not chunks:
            return "none"

        source_key = (metadata or {}).get("source_url") or (metadata or {}).get("file_path") or (metadata or {}).get("title")
        if not document_id or document_id == "auto":
            document_id = compute_document_id(norm_text, source_key)

        # Remove any stale chunks for this document_id (or matching source_url/file_path) to prevent duplicates
        self.delete_document(document_id)

        ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
        base_meta = _sanitize_metadata(metadata, document_id)

        # Try Gemini Cloud Embedding (768d MRL) when API Key is provided and prefer_local is False
        if self.api_key and not self.prefer_local:
            try:
                embeddings = self._get_gemini_embeddings(chunks, output_dimensionality=GEMINI_EMBEDDING_DIM)
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

        # Default / Fallback: 100% Local Offline ONNX Embedding (all-MiniLM-L6-v2, 384d)
        meta_with_mode = [{**base_meta, "embedding_mode": "local_onnx"} for _ in chunks]
        self.local_collection.upsert(
            documents=chunks,
            ids=ids,
            metadatas=meta_with_mode,
        )
        return "local_onnx"

    def _compute_bm25_lexical_ranks(self, query: str, pool_chunks: dict[str, dict]) -> list[str]:
        """
        Computes BM25 lexical scores with Vietnamese diacritic folding and compound bigrams
        over all indexed chunks, returning chunk_ids sorted from highest to lowest BM25 score.
        """
        q_tokens = _tokenize_multilingual(query)
        if not q_tokens or not pool_chunks:
            return []

        N = len(pool_chunks)
        doc_tokens: dict[str, Counter] = {}
        doc_lens: dict[str, int] = {}
        df: Counter = Counter()

        for cid, entry in pool_chunks.items():
            title_str = entry.get("title", "")
            toks = _tokenize_multilingual(f"{title_str} {entry['text']}")
            counts = Counter(toks)
            doc_tokens[cid] = counts
            doc_lens[cid] = max(1, len(toks))
            for t in counts.keys():
                df[t] += 1

        avgdl = sum(doc_lens.values()) / max(1, N)
        k1 = 1.5
        b = 0.75

        scored: list[tuple[float, str]] = []
        for cid, counts in doc_tokens.items():
            score = 0.0
            dl = doc_lens[cid]
            for qt in q_tokens:
                tf = counts.get(qt, 0)
                if tf == 0:
                    continue
                n_q = df.get(qt, 0)
                idf = math.log(1.0 + (N - n_q + 0.5) / (n_q + 0.5))
                denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
                score += idf * ((tf * (k1 + 1.0)) / denom)
            if score > 0.0:
                scored.append((score, cid))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [cid for _, cid in scored]

    def search_with_citations(self, query: str, n_results: int = 3) -> list[dict]:
        """
        Searches across active Local ONNX (384d) and Gemini (768d) collections plus
        a Vietnamese-aware BM25 Lexical Ranker, fusing rankings via Reciprocal Rank
        Fusion (RRF) so distances from incompatible embedding spaces are never mixed directly.
        """
        if not query or not query.strip():
            return []

        norm_query = unicodedata.normalize("NFC", query.strip())
        pool_size = max(n_results * 4, 12)

        # Map of chunk_id -> candidate metadata & per-modality ranks
        candidates_by_id: dict[str, dict] = {}
        local_rank_order: list[str] = []
        gemini_rank_order: list[str] = []

        # 1. Query Local ONNX collection (384d)
        try:
            local_cnt = self.local_collection.count()
            if local_cnt > 0:
                res = self.local_collection.query(
                    query_texts=[norm_query],
                    n_results=min(pool_size, local_cnt),
                    include=["documents", "metadatas", "distances"],
                )
                ids = (res.get("ids") or [[]])[0]
                docs = (res.get("documents") or [[]])[0]
                metas = (res.get("metadatas") or [[]])[0]
                dists = (res.get("distances") or [[]])[0]
                for cid, doc, meta, dist in zip(ids, docs, metas, dists):
                    local_rank_order.append(cid)
                    candidates_by_id[cid] = {
                        "chunk_id": cid,
                        "document_id": (meta or {}).get("document_id"),
                        "text": doc,
                        "title": (meta or {}).get("title", "Unknown Document"),
                        "source": (meta or {}).get("source_url") or (meta or {}).get("file_path") or "Local KB",
                        "embedding_mode": (meta or {}).get("embedding_mode", "local_onnx"),
                        "distance": float(dist) if dist is not None else 1.0,
                    }

                # Also load any remaining local chunks into candidate pool for full-collection Vietnamese BM25 scan
                if local_cnt <= 2000:
                    all_local = self.local_collection.get(include=["documents", "metadatas"])
                    for cid, doc, meta in zip(
                        all_local.get("ids") or [],
                        all_local.get("documents") or [],
                        all_local.get("metadatas") or [],
                    ):
                        if cid not in candidates_by_id and doc:
                            candidates_by_id[cid] = {
                                "chunk_id": cid,
                                "document_id": (meta or {}).get("document_id"),
                                "text": doc,
                                "title": (meta or {}).get("title", "Unknown Document"),
                                "source": (meta or {}).get("source_url") or (meta or {}).get("file_path") or "Local KB",
                                "embedding_mode": (meta or {}).get("embedding_mode", "local_onnx"),
                                "distance": 1.0,
                            }
        except Exception as e:
            print(f"[ChromaManager] Local search warning: {e}")

        # 2. Query Gemini Cloud collection (768d) if API Key is available
        if self.api_key:
            try:
                cnt = self.gemini_collection.count()
                if cnt > 0:
                    q_emb = self._get_gemini_embeddings([norm_query], output_dimensionality=GEMINI_EMBEDDING_DIM)
                    res = self.gemini_collection.query(
                        query_embeddings=q_emb,
                        n_results=min(pool_size, cnt),
                        include=["documents", "metadatas", "distances"],
                    )
                    ids = (res.get("ids") or [[]])[0]
                    docs = (res.get("documents") or [[]])[0]
                    metas = (res.get("metadatas") or [[]])[0]
                    dists = (res.get("distances") or [[]])[0]
                    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
                        key = f"gemini::{cid}"
                        gemini_rank_order.append(key)
                        candidates_by_id[key] = {
                            "chunk_id": cid,
                            "document_id": (meta or {}).get("document_id"),
                            "text": doc,
                            "title": (meta or {}).get("title", "Unknown Document"),
                            "source": (meta or {}).get("source_url") or (meta or {}).get("file_path") or "Local KB",
                            "embedding_mode": (meta or {}).get("embedding_mode", "gemini_api"),
                            "distance": float(dist) if dist is not None else 1.0,
                        }
            except Exception as e:
                print(f"[ChromaManager] Cloud collection search warning: {e}")

        if not candidates_by_id:
            return []

        # 3. Compute Multilingual / Vietnamese BM25 Lexical Ranking across candidate pool
        lexical_rank_order = self._compute_bm25_lexical_ranks(norm_query, candidates_by_id)

        # 4. Fuse rankings via Reciprocal Rank Fusion (RRF, k=60)
        rrf_scores: dict[str, float] = {cid: 0.0 for cid in candidates_by_id}

        for rank_idx, cid in enumerate(local_rank_order, start=1):
            rrf_scores[cid] += 1.0 / (RRF_K + rank_idx)

        for rank_idx, cid in enumerate(gemini_rank_order, start=1):
            rrf_scores[cid] += 1.0 / (RRF_K + rank_idx)

        for rank_idx, cid in enumerate(lexical_rank_order, start=1):
            # Lexical BM25 match gets slightly higher weight (1.15) to boost exact Vietnamese/technical terms
            rrf_scores[cid] += 1.15 / (RRF_K + rank_idx)

        ranked_ids = sorted(
            candidates_by_id.keys(),
            key=lambda cid: (rrf_scores[cid], -candidates_by_id[cid]["distance"]),
            reverse=True,
        )

        # Deduplicate identical chunk texts if the same document exists in both collections
        results: list[dict] = []
        seen_texts: set[str] = set()
        for cid in ranked_ids:
            item = dict(candidates_by_id[cid])
            norm_t = item["text"].strip()
            if norm_t in seen_texts:
                continue
            seen_texts.add(norm_t)
            item["rrf_score"] = round(rrf_scores[cid], 5)
            item.pop("chunk_id", None)
            results.append(item)
            if len(results) >= n_results:
                break

        return results

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
