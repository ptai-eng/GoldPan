"""Dual-Mode Hybrid Knowledge Base Manager (Multilingual E5 v3 ONNX + SQLite FTS5 + Gemini 768d MRL)."""

import hashlib
import math
import os
import re
import time
import unicodedata
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import chromadb

from ..filters.vn_encoding import fold_vietnamese_diacritics, normalize_vietnamese_text
from ..firewall.context_firewall import AgentContextFirewall
from .chunker import HierarchicalBreadcrumbChunker
from .fts_index import SQLiteFTS5Index
from .onnx_embedder import LegacyMiniLMEmbeddingFunction, MultilingualE5EmbeddingFunction

GEMINI_EMBEDDING_DIM = 768
RRF_K = 60


def compute_document_id(text: str, source: Optional[str] = None) -> str:
    """
    Computes a deterministic, content-addressed UUID5 for idempotent ingestion.
    Normalizes Vietnamese encodings (VNI, TCVN3, NFD -> NFC) so identical texts hash identically.
    """
    norm_text, _ = normalize_vietnamese_text((text or "").strip())
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
    """Converts Vietnamese & Latin text to ASCII-folded form (including đ/Đ -> d)."""
    return fold_vietnamese_diacritics(text)


def _tokenize_multilingual(text: str) -> list[str]:
    """
    Produces both NFC-accented unigrams and ASCII-folded unigrams + bigrams
    for Vietnamese & multilingual BM25 lexical scoring.
    """
    nfc_text, _ = normalize_vietnamese_text(text)
    nfc_lower = nfc_text.lower()
    raw_words = re.findall(r"\w+", nfc_lower, flags=re.UNICODE)
    if not raw_words:
        return []

    folded_words = [_fold_vietnamese_diacritics(w) for w in raw_words]
    tokens = list(raw_words)
    for fw in folded_words:
        if fw and fw not in raw_words:
            tokens.append(fw)

    for i in range(len(folded_words) - 1):
        if folded_words[i] and folded_words[i + 1]:
            tokens.append(f"{folded_words[i]}_{folded_words[i + 1]}")

    return tokens


def _sanitize_metadata(metadata: Optional[dict], document_id: str) -> dict:
    """Ensures all metadata values passed to ChromaDB are non-null primitives (str, int, float, bool)."""
    clean: dict[str, str | int | float | bool] = {"document_id": document_id}
    for k, v in (metadata or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            clean[str(k)] = v
    return clean


class ChromaManager:
    """
    GoldPan v3.0 Dual-Mode Hybrid Knowledge Base Manager:
    - Local v3 Mode (100% Offline Default): Uses `intfloat/multilingual-e5-small` ONNX (384-dim)
      in `goldpan_kb_e5_v3` with automatic `query: ` / `passage: ` prefixing, Hierarchical Breadcrumb
      Chunking (`Phần > Chương > Mục > Điều > Khoản`), VNI/TCVN3 decoding, and persistent SQLite FTS5
      (`goldpan_fts.db`) lexical BM25 ranking.
    - Legacy v2 Mode (`embedding_model="all-MiniLM-L6-v2"`): Isolated in `goldpan_kb_local` for
      side-by-side benchmark comparison in `goldpan bench` and v2->v3 migration detection.
    - Gemini Cloud Mode (Optional): Uses `gemini-embedding-001` with `outputDimensionality=768`
      and L2 normalization in `goldpan_kb_gemini`.
    - Reciprocal Rank Fusion (RRF, k=60): Fuses rankings across Local E5 ONNX, Gemini Cloud, and SQLite FTS5.
    """

    COLLECTION_V3 = "goldpan_kb_e5_v3"
    COLLECTION_V2_LEGACY = "goldpan_kb_local"
    COLLECTION_GEMINI = "goldpan_kb_gemini"

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        prefer_local: bool = False,
        persist_dir: Optional[str] = None,
        embedding_model: str = "multilingual-e5-small",
        device_override: Optional[str] = None,
    ) -> None:
        resolved_dir = persist_directory or persist_dir
        if resolved_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            resolved_dir = os.path.join(base_dir, "chroma_data")

        self.persist_directory = str(resolved_dir)
        self.persist_dir = self.persist_directory
        self.embedding_model = embedding_model
        self.device_override = device_override
        os.makedirs(self.persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.chunker = HierarchicalBreadcrumbChunker(chunk_size=1000, overlap=150)

        if self.embedding_model == "all-MiniLM-L6-v2":
            self.collection_name = self.COLLECTION_V2_LEGACY
            self.local_ef: Any = LegacyMiniLMEmbeddingFunction()
            self.embedder = self.local_ef
            self.fts_index: Optional[SQLiteFTS5Index] = None
            self.local_collection = self.client.get_or_create_collection(
                name=self.COLLECTION_V2_LEGACY,
                embedding_function=self.local_ef,
                metadata={
                    "description": "GoldPan Legacy v2 Knowledge Base",
                    "embedding_model": "all-MiniLM-L6-v2",
                    "dimension": 384,
                    "schema_version": "2.1",
                },
            )
        else:
            self.collection_name = self.COLLECTION_V3
            self.local_ef = MultilingualE5EmbeddingFunction(device_override=device_override)
            self.embedder = self.local_ef
            self.fts_index = SQLiteFTS5Index(Path(self.persist_directory) / "goldpan_fts.db")
            self.local_collection = self.client.get_or_create_collection(
                name=self.COLLECTION_V3,
                embedding_function=self.local_ef,
                metadata={
                    "description": "GoldPan AI v3 Multilingual E5 Knowledge Base",
                    "embedding_model": "intfloat/multilingual-e5-small",
                    "dimension": 384,
                    "schema_version": "3.0",
                },
            )

        self.gemini_collection = self.client.get_or_create_collection(
            name=self.COLLECTION_GEMINI
        )
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

    def set_api_key(self, api_key: Optional[str]) -> None:
        self.api_key = api_key.strip() if api_key else None

    def check_migration_status(self) -> dict[str, Any]:
        """Check if legacy v2 collection (`goldpan_kb_local`) has documents while v3 is empty."""
        v2_count = 0
        try:
            existing = [c.name if hasattr(c, "name") else str(c) for c in self.client.list_collections()]
            if self.COLLECTION_V2_LEGACY in existing:
                v2_col = self.client.get_collection(self.COLLECTION_V2_LEGACY)
                v2_count = v2_col.count()
        except Exception:
            v2_count = 0

        v3_count = self.local_collection.count() if self.collection_name == self.COLLECTION_V3 else 0
        needs_migration = v2_count > 0 and v3_count == 0
        return {
            "v2_collection": self.COLLECTION_V2_LEGACY,
            "v2_chunks": v2_count,
            "v3_collection": self.COLLECTION_V3,
            "v3_chunks": v3_count,
            "needs_migration": needs_migration,
            "warning": (
                f"Detected {v2_count} legacy v2 chunks in '{self.COLLECTION_V2_LEGACY}' while "
                f"'{self.COLLECTION_V3}' is empty. Run `goldpan kb reindex` to migrate to Multilingual E5 v3."
                if needs_migration
                else ""
            ),
        }

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

        embeddings: list[Any] = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_idx = {executor.submit(embed_single, text, i): i for i, text in enumerate(texts)}
            for future in as_completed(future_to_idx):
                idx, emb = future.result()
                embeddings[idx] = emb

        return embeddings

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
        """Structure-aware hierarchical breadcrumb chunking (v3) or flat paragraph chunking (v2 baseline)."""
        return self.chunk_markdown(text, chunk_size=chunk_size, overlap=overlap)

    def chunk_markdown(self, text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
        if not text or not text.strip():
            return []

        if self.embedding_model == "all-MiniLM-L6-v2":
            paragraphs = re.split(r"\n\s*\n", text)
            chunks: list[str] = []
            current_chunk = ""
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if len(current_chunk) + len(para) + 2 <= chunk_size:
                    current_chunk = f"{current_chunk}\n\n{para}".strip()
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    if len(para) > chunk_size:
                        for i in range(0, len(para), max(1, chunk_size - overlap)):
                            chunks.append(para[i : i + chunk_size])
                        current_chunk = ""
                    else:
                        current_chunk = para
            if current_chunk:
                chunks.append(current_chunk)
            return chunks

        custom_chunker = HierarchicalBreadcrumbChunker(chunk_size=chunk_size, overlap=overlap)
        items = custom_chunker.chunk_document(text)
        return [item.text for item in items]

    def ingest_document(
        self,
        document_id: Optional[str],
        text: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Splits Markdown/Vietnamese text into hierarchical breadcrumb chunks and indexes into
        ChromaDB + SQLite FTS5. Returns the embedding mode used ('local_onnx' or 'gemini_api').
        """
        if not text or not text.strip():
            return "none"

        norm_text, detected_enc = normalize_vietnamese_text(text)
        source_key = (
            (metadata or {}).get("source_url")
            or (metadata or {}).get("file_path")
            or (metadata or {}).get("title")
            or "document"
        )
        if not document_id or document_id == "auto":
            document_id = compute_document_id(norm_text, str(source_key))

        self.delete_document(document_id)

        chunk_items = self.chunker.chunk_document(norm_text, source_name=str(source_key))
        if not chunk_items:
            return "none"

        chunks = [item.text for item in chunk_items]
        ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
        base_meta = _sanitize_metadata(metadata, document_id)
        base_meta["detected_encoding"] = detected_enc

        per_chunk_metas: list[dict[str, Any]] = []
        for item in chunk_items:
            c_meta = dict(base_meta)
            for k, v in item.metadata.items():
                if isinstance(v, (str, int, float, bool)) and k not in c_meta:
                    c_meta[k] = v
            per_chunk_metas.append(c_meta)

        if self.api_key and not self.prefer_local:
            try:
                embeddings = self._get_gemini_embeddings(chunks, output_dimensionality=GEMINI_EMBEDDING_DIM)
                meta_with_mode = [{**m, "embedding_mode": "gemini_api"} for m in per_chunk_metas]
                self.gemini_collection.upsert(
                    documents=chunks,
                    ids=ids,
                    metadatas=meta_with_mode,
                    embeddings=embeddings,
                )
                if self.fts_index is not None:
                    fts_metas = [
                        {
                            **m,
                            "source": m.get("source_url") or m.get("file_path") or m.get("source") or str(source_key),
                        }
                        for m in meta_with_mode
                    ]
                    self.fts_index.upsert_chunks(ids, chunks, fts_metas)
                return "gemini_api"
            except Exception as e:
                print(f"[ChromaManager] Gemini embedding unavailable ({e}), falling back to Local ONNX.")

        meta_with_mode = [{**m, "embedding_mode": "local_onnx"} for m in per_chunk_metas]
        local_embeddings = self.local_ef(chunks)
        self.local_collection.upsert(
            documents=chunks,
            ids=ids,
            embeddings=local_embeddings,
            metadatas=meta_with_mode,
        )
        if self.fts_index is not None:
            fts_metas = [
                {
                    **m,
                    "source": m.get("source_url") or m.get("file_path") or m.get("source") or str(source_key),
                }
                for m in meta_with_mode
            ]
            self.fts_index.upsert_chunks(ids, chunks, fts_metas)
        return "local_onnx"

    def add_document(
        self,
        text: str,
        source_name: str,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        """Add a document and return the number of chunks indexed."""
        if not text or not text.strip():
            return 0

        if self.embedding_model == "all-MiniLM-L6-v2":
            raw_chunks = self.chunk_markdown(text)
            ids: list[str] = []
            metadatas: list[dict[str, Any]] = []
            for idx, chunk in enumerate(raw_chunks):
                content_hash = hashlib.md5(f"{source_name}_{idx}_{chunk}".encode("utf-8")).hexdigest()
                meta: dict[str, Any] = {"source": source_name, "chunk_index": idx, "title": source_name}
                if extra_metadata:
                    meta.update(extra_metadata)
                ids.append(f"doc_{content_hash}")
                metadatas.append(meta)
            embeddings = self.local_ef(raw_chunks)
            self.local_collection.upsert(
                ids=ids,
                documents=raw_chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            return len(ids)

        meta_in = {"title": source_name, "file_path": source_name, "source": source_name}
        if extra_metadata:
            meta_in.update(extra_metadata)
        doc_id = compute_document_id(text, source_name)
        self.ingest_document(document_id=doc_id, text=text, metadata=meta_in)
        return len(self.chunker.chunk_document(text, source_name=source_name))

    def _compute_bm25_lexical_ranks(self, query: str, pool_chunks: dict[str, dict]) -> list[str]:
        """Computes BM25 lexical scores with Vietnamese diacritic folding and compound bigrams over candidate pool."""
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
        Searches across active Local E5 ONNX (384d), Gemini (768d), and persistent SQLite FTS5
        lexical index, fusing rankings via Reciprocal Rank Fusion (RRF, k=60).
        """
        if not query or not query.strip():
            return []

        norm_query, _ = normalize_vietnamese_text(query.strip())
        pool_size = max(n_results * 4, 12)

        candidates_by_id: dict[str, dict] = {}
        local_rank_order: list[str] = []
        gemini_rank_order: list[str] = []
        fts_rank_order: list[str] = []

        # 1. Query Local ONNX collection (384d) using `query: ` prefix via embed_query
        try:
            local_cnt = self.local_collection.count()
            if local_cnt > 0:
                q_vec = self.local_ef.embed_query(norm_query)
                try:
                    res = self.local_collection.query(
                        query_embeddings=[q_vec],
                        n_results=min(pool_size, local_cnt),
                        include=["documents", "metadatas", "distances"],
                    )
                    ids = (res.get("ids") or [[]])[0]
                    docs = (res.get("documents") or [[]])[0]
                    metas = (res.get("metadatas") or [[]])[0]
                    dists = (res.get("distances") or [[]])[0]
                except Exception:
                    raw = self.local_collection.get(include=["documents", "metadatas", "embeddings"])
                    r_ids = raw.get("ids") or []
                    r_docs = raw.get("documents") or []
                    r_metas = raw.get("metadatas") or []
                    r_embs = raw.get("embeddings") or []
                    scored_local = []
                    for cid_i, doc_i, meta_i, emb_i in zip(r_ids, r_docs, r_metas, r_embs):
                        d_val = sum((float(a) - float(b)) ** 2 for a, b in zip(q_vec, emb_i))
                        scored_local.append((d_val, cid_i, doc_i, meta_i))
                    scored_local.sort(key=lambda x: x[0])
                    scored_local = scored_local[: min(pool_size, local_cnt)]
                    ids = [x[1] for x in scored_local]
                    docs = [x[2] for x in scored_local]
                    metas = [x[3] for x in scored_local]
                    dists = [x[0] for x in scored_local]

                for cid, doc, meta, dist in zip(ids, docs, metas, dists):
                    key = f"local::{cid}"
                    local_rank_order.append(key)
                    candidates_by_id[key] = {
                        "chunk_id": cid,
                        "document_id": (meta or {}).get("document_id"),
                        "text": doc,
                        "title": (meta or {}).get("title", "Unknown Document"),
                        "source": (meta or {}).get("source_url")
                        or (meta or {}).get("file_path")
                        or (meta or {}).get("source")
                        or "Local KB",
                        "embedding_mode": (meta or {}).get("embedding_mode", "local_onnx"),
                        "distance": float(dist) if dist is not None else 1.0,
                        "metadata": meta or {},
                    }
        except Exception as e:
            print(f"[ChromaManager] Local collection search warning: {e}")

        # 2. Query Gemini Cloud collection (768d MRL) if API key is configured
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
                            "source": (meta or {}).get("source_url")
                            or (meta or {}).get("file_path")
                            or (meta or {}).get("source")
                            or "Local KB",
                            "embedding_mode": (meta or {}).get("embedding_mode", "gemini_api"),
                            "distance": float(dist) if dist is not None else 1.0,
                            "metadata": meta or {},
                        }
            except Exception as e:
                print(f"[ChromaManager] Cloud collection search warning: {e}")

        # 3. Query Persistent SQLite FTS5 Lexical Index (if v3 enabled)
        if self.fts_index is not None:
            try:
                fts_hits = self.fts_index.search_bm25(norm_query, limit=pool_size)
                for hit in fts_hits:
                    cid = hit["id"]
                    meta = hit.get("metadata") or {}
                    emb_mode = meta.get("embedding_mode", "local_onnx")
                    prefix = "gemini" if emb_mode == "gemini_api" else "local"
                    key = f"{prefix}::{cid}"
                    fts_rank_order.append(key)
                    if key not in candidates_by_id:
                        candidates_by_id[key] = {
                            "chunk_id": cid,
                            "document_id": meta.get("document_id"),
                            "text": hit["text"],
                            "title": meta.get("title", "Unknown Document"),
                            "source": meta.get("source_url")
                            or meta.get("file_path")
                            or meta.get("source")
                            or "Local KB",
                            "embedding_mode": emb_mode,
                            "distance": 0.5,
                            "bm25_score": hit.get("bm25_score", 0.0),
                            "metadata": meta,
                        }
                    else:
                        candidates_by_id[key]["bm25_score"] = hit.get("bm25_score", 0.0)
            except Exception as e:
                print(f"[ChromaManager] SQLite FTS5 search warning: {e}")

        if not candidates_by_id:
            return []

        # 4. Also compute in-pool multilingual BM25 ranking (covers Gemini chunks or legacy mode)
        lexical_rank_order = fts_rank_order or self._compute_bm25_lexical_ranks(norm_query, candidates_by_id)

        # 5. Fuse rankings via Reciprocal Rank Fusion (RRF, k=60)
        rrf_scores: dict[str, float] = {cid: 0.0 for cid in candidates_by_id}

        for rank_idx, cid in enumerate(local_rank_order, start=1):
            rrf_scores[cid] += 1.15 / (RRF_K + rank_idx)

        for rank_idx, cid in enumerate(gemini_rank_order, start=1):
            rrf_scores[cid] += 1.15 / (RRF_K + rank_idx)

        for rank_idx, cid in enumerate(lexical_rank_order, start=1):
            if cid in rrf_scores:
                rrf_scores[cid] += 1.0 / (RRF_K + rank_idx)

        ranked_ids = sorted(
            candidates_by_id.keys(),
            key=lambda cid: (rrf_scores[cid], -candidates_by_id[cid]["distance"]),
            reverse=True,
        )

        results: list[dict] = []
        seen_texts: set[str] = set()
        for cid in ranked_ids:
            item = dict(candidates_by_id[cid])
            norm_t = item["text"].strip()
            if norm_t in seen_texts:
                continue
            seen_texts.add(norm_t)
            item["rrf_score"] = round(rrf_scores[cid], 5)
            item["id"] = item.pop("chunk_id", cid)
            results.append(item)
            if len(results) >= n_results:
                break

        return results

    def search(self, query: str, n_results: int = 3) -> list[str]:
        """Backward-compatible search returning top chunk strings."""
        hits = self.search_with_citations(query, n_results=n_results)
        return [h["text"] for h in hits]

    def query(self, query_text: str, n_results: int = 5) -> list[dict[str, Any]]:
        """Perform pure dense vector search."""
        total_docs = self.local_collection.count()
        if total_docs == 0:
            return []

        actual_n = min(n_results, total_docs)
        q_emb = self.local_ef.embed_query(query_text)
        res = self.local_collection.query(query_embeddings=[q_emb], n_results=actual_n)
        formatted: list[dict[str, Any]] = []
        if res and res.get("documents") and res["documents"][0]:
            docs = res["documents"][0]
            metas = res["metadatas"][0] if res.get("metadatas") else [{}] * len(docs)
            dists = res["distances"][0] if res.get("distances") else [0.0] * len(docs)
            ids = res["ids"][0] if res.get("ids") else [""] * len(docs)
            for cid, doc, meta, dist in zip(ids, docs, metas, dists):
                formatted.append(
                    {
                        "id": cid,
                        "text": doc,
                        "metadata": meta or {},
                        "distance": round(float(dist), 4),
                    }
                )
        return formatted

    def query_hybrid(
        self, query_text: str, n_results: int = 5, rrf_k: int = 60
    ) -> list[dict[str, Any]]:
        """Perform Hybrid Search (Dense Vector + SQLite FTS5 / BM25) with Reciprocal Rank Fusion."""
        if self.embedding_model == "all-MiniLM-L6-v2":
            # Legacy v2 baseline hybrid search (surface BM25Okapi + MiniLM without folding/VNI/E5)
            total_docs = self.local_collection.count()
            if total_docs == 0:
                return []
            pool_size = min(max(n_results * 3, 15), total_docs)
            dense_results = self.query(query_text=query_text, n_results=pool_size)

            doc_lookup: dict[str, dict[str, Any]] = {}
            dense_ranks: dict[str, int] = {}
            bm25_ranks: dict[str, int] = {}

            for rank, item in enumerate(dense_results):
                cid = item["id"]
                dense_ranks[cid] = rank + 1
                doc_lookup[cid] = {
                    "id": cid,
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "distance": item["distance"],
                    "bm25_score": 0.0,
                }

            all_data = self.local_collection.get(include=["documents", "metadatas"])
            all_ids: list[str] = all_data.get("ids") or []
            all_docs: list[str] = all_data.get("documents") or []
            all_metas: list[dict[str, Any]] = all_data.get("metadatas") or [{}] * len(all_docs)

            tokenized_corpus = [re.findall(r"\w+", doc.lower()) for doc in all_docs]
            tokenized_query = re.findall(r"\w+", query_text.lower())
            if any(tokenized_corpus) and tokenized_query:
                N = len(tokenized_corpus)
                avgdl = sum(len(d) for d in tokenized_corpus) / max(1, N)
                df: Counter = Counter()
                for d in tokenized_corpus:
                    for w in set(d):
                        df[w] += 1
                raw_scores: list[float] = []
                for d in tokenized_corpus:
                    tf_map = Counter(d)
                    dl = max(1, len(d))
                    sc = 0.0
                    for qw in tokenized_query:
                        tf = tf_map.get(qw, 0)
                        if tf == 0:
                            continue
                        nq = df.get(qw, 0)
                        idf = math.log(1.0 + (N - nq + 0.5) / (nq + 0.5))
                        sc += idf * ((tf * 2.5) / (tf + 1.5 * (0.25 + 0.75 * (dl / avgdl))))
                    raw_scores.append(sc)
                scored_indices = sorted(range(len(all_ids)), key=lambda i: raw_scores[i], reverse=True)
                for rank_idx, doc_idx in enumerate(scored_indices[:pool_size]):
                    if raw_scores[doc_idx] > 0:
                        cid = all_ids[doc_idx]
                        bm25_ranks[cid] = rank_idx + 1
                        if cid not in doc_lookup:
                            doc_lookup[cid] = {
                                "id": cid,
                                "text": all_docs[doc_idx],
                                "metadata": all_metas[doc_idx],
                                "distance": 0.5,
                                "bm25_score": round(float(raw_scores[doc_idx]), 4),
                            }

            rrf_scores: dict[str, float] = {}
            candidates = set(dense_ranks.keys()) | set(bm25_ranks.keys())
            for cid in candidates:
                s = 0.0
                if cid in dense_ranks:
                    s += 1.0 / (rrf_k + dense_ranks[cid])
                if cid in bm25_ranks:
                    s += 1.0 / (rrf_k + bm25_ranks[cid])
                rrf_scores[cid] = s

            sorted_cids = sorted(candidates, key=lambda c: rrf_scores[c], reverse=True)
            out: list[dict[str, Any]] = []
            for cid in sorted_cids[: min(n_results, total_docs)]:
                info = doc_lookup[cid]
                info["rrf_score"] = round(rrf_scores[cid], 6)
                out.append(info)
            return out

        hits = self.search_with_citations(query=query_text, n_results=n_results)
        return [
            {
                "id": h.get("id", ""),
                "text": h["text"],
                "metadata": h.get("metadata") or {"source": h.get("source"), "title": h.get("title")},
                "distance": h.get("distance", 0.5),
                "bm25_score": h.get("bm25_score", 0.0),
                "rrf_score": h.get("rrf_score", 0.0),
            }
            for h in hits
        ]

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
                                "source": meta.get("source_url")
                                or meta.get("file_path")
                                or meta.get("source")
                                or "Unknown",
                                "embedding_mode": meta.get("embedding_mode", "local_onnx"),
                                "physical_file": meta.get("physical_file"),
                            }
            except Exception:
                continue
        return list(docs.values())

    def delete_document(self, document_id: str) -> None:
        """Deletes a document by ID across all collections."""
        for col in (self.local_collection, self.gemini_collection, self.legacy_collection):
            if col is None:
                continue
            try:
                existing = col.get(where={"document_id": document_id})
                target_ids = existing.get("ids") if existing else None
                if target_ids:
                    col.delete(ids=target_ids)
            except Exception:
                continue

    def reindex_from_kb_dir(self, kb_dir: str | Path | None = None) -> dict[str, Any]:
        """Re-index all `.md` files in `KB/` or `kb_data/` into v3 E5 + SQLite FTS5."""
        base_root = Path(__file__).resolve().parents[3]
        candidate_dirs = [Path(kb_dir)] if kb_dir else [base_root / "KB", base_root / "kb_data"]

        self.clear()
        files_indexed = 0
        chunks_indexed = 0

        for folder in candidate_dirs:
            if not folder.exists():
                continue
            for md_file in sorted(folder.glob("*.md")):
                content = md_file.read_text(encoding="utf-8", errors="ignore")
                body = re.sub(r"^---\n.*?\n---\n*", "", content, flags=re.DOTALL).strip()
                if body:
                    added = self.add_document(text=body, source_name=md_file.name)
                    files_indexed += 1
                    chunks_indexed += added

        return {
            "collection": self.collection_name,
            "embedding_model": self.embedding_model,
            "files_indexed": files_indexed,
            "total_v3_chunks": self.count(),
            "fts5_chunks": self.fts_index.count() if self.fts_index else 0,
        }

    def audit_kb(self) -> dict[str, Any]:
        """Scan all chunks currently in the Knowledge Base using AgentContextFirewall(mode='audit')."""
        firewall = AgentContextFirewall(mode="audit")
        data = self.local_collection.get(include=["documents", "metadatas"])
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or [{}] * len(docs)

        findings: list[dict[str, Any]] = []
        for cid, doc_text, meta in zip(ids, docs, metas):
            _, report = firewall.process_text(doc_text or "")
            if report.is_suspicious:
                findings.append(
                    {
                        "chunk_id": cid,
                        "source": (meta or {}).get("source_url")
                        or (meta or {}).get("file_path")
                        or (meta or {}).get("source")
                        or "unknown",
                        "risk_score": report.risk_score,
                        "signals": report.signals,
                    }
                )

        return {
            "collection": self.collection_name,
            "total_chunks_scanned": len(ids),
            "suspicious_chunks_count": len(findings),
            "status": "clean" if not findings else "suspicious_items_detected",
            "findings": findings,
        }

    def count(self) -> int:
        return self.local_collection.count()

    def clear(self) -> None:
        self.client.delete_collection(self.collection_name)
        if self.embedding_model == "all-MiniLM-L6-v2":
            self.local_collection = self.client.get_or_create_collection(
                name=self.COLLECTION_V2_LEGACY,
                embedding_function=self.local_ef,
                metadata={
                    "description": "GoldPan Legacy v2 Knowledge Base",
                    "embedding_model": "all-MiniLM-L6-v2",
                    "dimension": 384,
                    "schema_version": "2.1",
                },
            )
        else:
            self.local_collection = self.client.get_or_create_collection(
                name=self.COLLECTION_V3,
                embedding_function=self.local_ef,
                metadata={
                    "description": "GoldPan AI v3 Multilingual E5 Knowledge Base",
                    "embedding_model": "intfloat/multilingual-e5-small",
                    "dimension": 384,
                    "schema_version": "3.0",
                },
            )
            if self.fts_index is not None:
                self.fts_index.clear()
