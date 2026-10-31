"""Persistent SQLite FTS5 Lexical Index for GoldPan AI v3.0."""

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from ..filters.vn_encoding import fold_vietnamese_diacritics, normalize_vietnamese_text

_WORD_RE = re.compile(r"[0-9a-zA-ZÀ-ỹđĐ]+(?:[/_-][0-9a-zA-ZÀ-ỹđĐ]+)*")
_VN_STOPWORDS = {
    "va",
    "cua",
    "la",
    "co",
    "nhung",
    "cac",
    "duoc",
    "trong",
    "cho",
    "voi",
    "theo",
    "tai",
    "nay",
    "do",
    "khi",
    "ve",
    "nhu",
    "the",
    "nao",
    "bao",
    "nhieu",
    "the",
    "from",
    "with",
    "that",
    "this",
    "which",
    "what",
    "how",
    "are",
    "for",
    "and",
    "in",
    "to",
    "of",
    "on",
    "as",
    "by",
    "or",
    "an",
    "is",
    "be",
    "such",
    "must",
    "many",
}


def _extract_search_tokens_and_bigrams(text: str, fold: bool = False) -> list[str]:
    """Tokenize text into unigrams and adjacent compound bigrams."""
    target = fold_vietnamese_diacritics(text) if fold else normalize_vietnamese_text(text)[0].lower()
    raw_words = [w.replace("/", "_").replace("-", "_") for w in _WORD_RE.findall(target)]
    words = [w for w in raw_words if len(w) > 1 or w.isdigit()]
    bigrams: list[str] = []
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        if w1 not in _VN_STOPWORDS and w2 not in _VN_STOPWORDS:
            bigrams.append(f"{w1}_{w2}")
    return words + bigrams


class SQLiteFTS5Index:
    """Persistent SQLite FTS5 lexical index with Vietnamese diacritic folding and compound bigrams."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kb_chunks_meta (
                    chunk_id TEXT PRIMARY KEY,
                    source TEXT,
                    raw_text TEXT,
                    metadata_json TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS kb_chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    source,
                    text_nfc,
                    text_folded,
                    tokenize = 'unicode61'
                );
                """
            )
            conn.commit()

    def upsert_chunks(
        self,
        chunk_ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Insert or update chunks in SQLite FTS5 atomically."""
        if not chunk_ids:
            return
        with self._connect() as conn:
            for cid, text, meta in zip(chunk_ids, texts, metadatas):
                source = str(meta.get("source", ""))
                nfc_tokens = " ".join(_extract_search_tokens_and_bigrams(text, fold=False))
                folded_tokens = " ".join(_extract_search_tokens_and_bigrams(text, fold=True))

                conn.execute("DELETE FROM kb_chunks_fts WHERE chunk_id = ?", (cid,))
                conn.execute(
                    """
                    INSERT OR REPLACE INTO kb_chunks_meta (chunk_id, source, raw_text, metadata_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (cid, source, text, json.dumps(meta, ensure_ascii=False)),
                )
                conn.execute(
                    """
                    INSERT INTO kb_chunks_fts (chunk_id, source, text_nfc, text_folded)
                    VALUES (?, ?, ?, ?)
                    """,
                    (cid, source, nfc_tokens, folded_tokens),
                )
            conn.commit()

    def search_bm25(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Query the persistent FTS5 index using BM25 over both accented NFC and folded tokens."""
        if not query or not query.strip():
            return []

        nfc_terms = [
            t
            for t in _extract_search_tokens_and_bigrams(query, fold=False)
            if fold_vietnamese_diacritics(t) not in _VN_STOPWORDS
        ]
        folded_terms = [
            t
            for t in _extract_search_tokens_and_bigrams(query, fold=True)
            if t not in _VN_STOPWORDS
        ]

        # Sanitize FTS5 query tokens (alphanumeric + underscore only)
        safe_nfc = [re.sub(r"[^\w]", "", t) for t in nfc_terms]
        safe_folded = [re.sub(r"[^\w]", "", t) for t in folded_terms]
        safe_nfc = [t for t in safe_nfc if t]
        safe_folded = [t for t in safe_folded if t]

        if not safe_nfc and not safe_folded:
            return []

        clauses: list[str] = []
        for tok in dict.fromkeys(safe_nfc[:24]):
            clauses.append(f'text_nfc:"{tok}"')
        for tok in dict.fromkeys(safe_folded[:24]):
            clauses.append(f'text_folded:"{tok}"')

        match_expr = " OR ".join(clauses)
        if not match_expr:
            return []

        sql = """
            SELECT
                f.chunk_id AS chunk_id,
                m.raw_text AS raw_text,
                m.metadata_json AS metadata_json,
                bm25(kb_chunks_fts, 0.0, 1.5, 3.0, 2.0) AS bm25_score
            FROM kb_chunks_fts f
            JOIN kb_chunks_meta m ON f.chunk_id = m.chunk_id
            WHERE kb_chunks_fts MATCH ?
            ORDER BY bm25_score ASC
            LIMIT ?
        """
        results: list[dict[str, Any]] = []
        with self._connect() as conn:
            rows = conn.execute(sql, (match_expr, max(1, limit))).fetchall()
            for rank_idx, row in enumerate(rows):
                meta = {}
                try:
                    meta = json.loads(row["metadata_json"] or "{}")
                except Exception:
                    meta = {}
                # bm25() in SQLite FTS5 returns negative values (more negative = better match)
                raw_bm25 = float(row["bm25_score"])
                results.append(
                    {
                        "id": row["chunk_id"],
                        "text": row["raw_text"],
                        "metadata": meta,
                        "bm25_score": abs(raw_bm25),
                        "fts_rank": rank_idx + 1,
                    }
                )
        return results

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM kb_chunks_meta").fetchone()
            return int(row["cnt"]) if row else 0

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM kb_chunks_fts;")
            conn.execute("DELETE FROM kb_chunks_meta;")
            conn.commit()
