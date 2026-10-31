"""Structure-aware Hierarchical Breadcrumb Chunker & Vietnamese Legal Metadata Extractor."""

import re
from dataclasses import dataclass, field
from typing import Any

from ..filters.vn_encoding import normalize_vietnamese_text

_RE_PHAN = re.compile(r"^\s*(Phần\s+(?:thứ\s+[\wÀ-ỹ]+|[IVXLCDM0-9]+)\s*[:.\-]?\s*.*)$", re.IGNORECASE)
_RE_CHUONG = re.compile(r"^\s*(Chương\s+[IVXLCDM0-9]+\s*[:.\-]?\s*.*)$", re.IGNORECASE)
_RE_MUC = re.compile(r"^\s*(Mục\s+[0-9IVXLCDM]+\s*[:.\-]?\s*.*)$", re.IGNORECASE)
_RE_DIEU = re.compile(r"^\s*(Điều\s+\d+[a-zA-Z]?\s*[:.\-]?\s*.*)$", re.IGNORECASE)
_RE_KHOAN = re.compile(r"^\s*(Khoản\s+\d+\s*[:.\-]?\s*.*)$", re.IGNORECASE)
_RE_MD_HEADER = re.compile(r"^\s*(#{1,4})\s+(.+?)\s*$")

_RE_SO_HIEU = re.compile(
    r"(?:Số|Luật\s+số|Nghị\s+định\s+số|Thông\s+tư\s+số|Quyết\s+định\s+số)\s*[:.]?\s*([0-9]+(?:/[0-9]{4})?/[A-ZĐa-zđ0-9\-]+)",
    re.IGNORECASE,
)
_RE_NGAY_BAN_HANH = re.compile(
    r"(?:Ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4}))",
    re.IGNORECASE,
)
_RE_LOAI_VAN_BAN = re.compile(
    r"\b(LUẬT|BỘ\s+LUẬT|NGHỊ\s+ĐỊNH|THÔNG\s+TƯ|QUYẾT\s+ĐỊNH|NGHỊ\s+QUYẾT|CHỈ\s+THỊ)\b",
    re.IGNORECASE,
)


@dataclass
class ChunkItem:
    """A single context-enriched document chunk ready for dense + FTS5 indexing."""

    chunk_index: int
    text: str
    raw_text: str
    breadcrumb: str
    metadata: dict[str, Any] = field(default_factory=dict)


def extract_legal_metadata(text: str) -> dict[str, str]:
    """Extract Vietnamese legal document metadata (so_hieu, ngay_ban_hanh, loai_van_ban)."""
    meta: dict[str, str] = {}
    if not text:
        return meta

    m_so = _RE_SO_HIEU.search(text)
    if m_so:
        meta["so_hieu"] = m_so.group(1).strip()

    m_ngay = _RE_NGAY_BAN_HANH.search(text)
    if m_ngay:
        day, month, year = m_ngay.groups()
        meta["ngay_ban_hanh"] = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

    m_loai = _RE_LOAI_VAN_BAN.search(text)
    if m_loai:
        meta["loai_van_ban"] = re.sub(r"\s+", " ", m_loai.group(1).upper()).strip()

    return meta


class HierarchicalBreadcrumbChunker:
    """Splits Markdown and Vietnamese legal texts while preserving structural hierarchy."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 150) -> None:
        self.chunk_size = max(250, chunk_size)
        self.overlap = max(0, min(overlap, self.chunk_size // 2))

    def chunk_document(self, text: str, source_name: str = "") -> list[ChunkItem]:
        """Normalize Vietnamese encoding, track hierarchy headers, and emit enriched chunks."""
        clean_text, detected_encoding = normalize_vietnamese_text(text)
        if not clean_text.strip():
            return []

        doc_legal_meta = extract_legal_metadata(clean_text)
        doc_legal_meta["detected_encoding"] = detected_encoding
        if source_name:
            doc_legal_meta["source"] = source_name

        # Hierarchy state slots:
        # Legal: phan -> chuong -> muc -> dieu
        # Markdown: h1 -> h2 -> h3
        hierarchy: dict[str, str] = {
            "h1": "",
            "h2": "",
            "h3": "",
            "phan": "",
            "chuong": "",
            "muc": "",
            "dieu": "",
        }

        lines = clean_text.splitlines()
        sections: list[tuple[str, str, dict[str, str]]] = []
        current_lines: list[str] = []
        current_breadcrumb = self._format_breadcrumb(hierarchy)
        current_dieu = ""

        def flush_section() -> None:
            nonlocal current_lines
            body = "\n".join(current_lines).strip()
            if body:
                sec_meta = dict(doc_legal_meta)
                if current_dieu:
                    sec_meta["dieu"] = current_dieu
                sections.append((current_breadcrumb, body, sec_meta))
            current_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                current_lines.append("")
                continue

            m_md = _RE_MD_HEADER.match(stripped)
            if m_md:
                flush_section()
                hashes, title = m_md.groups()
                level = len(hashes)
                if level == 1:
                    hierarchy["h1"] = title.strip()
                    hierarchy["h2"] = ""
                    hierarchy["h3"] = ""
                elif level == 2:
                    hierarchy["h2"] = title.strip()
                    hierarchy["h3"] = ""
                else:
                    hierarchy["h3"] = title.strip()
                current_breadcrumb = self._format_breadcrumb(hierarchy)
                current_lines.append(stripped)
                continue

            m_phan = _RE_PHAN.match(stripped)
            if m_phan:
                flush_section()
                hierarchy["phan"] = m_phan.group(1).strip()
                hierarchy["chuong"] = ""
                hierarchy["muc"] = ""
                hierarchy["dieu"] = ""
                current_breadcrumb = self._format_breadcrumb(hierarchy)
                current_lines.append(stripped)
                continue

            m_chuong = _RE_CHUONG.match(stripped)
            if m_chuong:
                flush_section()
                hierarchy["chuong"] = m_chuong.group(1).strip()
                hierarchy["muc"] = ""
                hierarchy["dieu"] = ""
                current_breadcrumb = self._format_breadcrumb(hierarchy)
                current_lines.append(stripped)
                continue

            m_muc = _RE_MUC.match(stripped)
            if m_muc:
                flush_section()
                hierarchy["muc"] = m_muc.group(1).strip()
                hierarchy["dieu"] = ""
                current_breadcrumb = self._format_breadcrumb(hierarchy)
                current_lines.append(stripped)
                continue

            m_dieu = _RE_DIEU.match(stripped)
            if m_dieu:
                flush_section()
                dieu_text = m_dieu.group(1).strip()
                hierarchy["dieu"] = dieu_text
                m_dieu_id = re.match(r"^(Điều\s+\d+[a-zA-Z]?)", dieu_text, re.IGNORECASE)
                current_dieu = m_dieu_id.group(1) if m_dieu_id else dieu_text
                current_breadcrumb = self._format_breadcrumb(hierarchy)
                current_lines.append(stripped)
                continue

            current_lines.append(line)

        flush_section()

        chunks: list[ChunkItem] = []
        chunk_idx = 0
        for breadcrumb, sec_body, sec_meta in sections:
            sub_chunks = self._split_section_body(sec_body)
            for raw_sub in sub_chunks:
                if breadcrumb:
                    enriched = f"[Ngữ cảnh: {breadcrumb}]\n{raw_sub}"
                else:
                    enriched = raw_sub
                meta = dict(sec_meta)
                meta["breadcrumb"] = breadcrumb
                chunks.append(
                    ChunkItem(
                        chunk_index=chunk_idx,
                        text=enriched,
                        raw_text=raw_sub,
                        breadcrumb=breadcrumb,
                        metadata=meta,
                    )
                )
                chunk_idx += 1

        return chunks

    @staticmethod
    def _format_breadcrumb(hierarchy: dict[str, str]) -> str:
        parts: list[str] = []
        for key in ("h1", "h2", "h3", "phan", "chuong", "muc", "dieu"):
            val = hierarchy.get(key, "").strip()
            if val and val not in parts:
                parts.append(val)
        return " > ".join(parts)

    def _split_section_body(self, body: str) -> list[str]:
        if len(body) <= self.chunk_size:
            return [body]

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        if not paragraphs:
            return [body]

        results: list[str] = []
        current = ""
        for para in paragraphs:
            candidate = f"{current}\n\n{para}".strip() if current else para
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    results.append(current)
                if len(para) <= self.chunk_size:
                    current = para
                else:
                    # Hard window split with overlap
                    step = max(1, self.chunk_size - self.overlap)
                    for start in range(0, len(para), step):
                        piece = para[start : start + self.chunk_size].strip()
                        if piece:
                            results.append(piece)
                    current = ""
        if current:
            results.append(current)
        return results
