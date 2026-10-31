"""GoldPan AI MCP Server (stdio JSON-RPC 2.0) following rea-tool-design contracts & v3.0 Context Firewall."""

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

from . import __version__
from .core.pipeline import GoldPanPipeline
from .core.schemas import FirewallReport
from .extractors.liteparse_wrapper import LiteParseExtractor
from .extractors.markitdown_wrapper import MarkItDownExtractor
from .extractors.web_extractor import HybridWebExtractor
from .filters.heuristic_filter import HeuristicFilter
from .filters.semantic_router import SemanticRouterFilter
from .firewall.context_firewall import build_provenance_meta, spotlight_untrusted_content
from .formatters.markdown_formatter import AIFriendlyMarkdownFormatter
from .kb.chroma_manager import ChromaManager, compute_document_id
from .kb.onnx_embedder import collect_doctor_diagnostics

PROTOCOL_VERSION = "2024-11-05"


def _is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def _build_pipeline(
    source: str,
    api_key: Optional[str] = None,
    use_playwright_fallback: bool = True,
    firewall_mode: str = "sanitize",
    enable_ocr: bool = True,
) -> GoldPanPipeline:
    if _is_url(source):
        extractor = HybridWebExtractor(
            use_playwright_fallback=use_playwright_fallback,
            firewall_mode=firewall_mode,
        )
    else:
        ext = os.path.splitext(source)[1].lower()
        if ext in [".txt", ".csv", ".json", ".md", ".xml"]:
            extractor = LiteParseExtractor()
        else:
            extractor = MarkItDownExtractor(
                api_key=api_key,
                firewall_mode=firewall_mode,
                enable_ocr=enable_ocr,
            )

    pipeline = GoldPanPipeline(extractor=extractor)
    pipeline.add_filter(HeuristicFilter())
    pipeline.add_filter(SemanticRouterFilter(api_key=api_key))
    pipeline.set_formatter(AIFriendlyMarkdownFormatter(firewall_mode=firewall_mode))
    return pipeline


class GoldPanMCPServer:
    """
    Model Context Protocol (MCP) server for GoldPan AI v3.0.
    Enforces rea-tool-design principles + Agent Context Firewall:
    1. Explicit `coverage` & `residual_unknowns`
    2. Untrusted content provenance (`_meta.provenance.untrusted_content = true` + `_meta.provenance.firewall`)
    3. Response budget & progressive detail (`detail: "summary" | "complete"`)
    4. Optional XML Data Spotlighting (`<untrusted_document_content>`)
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        kb_dir: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.kb_dir = kb_dir or os.path.join(repo_root, "KB")
        os.makedirs(self.kb_dir, exist_ok=True)
        self.chroma = ChromaManager(persist_directory=persist_directory)
        self.api_key = os.getenv("GEMINI_API_KEY") if api_key is None else (api_key or None)
        if self.api_key:
            self.chroma.set_api_key(self.api_key)

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "goldpan_extract",
                "description": (
                    "Extract clean, AI-ready Markdown from any URL (Web, GitHub blob, YouTube) or local file "
                    "(PDF, DOCX, CSV, JSON, XML, Image, Audio) protected by the Agent Context Firewall. "
                    "Defaults to detail='summary' with a bounded character budget; pass detail='complete' for full output."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "HTTP/HTTPS URL or local file path to extract.",
                        },
                        "detail": {
                            "type": "string",
                            "enum": ["summary", "complete"],
                            "default": "summary",
                            "description": "Response detail level: 'summary' caps output at max_chars; 'complete' returns full text.",
                        },
                        "max_chars": {
                            "type": "integer",
                            "default": 4000,
                            "minimum": 200,
                            "maximum": 200000,
                            "description": "Maximum characters returned when detail='summary'.",
                        },
                        "ingest_to_kb": {
                            "type": "boolean",
                            "default": False,
                            "description": "If true, automatically chunks and ingests the extracted Markdown into ChromaDB and KB/.",
                        },
                        "use_playwright_fallback": {
                            "type": "boolean",
                            "default": True,
                            "description": "Allow headless Playwright fallback for JS-heavy web pages.",
                        },
                        "firewall_mode": {
                            "type": "string",
                            "enum": ["sanitize", "audit", "strict"],
                            "default": "sanitize",
                            "description": "Agent Context Firewall mode: 'sanitize' (default), 'audit', or 'strict'.",
                        },
                        "enable_ocr": {
                            "type": "boolean",
                            "default": True,
                            "description": "Enable offline Vietnamese & multilingual OCR for scanned PDFs and images.",
                        },
                        "spotlight": {
                            "type": "boolean",
                            "default": False,
                            "description": "If true, wraps returned markdown in <untrusted_document_content> XML spotlight tags.",
                        },
                    },
                    "required": ["source"],
                },
            },
            {
                "name": "goldpan_kb_ingest",
                "description": (
                    "Ingest Markdown content into the local GoldPan v3 Knowledge Base (Multilingual E5 ONNX + "
                    "SQLite FTS5 + Hierarchical Breadcrumb Chunker + physical KB/*.md mirror)."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "markdown": {
                            "type": "string",
                            "description": "Clean Markdown content to chunk and index.",
                        },
                        "title": {
                            "type": "string",
                            "description": "Human-readable document title.",
                        },
                        "source": {
                            "type": "string",
                            "description": "Origin URL or file path for citation tracking.",
                        },
                        "prefer_local": {
                            "type": "boolean",
                            "default": True,
                            "description": "Use 100% offline Local ONNX embeddings (intfloat/multilingual-e5-small, 384d).",
                        },
                    },
                    "required": ["markdown", "title"],
                },
            },
            {
                "name": "goldpan_kb_search",
                "description": (
                    "Run Hybrid RRF search (Multilingual E5 384d + SQLite FTS5 BM25 + Gemini 768d) across the "
                    "GoldPan Knowledge Base and return ranked passages with citations, coverage, and firewall provenance."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query (supports accented/unaccented Vietnamese & English).",
                        },
                        "n_results": {
                            "type": "integer",
                            "default": 4,
                            "minimum": 1,
                            "maximum": 20,
                            "description": "Maximum number of relevant chunks to retrieve.",
                        },
                        "detail": {
                            "type": "string",
                            "enum": ["summary", "complete"],
                            "default": "summary",
                            "description": "In 'summary' mode each chunk preview is capped at 800 chars; 'complete' returns full chunks.",
                        },
                        "spotlight": {
                            "type": "boolean",
                            "default": False,
                            "description": "If true, wraps each retrieved passage in <untrusted_document_content> tags.",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "goldpan_kb_list",
                "description": "List all indexed documents in the GoldPan Knowledge Base with their IDs, titles, sources, and embedding modes.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "goldpan_kb_delete",
                "description": "Delete a document by document_id from ChromaDB collections and remove its physical KB/*.md file.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "UUID of the document to delete.",
                        },
                    },
                    "required": ["document_id"],
                },
            },
            {
                "name": "goldpan_kb_audit",
                "description": "Scan all chunks currently in the Knowledge Base for prompt injection signatures using AgentContextFirewall(mode='audit').",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "goldpan_doctor",
                "description": "Inspect hardware execution providers (Tier 1 CPU INT8, Tier 2 DirectML/CoreML/OpenVINO, Tier 3 CUDA), warmup latency, SQLite FTS5 status, and v2->v3 migration health.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "goldpan_ocr",
                "description": (
                    "Run 100% offline Vietnamese & Multilingual OCR on a scanned PDF or image file (.png, .jpg, .webp). "
                    "Performs image contrast enhancement, binarization, and Agent Context Firewall sanitization."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "Local file path to scanned PDF or image file.",
                        },
                        "dpi": {
                            "type": "integer",
                            "default": 200,
                            "description": "Rasterization DPI for PDF pages (150-300).",
                        },
                        "spotlight": {
                            "type": "boolean",
                            "default": False,
                            "description": "Wrap OCR markdown in <untrusted_document_content> XML spotlight tags.",
                        },
                    },
                    "required": ["source"],
                },
            },
        ]

    def get_resources(self) -> List[Dict[str, Any]]:
        docs = self.chroma.get_all_documents()
        resources = []
        for doc in docs:
            doc_id = doc.get("id")
            if not doc_id:
                continue
            resources.append(
                {
                    "uri": f"goldpan://kb/{doc_id}",
                    "name": doc.get("title", doc_id),
                    "description": f"Ingested Markdown document from {doc.get('source', 'local')} ({doc.get('embedding_mode', 'local_onnx')})",
                    "mimeType": "text/markdown",
                }
            )
        return resources

    def read_resource(self, uri: str) -> Dict[str, Any]:
        if not uri.startswith("goldpan://kb/"):
            raise ValueError(f"Unsupported resource URI: {uri}")
        doc_id = uri.replace("goldpan://kb/", "", 1).strip()
        for doc in self.chroma.get_all_documents():
            if doc.get("id") == doc_id:
                p = doc.get("physical_file")
                if p and os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as f:
                        text = f.read()
                    return {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "text/markdown",
                                "text": text,
                            }
                        ],
                        "_meta": build_provenance_meta(FirewallReport(mode="sanitize"), source=str(doc.get("source", p))),
                    }
        raise ValueError(f"Knowledge Base document not found for URI: {uri}")

    def _save_physical_kb_file(self, doc_id: str, source_name: str, markdown: str) -> str:
        safe_stem = re.sub(r'[\\/*?:"<>|]', "", os.path.basename(source_name))[:50] or "document"
        safe_filename = f"{safe_stem}_{doc_id[:8]}.md"
        physical_path = os.path.join(self.kb_dir, safe_filename)
        with open(physical_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        return physical_path

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if name == "goldpan_extract":
            source = str(arguments.get("source", "")).strip()
            if not source or "\x00" in source:
                raise ValueError("Invalid 'source' parameter.")
            if not _is_url(source):
                source = os.path.abspath(source)
                if not os.path.exists(source):
                    raise FileNotFoundError(f"Source file not found: {source}")

            detail = arguments.get("detail", "summary")
            max_chars = int(arguments.get("max_chars", 4000))
            ingest_to_kb = bool(arguments.get("ingest_to_kb", False))
            use_pw = bool(arguments.get("use_playwright_fallback", True))
            firewall_mode = str(arguments.get("firewall_mode", "sanitize"))
            spotlight = bool(arguments.get("spotlight", False))
            enable_ocr = bool(arguments.get("enable_ocr", True))

            pipeline = _build_pipeline(
                source,
                api_key=self.api_key,
                use_playwright_fallback=use_pw,
                firewall_mode=firewall_mode,
                enable_ocr=enable_ocr,
            )
            result = pipeline.process(source)
            full_text = result.text
            total_chars = len(full_text)
            fw_report = result.firewall or FirewallReport(mode=firewall_mode)

            residual_unknowns: List[str] = []
            router_mode = result.stats.get("router_mode", "local_heuristic")
            if router_mode == "local_heuristic":
                residual_unknowns.append(
                    "SemanticRouterFilter executed in local_heuristic mode (no GEMINI_API_KEY configured or document < 1000 chars)."
                )
            if not _is_url(source) and os.path.splitext(source)[1].lower() in [".jpg", ".jpeg", ".png", ".webp"] and not self.api_key:
                residual_unknowns.append(
                    "Image processed via local Pillow EXIF/geometry inspector; Cloud Vision OCR was not invoked."
                )

            truncated = False
            output_text = full_text
            if detail == "summary" and total_chars > max_chars:
                output_text = full_text[:max_chars]
                truncated = True
                residual_unknowns.append(
                    f"Output truncated at {max_chars}/{total_chars} characters under detail='summary'. Pass detail='complete' or increase max_chars for full text."
                )

            if spotlight:
                output_text = spotlight_untrusted_content(output_text, source=source, report=fw_report)

            ingested_info = None
            if ingest_to_kb:
                title = result.doc_info.title or os.path.basename(source)
                doc_id = compute_document_id(full_text, source)
                physical_path = self._save_physical_kb_file(doc_id, source, full_text)
                meta = {
                    "document_id": doc_id,
                    "title": title,
                    "source_url": result.doc_info.source_url,
                    "file_path": result.doc_info.file_path,
                    "physical_file": physical_path,
                }
                emb_mode = self.chroma.ingest_document(document_id=doc_id, text=full_text, metadata=meta)
                ingested_info = {
                    "document_id": doc_id,
                    "embedding_mode": emb_mode,
                    "physical_file": physical_path,
                }

            meta_obj = build_provenance_meta(fw_report, source=source)
            payload = {
                "title": result.doc_info.title,
                "source": source,
                "detail": detail,
                "truncated": truncated,
                "markdown": output_text,
                "stats": {
                    **result.stats,
                    "total_chars": total_chars,
                    "returned_chars": len(output_text),
                },
                "firewall": fw_report.model_dump(),
                "ingested": ingested_info,
                "coverage": {
                    "completeness": "partial" if (truncated or router_mode == "local_heuristic") else "complete",
                    "router_mode": router_mode,
                    "total_chars": total_chars,
                    "returned_chars": len(output_text),
                },
                "residual_unknowns": residual_unknowns,
                "_meta": meta_obj,
            }
            return {
                "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}],
                "_meta": payload["_meta"],
            }

        elif name == "goldpan_kb_ingest":
            markdown = str(arguments.get("markdown", "")).strip()
            title = str(arguments.get("title", "Untitled Document")).strip()
            source = str(arguments.get("source", title)).strip()
            prefer_local = bool(arguments.get("prefer_local", True))
            if not markdown:
                raise ValueError("'markdown' cannot be empty.")

            doc_id = compute_document_id(markdown, source)
            self.chroma.prefer_local = prefer_local
            physical_path = self._save_physical_kb_file(doc_id, source, markdown)
            meta = {
                "document_id": doc_id,
                "title": title,
                "source_url": source if _is_url(source) else None,
                "file_path": None if _is_url(source) else source,
                "physical_file": physical_path,
            }
            emb_mode = self.chroma.ingest_document(document_id=doc_id, text=markdown, metadata=meta)
            chunks_count = len(self.chroma._chunk_text(markdown))

            payload = {
                "status": "success",
                "document_id": doc_id,
                "title": title,
                "embedding_mode": emb_mode,
                "chunks_indexed": chunks_count,
                "physical_file": physical_path,
                "coverage": {
                    "completeness": "complete",
                    "embedding_mode": emb_mode,
                    "chunks_indexed": chunks_count,
                },
                "residual_unknowns": [],
            }
            return {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}]}

        elif name == "goldpan_kb_search":
            query = str(arguments.get("query", "")).strip()
            if not query:
                raise ValueError("'query' is required.")
            n_results = int(arguments.get("n_results", 4))
            detail = arguments.get("detail", "summary")
            spotlight = bool(arguments.get("spotlight", False))

            hits = self.chroma.search_with_citations(query, n_results=n_results)
            residual_unknowns: List[str] = []
            if not self.api_key and self.chroma.gemini_collection.count() > 0:
                residual_unknowns.append(
                    "Cloud collection 'goldpan_kb_gemini' contains vectors but was skipped because GEMINI_API_KEY is not set; searched local E5 ONNX + SQLite FTS5 only."
                )
            mig = self.chroma.check_migration_status()
            if mig.get("needs_migration"):
                residual_unknowns.append(mig["warning"])

            formatted_hits = []
            for h in hits:
                text = h["text"]
                truncated = False
                if detail == "summary" and len(text) > 800:
                    text = text[:800] + "..."
                    truncated = True
                if spotlight:
                    text = spotlight_untrusted_content(text, source=str(h["source"]))
                formatted_hits.append(
                    {
                        "title": h["title"],
                        "source": h["source"],
                        "embedding_mode": h["embedding_mode"],
                        "distance": round(h["distance"], 4),
                        "rrf_score": h.get("rrf_score", 0.0),
                        "truncated": truncated,
                        "text": text,
                    }
                )

            meta_obj = build_provenance_meta(FirewallReport(mode="sanitize"), source="goldpan_kb")
            payload = {
                "query": query,
                "detail": detail,
                "results_count": len(formatted_hits),
                "results": formatted_hits,
                "coverage": {
                    "completeness": "partial" if residual_unknowns else "complete",
                    "local_collection_chunks": self.chroma.local_collection.count(),
                    "gemini_collection_chunks": self.chroma.gemini_collection.count(),
                    "sqlite_fts5_chunks": self.chroma.fts_index.count() if self.chroma.fts_index else 0,
                },
                "residual_unknowns": residual_unknowns,
                "_meta": meta_obj,
            }
            return {
                "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}],
                "_meta": payload["_meta"],
            }

        elif name == "goldpan_kb_list":
            docs = self.chroma.get_all_documents()
            payload = {
                "total_documents": len(docs),
                "documents": docs,
                "coverage": {
                    "completeness": "complete",
                    "local_collection_chunks": self.chroma.local_collection.count(),
                    "gemini_collection_chunks": self.chroma.gemini_collection.count(),
                },
                "residual_unknowns": [],
            }
            return {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}]}

        elif name == "goldpan_kb_delete":
            doc_id = str(arguments.get("document_id", "")).strip()
            if not doc_id:
                raise ValueError("'document_id' is required.")
            for doc in self.chroma.get_all_documents():
                if doc.get("id") == doc_id and doc.get("physical_file"):
                    p = doc["physical_file"]
                    if os.path.exists(p):
                        os.remove(p)
                    break
            self.chroma.delete_document(doc_id)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"status": "success", "deleted_document_id": doc_id}, indent=2),
                    }
                ]
            }

        elif name == "goldpan_kb_audit":
            audit_res = self.chroma.audit_kb()
            return {"content": [{"type": "text", "text": json.dumps(audit_res, ensure_ascii=False, indent=2)}]}

        elif name == "goldpan_doctor":
            diag = collect_doctor_diagnostics(persist_dir=self.chroma.persist_directory)
            return {"content": [{"type": "text", "text": json.dumps(diag, ensure_ascii=False, indent=2)}]}

        elif name == "goldpan_ocr":
            source = str(arguments.get("source", "")).strip()
            if not source or "\x00" in source:
                raise ValueError("Invalid 'source' parameter.")
            source = os.path.abspath(source)
            if not os.path.exists(source):
                raise FileNotFoundError(f"File not found: {source}")

            from .ocr.engine import VietnameseOfflineOCR

            ocr_engine = VietnameseOfflineOCR()
            ext = os.path.splitext(source)[1].lower()
            dpi = int(arguments.get("dpi", 200))
            spotlight = bool(arguments.get("spotlight", False))

            if ext == ".pdf":
                res = ocr_engine.extract_from_pdf(source, dpi=dpi)
                text = res.text
                meta_info = {
                    "is_scanned": res.is_scanned,
                    "avg_confidence": res.avg_confidence,
                    "backend": res.backend,
                    "pages_count": len(res.pages),
                    "duration_ms": res.total_duration_ms,
                }
            else:
                p_res = ocr_engine.extract_from_image(source)
                text = p_res.text
                meta_info = {
                    "confidence": p_res.confidence,
                    "backend": p_res.backend,
                    "lines_count": p_res.lines_count,
                    "duration_ms": p_res.duration_ms,
                }

            if spotlight:
                text = spotlight_untrusted_content(text, source=source)

            payload = {
                "source": source,
                "text": text,
                "ocr": meta_info,
                "_meta": build_provenance_meta(FirewallReport(mode="sanitize"), source=source),
            }
            return {
                "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}],
                "_meta": payload["_meta"],
            }

        else:
            raise ValueError(f"Unknown tool: {name}")

    def handle_request(self, req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method = req.get("method")
        req_id = req.get("id")
        params = req.get("params") or {}

        if req_id is None:
            return None

        try:
            if method == "initialize":
                result = {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {"subscribe": False, "listChanged": False},
                    },
                    "serverInfo": {
                        "name": "goldpan-mcp",
                        "version": __version__,
                    },
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": self.get_tools()}
            elif method == "tools/call":
                tool_name = params.get("name", "")
                tool_args = params.get("arguments") or {}
                result = self.call_tool(tool_name, tool_args)
            elif method == "resources/list":
                result = {"resources": self.get_resources()}
            elif method == "resources/read":
                uri = params.get("uri", "")
                result = self.read_resource(uri)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }

            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps({"error": str(e)}, ensure_ascii=False),
                        }
                    ],
                    "isError": True,
                },
            }


def main() -> None:
    """Run GoldPan MCP Server over stdio (supports both newline-delimited and Content-Length JSON-RPC)."""
    server = GoldPanMCPServer()
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    while True:
        line = stdin.readline()
        if not line:
            break

        stripped = line.strip()
        if not stripped:
            continue

        use_content_length = False
        if stripped.lower().startswith(b"content-length:"):
            use_content_length = True
            length = int(stripped.split(b":", 1)[1].strip())
            while True:
                hdr = stdin.readline()
                if not hdr or hdr.strip() == b"":
                    break
            body = stdin.read(length)
        else:
            body = stripped

        try:
            req = json.loads(body.decode("utf-8"))
        except Exception:
            continue

        resp = server.handle_request(req)
        if resp is not None:
            out_bytes = json.dumps(resp, ensure_ascii=False).encode("utf-8")
            if use_content_length:
                header = f"Content-Length: {len(out_bytes)}\r\n\r\n".encode("ascii")
                stdout.write(header + out_bytes)
            else:
                stdout.write(out_bytes + b"\n")
            stdout.flush()


if __name__ == "__main__":
    main()
