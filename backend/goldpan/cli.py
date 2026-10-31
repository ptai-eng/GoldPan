"""GoldPan AI v3.0 Command-Line Interface (CLI) powered by Typer & Rich."""

import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .bench.runner import GoldPanBenchRunner
from .kb.chroma_manager import ChromaManager
from .kb.onnx_embedder import collect_doctor_diagnostics
from .mcp_server import GoldPanMCPServer

app = typer.Typer(
    name="goldpan",
    help="GoldPan AI v3.2 — Privacy-First Multimodal Data Refinery, Playwright Tier-2 Firewall, Offline Vietnamese OCR, Multilingual E5 ONNX RAG & MCP Engine.",
    add_completion=False,
)
console = Console()


@app.command("extract")
def extract_cmd(
    source: str = typer.Argument(..., help="URL (Web, YouTube, GitHub blob) or local file path to extract."),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Write extracted Markdown to a file."),
    detail: str = typer.Option("summary", "--detail", "-d", help="Detail level: 'summary' or 'complete'."),
    max_chars: int = typer.Option(4000, "--max-chars", help="Character cap when detail='summary'."),
    ingest: bool = typer.Option(False, "--ingest", "-i", help="Also ingest extracted Markdown into the Knowledge Base."),
    no_playwright: bool = typer.Option(False, "--no-playwright", help="Disable headless Playwright fallback for URLs."),
    firewall_mode: str = typer.Option("sanitize", "--firewall-mode", help="Firewall mode: 'sanitize', 'audit', or 'strict'."),
    ocr: bool = typer.Option(True, "--ocr/--no-ocr", help="Enable offline OCR for scanned PDFs and image files."),
    spotlight: bool = typer.Option(False, "--spotlight", help="Wrap output in <untrusted_document_content> tags."),
    as_json: bool = typer.Option(False, "--json", help="Emit structured JSON envelope with coverage & provenance."),
) -> None:
    """Extract clean AI-ready Markdown from any file or URL with Agent Context Firewall protection."""
    server = GoldPanMCPServer()
    res = server.call_tool(
        "goldpan_extract",
        {
            "source": source,
            "detail": "complete" if output else detail,
            "max_chars": max_chars,
            "ingest_to_kb": ingest,
            "use_playwright_fallback": not no_playwright,
            "firewall_mode": firewall_mode,
            "enable_ocr": ocr,
            "spotlight": spotlight,
        },
    )
    payload = json.loads(res["content"][0]["text"])

    if output:
        out_path = os.path.abspath(output)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(payload["markdown"])

    if as_json:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    stats = payload.get("stats", {})
    fw = payload.get("firewall", {})
    table = Table(title=f"GoldPan Refinery Telemetry — {payload.get('title') or source}", show_header=True)
    table.add_column("Metric", style="bold yellow")
    table.add_column("Value", style="white")
    table.add_row("Router Mode", str(stats.get("router_mode", "local_heuristic")))
    table.add_row("Detected Encoding", str(stats.get("detected_encoding", "unicode_nfc")))
    table.add_row("Heuristic Noise Stripped", f"{stats.get('heuristic_noise_removed_bytes', 0)} bytes")
    table.add_row("Semantic Noise Stripped", f"{stats.get('semantic_noise_removed_bytes', 0)} bytes")
    table.add_row(
        "Context Firewall",
        f"mode={fw.get('mode', 'sanitize')} | risk={fw.get('risk_score', 0.0)} | signals={fw.get('signals', [])}",
    )
    table.add_row("Total Clean Characters", str(stats.get("total_chars", 0)))
    table.add_row("Completeness", payload.get("coverage", {}).get("completeness", "complete"))
    if payload.get("ingested"):
        ing = payload["ingested"]
        table.add_row("KB Ingested ID", f"{ing['document_id']} ({ing['embedding_mode']})")
    if output:
        table.add_row("Saved To", os.path.abspath(output))

    console.print(table)
    if not output:
        console.print(Panel(payload["markdown"], title="Refined Markdown Output", border_style="yellow"))


@app.command("ingest")
def ingest_cmd(
    source: str = typer.Argument(..., help="File path or URL to extract and ingest into the Knowledge Base."),
    local: bool = typer.Option(True, "--local/--cloud", help="Use 100% offline Local E5 ONNX embeddings or Gemini Cloud."),
    firewall_mode: str = typer.Option("sanitize", "--firewall-mode", help="Firewall mode: 'sanitize', 'audit', or 'strict'."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Extract and ingest a document or URL directly into the Dual-Mode Knowledge Base."""
    server = GoldPanMCPServer()
    server.chroma.prefer_local = local
    res = server.call_tool(
        "goldpan_extract",
        {
            "source": source,
            "detail": "summary",
            "max_chars": 500,
            "ingest_to_kb": True,
            "firewall_mode": firewall_mode,
        },
    )
    payload = json.loads(res["content"][0]["text"])
    if as_json:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    ing = payload.get("ingested") or {}
    console.print(
        f"[bold green]✔ Ingested[/bold green] [yellow]{payload.get('title')}[/yellow] "
        f"(ID: [cyan]{ing.get('document_id')}[/cyan], Mode: [magenta]{ing.get('embedding_mode')}[/magenta])"
    )
    console.print(f"  Physical Mirror: {ing.get('physical_file')}")


@app.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="Multilingual/Vietnamese hybrid search query."),
    top_k: int = typer.Option(4, "--top-k", "-k", help="Number of top passages to retrieve."),
    detail: str = typer.Option("summary", "--detail", "-d", help="'summary' or 'complete'."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Search the local Knowledge Base using Hybrid RRF (Multilingual E5 ONNX + SQLite FTS5)."""
    server = GoldPanMCPServer()
    res = server.call_tool(
        "goldpan_kb_search",
        {"query": query, "n_results": top_k, "detail": detail},
    )
    payload = json.loads(res["content"][0]["text"])
    if as_json:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    hits = payload.get("results", [])
    if not hits:
        console.print("[yellow]No matching passages found in Knowledge Base.[/yellow]")
        return

    for idx, h in enumerate(hits, 1):
        title = f"[{idx}] {h['title']} ({h['source']}) · mode={h['embedding_mode']} · rrf={h.get('rrf_score', 0)}"
        console.print(Panel(h["text"], title=title, border_style="yellow"))


@app.command("kb-list")
def kb_list_cmd(as_json: bool = typer.Option(False, "--json", help="Emit JSON output.")) -> None:
    """List all documents currently indexed in the Knowledge Base."""
    server = GoldPanMCPServer()
    res = server.call_tool("goldpan_kb_list", {})
    payload = json.loads(res["content"][0]["text"])
    if as_json:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    docs = payload.get("documents", [])
    table = Table(title=f"GoldPan Knowledge Base ({len(docs)} documents)", show_header=True)
    table.add_column("Document ID", style="cyan")
    table.add_column("Title", style="bold white")
    table.add_column("Source", style="dim")
    table.add_column("Embedding Mode", style="yellow")
    for d in docs:
        table.add_row(str(d.get("id")), str(d.get("title")), str(d.get("source")), str(d.get("embedding_mode")))
    console.print(table)


@app.command("kb-delete")
def kb_delete_cmd(document_id: str = typer.Argument(..., help="Document UUID to delete from the Knowledge Base.")) -> None:
    """Delete a document from ChromaDB and remove its physical KB/*.md mirror."""
    server = GoldPanMCPServer()
    server.call_tool("goldpan_kb_delete", {"document_id": document_id})
    console.print(f"[bold green]✔ Deleted document[/bold green] [cyan]{document_id}[/cyan]")


@app.command("reindex")
def reindex_cmd(
    kb_dir: Optional[str] = typer.Option(None, "--kb-dir", help="Optional path to KB markdown directory."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Migrate/re-index Markdown documents into the v3 Multilingual E5 + SQLite FTS5 Knowledge Base."""
    kb = ChromaManager()
    summary = kb.reindex_from_kb_dir(kb_dir=kb_dir)
    if as_json:
        typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    console.print(
        f"[bold green]✔ Re-indexed Knowledge Base into[/bold green] [cyan]{summary['collection']}[/cyan] "
        f"(Files: {summary['files_indexed']}, v3 Chunks: {summary['total_v3_chunks']}, FTS5 Chunks: {summary['fts5_chunks']})"
    )


@app.command("audit")
def audit_cmd(
    as_json: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Audit all chunks currently in the Knowledge Base for prompt injection signatures."""
    kb = ChromaManager()
    report = kb.audit_kb()
    if as_json:
        typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
        return

    table = Table(
        title=f"GoldPan KB Security Audit — {report['collection']} ({report['total_chunks_scanned']} chunks scanned)",
        show_header=True,
    )
    table.add_column("Chunk ID", style="cyan")
    table.add_column("Source", style="white")
    table.add_column("Risk Score", style="bold red")
    table.add_column("Signals", style="yellow")

    if not report["findings"]:
        console.print(f"[bold green]✔ Knowledge Base Clean[/bold green] ({report['total_chunks_scanned']} chunks scanned, 0 suspicious).")
        return

    for f in report["findings"]:
        table.add_row(str(f["chunk_id"]), str(f["source"]), str(f["risk_score"]), ", ".join(f["signals"]))
    console.print(table)


@app.command("doctor")
def doctor_cmd(
    device: Optional[str] = typer.Option(None, "--device", help="Override device selection ('auto', 'cpu', 'directml', 'cuda')."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON output."),
) -> None:
    """Run hardware, ONNX Runtime Execution Provider, SQLite FTS5, and v2->v3 collection diagnostics."""
    diag = collect_doctor_diagnostics(device_override=device)
    if as_json:
        typer.echo(json.dumps(diag, ensure_ascii=False, indent=2))
        return

    table = Table(title="GoldPan AI v3.2 — System & Hardware Doctor", show_header=True)
    table.add_column("Diagnostic Check", style="bold cyan")
    table.add_column("Status / Value", style="white")
    table.add_row("Operating System", str(diag["os"]))
    table.add_row("Python / ONNX Runtime", f"Python {diag['python_version']} | ORT {diag['onnxruntime_version']}")
    table.add_row("Available ORT Providers", ", ".join(diag["available_providers"]))
    table.add_row("Selected GPU Tier", str(diag["selected_tier"]))
    table.add_row("Active Execution Provider", f"{diag['active_provider']} ({diag['precision']}, batch_size={diag['batch_size']})")
    table.add_row("Embedding Model", f"{diag['embedding_model']} ({diag['embedding_dimension']}d)")
    table.add_row("Warmup / Sample Embed Latency", f"{diag['warmup_ms']} ms / {diag['sample_embed_ms']} ms")
    table.add_row("SQLite FTS5 Index", f"{diag['sqlite_fts5_db']} (exists={diag['sqlite_fts5_exists']})")
    table.add_row("Chroma Collections", f"v3={diag['v3_e5_chunks']} chunks | v2_legacy={diag['v2_legacy_chunks']} chunks")
    if diag.get("fallback_triggered"):
        table.add_row("Fallback Notice", f"[yellow]{diag['fallback_reason']}[/yellow]")
    if diag.get("migration_recommended"):
        table.add_row("Migration Advisory", "[bold yellow]Run `goldpan reindex` to migrate v2 chunks to v3 E5.[/bold yellow]")

    console.print(table)


@app.command("ocr")
def ocr_cmd(
    source: str = typer.Argument(..., help="Path to scanned PDF or image file (.png, .jpg, .webp)."),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Write OCR text to a file."),
    dpi: int = typer.Option(200, "--dpi", help="PDF page rasterization DPI (150-300)."),
    spotlight: bool = typer.Option(False, "--spotlight", help="Wrap output in <untrusted_document_content> tags."),
    as_json: bool = typer.Option(False, "--json", help="Emit structured JSON with telemetry."),
) -> None:
    """Run offline Vietnamese and multilingual OCR on scanned PDFs or image files."""
    server = GoldPanMCPServer()
    res = server.call_tool("goldpan_ocr", {"source": source, "dpi": dpi, "spotlight": spotlight})
    payload = json.loads(res["content"][0]["text"])

    if output:
        out_path = os.path.abspath(output)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(payload["text"])

    if as_json:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    ocr_meta = payload.get("ocr", {})
    table = Table(title=f"GoldPan Offline OCR — {source}", show_header=True)
    table.add_column("Property", style="bold cyan")
    table.add_column("Value", style="white")
    table.add_row("OCR Backend", str(ocr_meta.get("backend", "none")))
    conf_val = ocr_meta.get("avg_confidence") if ocr_meta.get("avg_confidence") is not None else ocr_meta.get("confidence", 0.0)
    table.add_row("Confidence", f"{float(conf_val):.2f}")
    if "is_scanned" in ocr_meta:
        table.add_row("Scanned Document", str(ocr_meta.get("is_scanned")))
    if "pages_count" in ocr_meta:
        table.add_row("Pages Processed", str(ocr_meta.get("pages_count")))
    table.add_row("Duration", f"{ocr_meta.get('duration_ms', 0):.2f} ms")
    if output:
        table.add_row("Saved To", os.path.abspath(output))

    console.print(table)
    if not output:
        console.print(Panel(payload["text"], title="Extracted OCR Markdown", border_style="yellow"))


@app.command("bench")
def bench_cmd(
    device: Optional[str] = typer.Option(None, "--device", help="Device override ('auto', 'cpu', 'directml')."),
    output_md: Optional[str] = typer.Option(None, "--output-md", help="Write Markdown benchmark report to file."),
    output_json: Optional[str] = typer.Option(None, "--output-json", help="Write JSON benchmark results to file."),
    ci_gate: bool = typer.Option(False, "--ci-gate", help="Enforce v3.0 quality gates (exit code 1 on regression)."),
    as_json: bool = typer.Option(False, "--json", help="Emit raw JSON results to stdout."),
) -> None:
    """Run GoldPan Bench (Track A Firewall Security + Track B Multilingual Retrieval with 95% Wilson CIs)."""
    runner = GoldPanBenchRunner()
    results = runner.run_all(device_override=device)
    md_report = runner.format_markdown_report(results)

    if output_md:
        Path(output_md).write_text(md_report, encoding="utf-8")
    if output_json:
        Path(output_json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    if as_json:
        typer.echo(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        console.print(Panel(md_report, title="GoldPan Bench v3.0 Evaluation Report", border_style="green"))

    if ci_gate:
        tb = results["track_b"]["systems"]
        ta = results["track_a"]
        v3_r3 = tb["v3_0_e5_fts5_rrf"]["recall_at_3"]
        v2_r3 = tb["v2_1_minilm_rrf"]["recall_at_3"]
        tpr = ta["detection_tpr"]
        san = ta["sanitization_rate"]
        fpr = ta["hard_negative_fpr"]

        failures: list[str] = []
        if v3_r3 < 0.85 or v3_r3 <= v2_r3:
            failures.append(f"Track B Recall@3 gate failed: v3.0={v3_r3:.3f} vs v2.1={v2_r3:.3f}")
        if tpr < 0.90:
            failures.append(f"Track A Detection TPR gate failed: {tpr:.3f} < 0.90")
        if san < 0.90:
            failures.append(f"Track A Sanitization Rate gate failed: {san:.3f} < 0.90")
        if fpr > 0.05:
            failures.append(f"Track A Hard-Negative FPR gate failed: {fpr:.3f} > 0.05")

        if failures:
            for msg in failures:
                console.print(f"[bold red]✘ CI Gate Failure:[/bold red] {msg}")
            raise typer.Exit(code=1)


@app.command("mcp")
def mcp_cmd() -> None:
    """Start the GoldPan Model Context Protocol (MCP) server over stdio."""
    from .mcp_server import main as run_mcp

    run_mcp()


@app.command("version")
def version_cmd() -> None:
    """Show GoldPan AI version."""
    typer.echo(f"GoldPan AI v{__version__}")


if __name__ == "__main__":
    app()
