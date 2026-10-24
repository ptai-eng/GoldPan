"""GoldPan AI Command-Line Interface (CLI) powered by Typer & Rich."""

import json
import os
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .mcp_server import GoldPanMCPServer

app = typer.Typer(
    name="goldpan",
    help="GoldPan AI v2.0 — Privacy-First Multimodal Data Refinery, Local ONNX RAG & MCP Engine.",
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
    as_json: bool = typer.Option(False, "--json", help="Emit structured JSON envelope with coverage & provenance."),
):
    """Extract clean AI-ready Markdown from any file or URL."""
    server = GoldPanMCPServer()
    res = server.call_tool(
        "goldpan_extract",
        {
            "source": source,
            "detail": "complete" if output else detail,
            "max_chars": max_chars,
            "ingest_to_kb": ingest,
            "use_playwright_fallback": not no_playwright,
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
    table = Table(title=f"GoldPan Refinery Telemetry — {payload.get('title') or source}", show_header=True)
    table.add_column("Metric", style="bold yellow")
    table.add_column("Value", style="white")
    table.add_row("Router Mode", str(stats.get("router_mode", "local_heuristic")))
    table.add_row("Heuristic Noise Stripped", f"{stats.get('heuristic_noise_removed_bytes', 0)} bytes")
    table.add_row("Semantic Noise Stripped", f"{stats.get('semantic_noise_removed_bytes', 0)} bytes")
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
    local: bool = typer.Option(True, "--local/--cloud", help="Use 100% offline Local ONNX embeddings or Gemini Cloud."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON output."),
):
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
    query: str = typer.Argument(..., help="Semantic search query."),
    top_k: int = typer.Option(4, "--top-k", "-k", help="Number of top passages to retrieve."),
    detail: str = typer.Option("summary", "--detail", "-d", help="'summary' or 'complete'."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON output."),
):
    """Search the local Knowledge Base using semantic vector similarity."""
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
        title = f"[{idx}] {h['title']} ({h['source']}) · mode={h['embedding_mode']} · dist={h['distance']}"
        console.print(Panel(h["text"], title=title, border_style="yellow"))


@app.command("kb-list")
def kb_list_cmd(as_json: bool = typer.Option(False, "--json", help="Emit JSON output.")):
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
def kb_delete_cmd(document_id: str = typer.Argument(..., help="Document UUID to delete from the Knowledge Base.")):
    """Delete a document from ChromaDB and remove its physical KB/*.md mirror."""
    server = GoldPanMCPServer()
    server.call_tool("goldpan_kb_delete", {"document_id": document_id})
    console.print(f"[bold green]✔ Deleted document[/bold green] [cyan]{document_id}[/cyan]")


@app.command("mcp")
def mcp_cmd():
    """Start the GoldPan Model Context Protocol (MCP) server over stdio."""
    from .mcp_server import main as run_mcp
    run_mcp()


@app.command("version")
def version_cmd():
    """Show GoldPan AI version."""
    typer.echo(f"GoldPan AI v{__version__}")


if __name__ == "__main__":
    app()
