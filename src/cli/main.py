import typer
from typing import Optional
from pathlib import Path

from src.agents.teaching_agent import TeachingAgent

app = typer.Typer(name="smit-teacher", help="SMIT Teaching Agent — RAG-powered educational assistant")
_agent: Optional[TeachingAgent] = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = TeachingAgent()
    return _agent


@app.command()
def ask(
    query: str = typer.Argument(..., help="Your question about SMIT courses, concepts, or syllabus"),
    session: str = typer.Option("default", "--session", "-s", help="Session ID for conversation memory"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass response cache"),
):
    """Ask a question to the SMIT Teaching Agent."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown

    console = Console()
    with console.status("[bold green]Thinking...", spinner="dots"):
        result = agent.ask(query, session_id=session)

    md = Markdown(result.answer)
    confidence_color = "green" if result.confidence > 0.7 else "yellow" if result.confidence > 0.4 else "red"
    mode_icon = {"teaching": "📚", "syllabus": "📋", "assignment": "✏️", "general": "💡", "greeting": "👋"}

    title = f"{mode_icon.get(result.mode, '💡')} {result.mode.title()} | Confidence: [{confidence_color}]{result.confidence:.2f}[/] | Model: {result.model}"
    if result.from_cache:
        title += " [dim](cached)[/]"

    console.print(Panel(md, title=title, border_style="blue"))

    if result.sources:
        table = Table(title="Sources", show_header=True, header_style="bold cyan")
        table.add_column("#", style="dim")
        table.add_column("Source", style="cyan")
        table.add_column("Score", justify="right")
        for i, s in enumerate(result.sources[:5], 1):
            table.add_row(str(i), s.get("filename", "unknown"), f"{s.get('score', 0):.3f}")
        console.print(table)


@app.command()
def chat(
    session: str = typer.Option("default", "--session", "-s", help="Session ID"),
):
    """Start an interactive chat session with the SMIT Teaching Agent."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown

    console = Console()
    console.print(Panel("[bold blue]SMIT Teaching Agent[/]\nType 'exit' or 'quit' to end the session.\nType 'clear' to reset conversation history.", title_align="left"))

    while True:
        query = typer.prompt("You", default="")
        if query.lower() in ("exit", "quit"):
            break
        if query.lower() == "clear":
            agent.clear_history(session)
            console.print("[yellow]Conversation history cleared.[/]")
            continue
        if not query.strip():
            continue

        with console.status("[bold green]Thinking...", spinner="dots"):
            result = agent.ask(query, session_id=session)

        md = Markdown(result.answer)
        console.print(Panel(md, border_style="green"))
        if result.sources:
            sources_str = " | ".join(f"[cyan]{s.get('filename', 'unknown')}[/] ({s.get('score', 0):.2f})" for s in result.sources[:3])
            console.print(f"[dim]Sources: {sources_str}[/]")
        console.print()


@app.command()
def ingest(
    path: str = typer.Argument(..., help="File or directory path to ingest"),
    recursive: bool = typer.Option(True, "--recursive", "-r", help="Recursively scan directories"),
):
    """Ingest documents into the vector database."""
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()
    p = Path(path)
    if not p.exists():
        console.print(f"[red]Error: Path '{path}' does not exist[/]")
        raise typer.Exit(code=1)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task(description="[green]Ingesting documents...", total=None)
        result = agent.ingest(path)

    console.print(f"[green]✓[/] Ingested [bold]{result['files']}[/] files → [bold]{result['chunks']}[/] chunks")
    console.print(f"[dim]  Total documents: {result['documents']}[/]")
    console.print(f"[dim]  Total chunks in DB: {agent.pipeline.retriever.store.count()}[/]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind address"),
    port: int = typer.Option(8000, "--port", "-p", help="Port number"),
):
    """Start the FastAPI web server."""
    from rich.console import Console
    console = Console()
    console.print(f"[green]Starting SMIT Teaching Agent API on [bold]{host}:{port}[/]...[/]")
    import uvicorn
    uvicorn.run("src.api.server:app", host=host, port=port, reload=True)


@app.command()
def config():
    """Show current configuration."""
    from rich.console import Console
    from rich.table import Table
    from src.config import settings

    console = Console()
    table = Table(title="Current Configuration")
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    table.add_row("LLM Provider", settings.llm_provider)
    table.add_row("Embedding Model", settings.embedding_model)
    table.add_row("ChromaDB Path", str(settings.chroma_persist_path))
    table.add_row("Log Level", settings.log_level)
    console.print(table)


@app.command()
def stats():
    """Show vector database statistics."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    count = agent.pipeline.retriever.store.count()
    table = Table(title="Vector Database Stats")
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    table.add_row("Total Chunks", str(count))
    table.add_row("Collection Name", "smit_docs")
    table.add_row("Vector DB Type", "ChromaDB")
    table.add_row("Cache Directory", "data/cache")
    console.print(table)


if __name__ == "__main__":
    app()
