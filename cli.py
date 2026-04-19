"""Command-line interface for Airllama."""

import asyncio
import sys
import logging
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from app.config import settings
from app.services.model_manager import model_manager

# Initialize CLI app and console
app = typer.Typer(
    name="airllama",
    help="Airllama - Run large language models locally (Ollama-compatible CLI)",
    no_args_is_help=True,
)
console = Console()

# Configure logging — quiet by default, verbose only when AIRLLAMA_DEBUG is set
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(11434, help="Port to listen on"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload on file changes"),
):
    """Start the Airllama server.

    Examples:

        airllama serve

        airllama serve --host 127.0.0.1 --port 8080
    """
    import uvicorn

    console.print(f"[bold green]Airllama[/bold green] server listening on [bold]{host}:{port}[/bold]")
    console.print(f"  Cache directory : {settings.cache_path}")
    console.print(f"  Compression     : {settings.default_compression or 'none'}\n")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


# ---------------------------------------------------------------------------
# pull
# ---------------------------------------------------------------------------

@app.command()
def pull(
    model: str = typer.Argument(..., help="HuggingFace model ID to pull"),
):
    """Pull a model from HuggingFace and cache it locally.

    Examples:

        airllama pull mistralai/Mistral-7B-Instruct-v0.2

        airllama pull microsoft/phi-2

        airllama pull mistralai/Mistral-7B-Instruct-v0.2
    """
    console.print(f"pulling manifest")

    try:
        model_manager._load_registry()
        asyncio.run(_pull_model(model))
        console.print(f"[bold green]✓[/bold green] {model}")
    except Exception as e:
        console.print(f"[bold red]error[/bold red] {e}")
        raise typer.Exit(code=1)


async def _pull_model(model_name: str):
    async for progress in model_manager.pull_model(model_name, stream_progress=True):
        status = progress.get("status", "")
        if status == "error":
            raise Exception(progress.get("error", "Unknown error"))
        elif status and status not in ("success",):
            console.print(status)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@app.command(name="list")
def list_models():
    """List models that are available locally.

    Examples:

        airllama list
    """
    try:
        model_manager._load_registry()
        models = model_manager.list_models()

        if not models:
            console.print("No models found. Pull one with: [bold]airllama pull <model>[/bold]")
            return

        table = Table(show_header=True, header_style="bold")
        table.add_column("NAME")
        table.add_column("ARCHITECTURE")
        table.add_column("SIZE")
        table.add_column("COMPRESSION")
        table.add_column("MODIFIED")

        for m in models:
            table.add_row(
                m.name,
                m.architecture,
                m.parameter_size,
                m.quantization_level or "none",
                m.modified_at[:10] if m.modified_at else "",
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# rm
# ---------------------------------------------------------------------------

@app.command()
def rm(
    model: str = typer.Argument(..., help="Model name to remove"),
):
    """Remove a model from local storage.

    Examples:

        airllama rm mistralai/Mistral-7B-Instruct-v0.2
    """
    try:
        model_manager._load_registry()
        asyncio.run(model_manager.delete_model(model))
        console.print(f"deleted '{model}'")
    except Exception as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------

@app.command()
def show(
    model: str = typer.Argument(..., help="Model name to inspect"),
):
    """Show information for a model.

    Examples:

        airllama show mistralai/Mistral-7B-Instruct-v0.2
    """
    try:
        model_manager._load_registry()
        meta = model_manager.registry.get(model)

        if meta is None:
            console.print(f"[red]error:[/red] model '{model}' not found locally. Pull it first.")
            raise typer.Exit(code=1)

        console.print(f"\n  [bold]Model[/bold]         {meta.name}")
        console.print(f"  [bold]Architecture[/bold]  {meta.architecture}")
        console.print(f"  [bold]Parameters[/bold]    {meta.parameter_size}")
        console.print(f"  [bold]Compression[/bold]   {meta.quantization_level or 'none'}")
        console.print(f"  [bold]Format[/bold]        {meta.format}")
        console.print(f"  [bold]Families[/bold]      {', '.join(meta.families) if meta.families else '-'}")
        console.print(f"  [bold]Modified[/bold]      {meta.modified_at[:10] if meta.modified_at else '-'}")
        console.print()

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# ps  (running / loaded models)
# ---------------------------------------------------------------------------

@app.command()
def ps():
    """List models currently loaded in memory.

    Examples:

        airllama ps
    """
    try:
        loaded = model_manager.loaded_models

        if not loaded:
            console.print("No models currently loaded.")
            return

        table = Table(show_header=True, header_style="bold")
        table.add_column("NAME")
        table.add_column("LOADED AT")
        table.add_column("LAST USED")
        table.add_column("COMPRESSION")

        for name, info in loaded.items():
            table.add_row(
                name,
                info.get("loaded_at", "")[:19],
                info.get("last_used", "")[:19],
                info.get("compression") or "none",
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@app.command()
def run(
    model: str = typer.Argument(..., help="Model to run"),
    prompt: Optional[str] = typer.Argument(None, help="Prompt to send (omit for interactive mode)"),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Maximum tokens to generate"),
):
    """Run a model and generate a response.

    Examples:

        airllama run mistralai/Mistral-7B-Instruct-v0.2 "Why is the sky blue?"

        airllama run microsoft/phi-2 "Write a poem about autumn"
    """
    if prompt is None:
        # Interactive REPL with multi-turn history
        console.print(f"[bold green]Airllama[/bold green] — [bold]{model}[/bold]")
        console.print("Type [bold]/bye[/bold] or Ctrl-C to exit.\n")
        history: list = []
        try:
            while True:
                user_input = typer.prompt(">>>")
                if user_input.strip().lower() in ("/bye", "/exit", "/quit"):
                    break
                history.append({"role": "user", "content": user_input})
                response = asyncio.run(_run_model(model, list(history), max_tokens))
                if response:
                    history.append({"role": "assistant", "content": response})
        except (KeyboardInterrupt, EOFError):
            console.print("\nBye!")
    else:
        try:
            asyncio.run(_run_model(model, [{"role": "user", "content": prompt}], max_tokens))
        except Exception as e:
            console.print(f"[red]error:[/red] {e}")
            raise typer.Exit(code=1)


async def _run_model(model_name: str, messages: list, max_tokens: int) -> str:
    from app.services.generation import generation_service

    model_info = await model_manager.get_model(model_name)
    model_obj = model_info["model"]
    tokenizer = model_info["tokenizer"]

    console.print()
    parts: list = []
    async for chunk in generation_service.stream_chat_completion(
        model=model_obj,
        tokenizer=tokenizer,
        messages=messages,
        max_new_tokens=max_tokens,
    ):
        token = chunk.get("token", "")
        if token:
            console.print(token, end="", highlight=False)
            parts.append(token)
    console.print("\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main():
    app()


if __name__ == "__main__":
    main()

