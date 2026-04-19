"""Command-line interface for AirLLM API."""

import asyncio
import sys
import logging
from typing import Optional

import typer
from rich.console import Console
from rich.spinner import Spinner

from app.config import settings
from app.services.model_manager import model_manager

# Initialize CLI app and console
app = typer.Typer(
    name="airllm",
    help="AirLLM - Lightweight LLM API with model management",
    no_args_is_help=True,
)
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@app.command()
def pull(model: str = typer.Argument(..., help="Model name to pull (e.g., TinyLlama/TinyLlama-1.1B-Chat-v1.0)")):
    """Pull and cache a model locally.
    
    Examples:
        airllm pull TinyLlama/TinyLlama-1.1B-Chat-v1.0
        airllm pull microsoft/phi-2
        airllm pull mistralai/Mistral-7B-Instruct-v0.2
    """
    console.print(f"[bold cyan]Pulling model:[/bold cyan] {model}")
    
    try:
        # Initialize model manager if needed
        model_manager._load_registry()
        
        # Run the async pull operation
        asyncio.run(_pull_model(model))
        
        console.print(f"[bold green]✓ Successfully pulled model: {model}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Failed to pull model: {str(e)}[/bold red]", style="red")
        raise typer.Exit(code=1)


async def _pull_model(model_name: str):
    """Internal async function to pull a model."""
    with console.status("[bold cyan]Loading model...", spinner="dots"):
        try:
            async for progress in model_manager.pull_model(model_name, stream_progress=True):
                if progress.get("status") == "error":
                    raise Exception(progress.get("error", "Unknown error"))
        except Exception as e:
            raise Exception(f"Model loading failed: {str(e)}")


@app.command()
def list_models():
    """List all cached models."""
    try:
        model_manager._load_registry()
        
        models = model_manager.list_models()
        
        if not models:
            console.print("[yellow]No models cached yet.[/yellow]")
            console.print("\nTo pull a model, use: [bold]airllm pull <model-name>[/bold]")
            return
        
        console.print("\n[bold cyan]Cached Models:[/bold cyan]\n")
        for model_meta in models:
            console.print(f"[bold]{model_meta.name}[/bold]")
            console.print(f"  Architecture: {model_meta.architecture}")
            console.print(f"  Size: {model_meta.parameter_size}")
            console.print(f"  Compression: {model_meta.quantization_level}")
            console.print()
            
    except Exception as e:
        console.print(f"[red]Error listing models: {str(e)}[/red]")
        raise typer.Exit(code=1)


@app.command()
def remove(model: str = typer.Argument(..., help="Model name to remove")):
    """Remove a cached model.
    
    Examples:
        airllm remove TinyLlama/TinyLlama-1.1B-Chat-v1.0
    """
    try:
        model_manager._load_registry()
        
        if model in model_manager.loaded_models:
            del model_manager.loaded_models[model]
            console.print(f"[bold green]✓ Removed model: {model}[/bold green]")
        else:
            console.print(f"[yellow]Model not found: {model}[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error removing model: {str(e)}[/red]")
        raise typer.Exit(code=1)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(11434, help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload on file changes"),
):
    """Start the AirLLM API server.
    
    Examples:
        airllm serve
        airllm serve --host 127.0.0.1 --port 8000
        airllm serve --reload
    """
    import uvicorn
    
    console.print(f"[bold cyan]Starting AirLLM API server...[/bold cyan]")
    console.print(f"  Host: {host}")
    console.print(f"  Port: {port}")
    console.print(f"  Auto-reload: {reload}")
    console.print(f"  Cache directory: {settings.cache_path}\n")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


@app.command()
def info():
    """Show AirLLM configuration and status."""
    try:
        model_manager._load_registry()
        models = model_manager.list_models()
        
        console.print("\n[bold cyan]AirLLM Configuration:[/bold cyan]\n")
        console.print(f"  Cache Directory: {settings.cache_path}")
        console.print(f"  Default Compression: {settings.default_compression}")
        console.print(f"  Max Loaded Models: {settings.max_loaded_models}")
        console.print(f"  Max Length: {settings.default_max_length}")
        console.print(f"  Max New Tokens: {settings.default_max_new_tokens}")
        
        console.print(f"\n[bold cyan]Status:[/bold cyan]\n")
        console.print(f"  Loaded Models: {len(model_manager.loaded_models)}")
        console.print(f"  Registered Models: {len(models)}")
        console.print()
        
    except Exception as e:
        console.print(f"[red]Error getting info: {str(e)}[/red]")
        raise typer.Exit(code=1)


def main():
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
