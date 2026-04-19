"""Main FastAPI application."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from app.config import settings
from app.routers import ollama, openai_compat
from app.services.model_manager import model_manager
from app import __version__

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Airllama server...")
    logger.info(f"Version: {__version__}")
    logger.info(f"Model cache directory: {settings.cache_path}")
    logger.info(f"Default compression: {settings.default_compression}")
    
    # Initialize model manager
    model_manager._load_registry()
    logger.info("Model manager initialized")
    
    yield
    
    logger.info(f"Shutting down Airllama server...")


# Create FastAPI app
app = FastAPI(
    title="Airllama API",
    description="Ollama-compatible API for running LLMs locally via AirLLM",
    version=__version__,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ollama.router)
app.include_router(openai_compat.router)


@app.get("/", response_class=PlainTextResponse)
async def root():
    """Root endpoint - Ollama health check."""
    return "Ollama is running"


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": __version__,
        "models_loaded": len(model_manager.loaded_models),
        "models_registered": len(model_manager.registry),
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
