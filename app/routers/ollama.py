"""Ollama native API router."""

import logging
from datetime import datetime, timedelta
from typing import AsyncIterator
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import json

from app.schemas.ollama import (
    GenerateRequest, GenerateResponse, GenerateStreamResponse,
    ChatRequest, ChatResponse, ChatStreamResponse, ChatResponseMessage,
    EmbedRequest, EmbedResponse,
    EmbeddingsRequest, EmbeddingsResponse,
    ListModelsResponse, ModelInfo, ModelDetails,
    ShowModelRequest, ShowModelResponse,
    PullRequest, PullResponse,
    CopyRequest, DeleteRequest,
    ListRunningResponse, RunningModel,
    VersionResponse,
)
from app.services.model_manager import model_manager
from app.services.generation import generation_service
from app import __version__

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["ollama"])


@router.get("/")
async def root():
    """Health check endpoint."""
    return "Ollama is running"


@router.get("/version")
async def version() -> VersionResponse:
    """Get API version."""
    return VersionResponse(version=__version__)


@router.post("/generate")
async def generate(request: GenerateRequest):
    """
    Generate completion for a prompt.
    
    Supports both streaming and non-streaming responses.
    """
    try:
        # Get model
        model_info = await model_manager.get_model(request.model)
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]
        
        # Parse options
        options = request.options or {}
        max_new_tokens = options.get("num_predict", None)
        temperature = options.get("temperature", 0.8)
        top_p = options.get("top_p", 0.9)
        top_k = options.get("top_k", 40)
        seed = options.get("seed", None)
        stop = options.get("stop", None)
        
        # Validate parameters
        if max_new_tokens is not None and max_new_tokens <= 0:
            raise HTTPException(status_code=400, detail="num_predict must be positive")
        if not 0.0 <= temperature <= 2.0:
            raise HTTPException(status_code=400, detail="temperature must be between 0.0 and 2.0")
        if not 0.0 <= top_p <= 1.0:
            raise HTTPException(status_code=400, detail="top_p must be between 0.0 and 1.0")
        
        if isinstance(stop, str):
            stop = [stop]
        
        # Handle streaming
        if request.stream:
            async def generate_stream() -> AsyncIterator[str]:
                async with model_manager.generation_lock:
                    async for chunk in generation_service.stream_completion(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=request.prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        seed=seed,
                        stop=stop,
                    ):
                        response = GenerateStreamResponse(
                            model=request.model,
                            created_at=datetime.now().isoformat(),
                            response=chunk["token"],
                            done=chunk["done"],
                            **(chunk.get("stats", {}))
                        )
                        yield json.dumps(response.model_dump()) + "\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="application/x-ndjson"
            )
        else:
            # Non-streaming
            async with model_manager.generation_lock:
                result = await generation_service.generate_completion(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=request.prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed,
                    stop=stop,
                )
            
            response = GenerateResponse(
                model=request.model,
                created_at=datetime.now().isoformat(),
                response=result["text"],
                done=True,
                **result["stats"]
            )
            return response
            
    except Exception as e:
        logger.error(f"Generate failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Generate chat completion.
    
    Supports both streaming and non-streaming responses.
    """
    try:
        # Get model
        model_info = await model_manager.get_model(request.model)
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]
        
        # Convert messages to dict format
        messages = [msg.model_dump() for msg in request.messages]
        
        # Parse options
        options = request.options or {}
        max_new_tokens = options.get("num_predict", None)
        temperature = options.get("temperature", 0.8)
        top_p = options.get("top_p", 0.9)
        top_k = options.get("top_k", 40)
        seed = options.get("seed", None)
        stop = options.get("stop", None)
        
        # Validate parameters
        if max_new_tokens is not None and max_new_tokens <= 0:
            raise HTTPException(status_code=400, detail="num_predict must be positive")
        if not 0.0 <= temperature <= 2.0:
            raise HTTPException(status_code=400, detail="temperature must be between 0.0 and 2.0")
        if not 0.0 <= top_p <= 1.0:
            raise HTTPException(status_code=400, detail="top_p must be between 0.0 and 1.0")
        
        if isinstance(stop, str):
            stop = [stop]
        
        # Handle streaming
        if request.stream:
            async def chat_stream() -> AsyncIterator[str]:
                async with model_manager.generation_lock:
                    async for chunk in generation_service.stream_chat_completion(
                        model=model,
                        tokenizer=tokenizer,
                        messages=messages,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        seed=seed,
                        stop=stop,
                    ):
                        response = ChatStreamResponse(
                            model=request.model,
                            created_at=datetime.now().isoformat(),
                            message=ChatResponseMessage(
                                role="assistant",
                                content=chunk["token"]
                            ),
                            done=chunk["done"],
                            done_reason="stop" if chunk["done"] else None,
                            **(chunk.get("stats", {}))
                        )
                        yield json.dumps(response.model_dump()) + "\n"
            
            return StreamingResponse(
                chat_stream(),
                media_type="application/x-ndjson"
            )
        else:
            # Non-streaming
            async with model_manager.generation_lock:
                result = await generation_service.generate_chat_completion(
                    model=model,
                    tokenizer=tokenizer,
                    messages=messages,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=seed,
                    stop=stop,
                )
            
            response = ChatResponse(
                model=request.model,
                created_at=datetime.now().isoformat(),
                message=ChatResponseMessage(
                    role="assistant",
                    content=result["text"]
                ),
                done=True,
                done_reason="stop",
                **result["stats"]
            )
            return response
            
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embed")
async def embed(request: EmbedRequest) -> EmbedResponse:
    """Generate embeddings."""
    try:
        # Get model
        model_info = await model_manager.get_model(request.model)
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]
        
        # Ensure input is a list
        if isinstance(request.input, str):
            texts = [request.input]
        else:
            texts = request.input
        
        # Generate embeddings
        async with model_manager.generation_lock:
            embeddings = await generation_service.generate_embeddings(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
            )
        
        return EmbedResponse(
            model=request.model,
            embeddings=embeddings,
            prompt_eval_count=sum(len(tokenizer.encode(t)) for t in texts),
        )
        
    except Exception as e:
        logger.error(f"Embed failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embeddings")
async def embeddings(request: EmbeddingsRequest) -> EmbeddingsResponse:
    """Generate embeddings (legacy endpoint)."""
    try:
        # Get model
        model_info = await model_manager.get_model(request.model)
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]
        
        # Generate embedding
        async with model_manager.generation_lock:
            embeddings = await generation_service.generate_embeddings(
                model=model,
                tokenizer=tokenizer,
                texts=[request.prompt],
            )
        
        return EmbeddingsResponse(
            embedding=embeddings[0]
        )
        
    except Exception as e:
        logger.error(f"Embeddings failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tags")
async def list_models() -> ListModelsResponse:
    """List all registered models."""
    models = model_manager.list_models()
    
    model_infos = []
    for meta in models:
        model_infos.append(
            ModelInfo(
                name=meta.name,
                model=meta.name,
                modified_at=meta.modified_at,
                size=meta.size,
                digest=meta.digest,
                details=ModelDetails(
                    parent_model="",
                    format=meta.format,
                    family=meta.families[0] if meta.families else "",
                    families=meta.families,
                    parameter_size=meta.parameter_size,
                    quantization_level=meta.quantization_level,
                )
            )
        )
    
    return ListModelsResponse(models=model_infos)


@router.post("/show")
async def show_model(request: ShowModelRequest) -> ShowModelResponse:
    """Show model information."""
    meta = model_manager.show_model(request.model)
    
    if not meta:
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found")
    
    return ShowModelResponse(
        details=ModelDetails(
            parent_model="",
            format=meta.format,
            family=meta.families[0] if meta.families else "",
            families=meta.families,
            parameter_size=meta.parameter_size,
            quantization_level=meta.quantization_level,
        )
    )


@router.post("/pull")
async def pull_model(request: PullRequest):
    """Download a model from HuggingFace."""
    try:
        if request.stream:
            async def pull_stream() -> AsyncIterator[str]:
                async for progress in model_manager.pull_model(
                    request.model,
                    stream_progress=True
                ):
                    response = PullResponse(**progress)
                    yield json.dumps(response.model_dump()) + "\n"
            
            return StreamingResponse(
                pull_stream(),
                media_type="application/x-ndjson"
            )
        else:
            # Non-streaming
            async for _ in model_manager.pull_model(request.model, stream_progress=False):
                pass
            
            return PullResponse(status="success")
            
    except Exception as e:
        logger.error(f"Pull failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/copy")
async def copy_model(request: CopyRequest):
    """Copy a model."""
    try:
        await model_manager.copy_model(request.source, request.destination)
        return JSONResponse(content={}, status_code=200)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Copy failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete")
@router.post("/delete")
async def delete_model(request: DeleteRequest):
    """Delete a model."""
    try:
        await model_manager.delete_model(request.model)
        return JSONResponse(content={}, status_code=200)
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ps")
async def list_running() -> ListRunningResponse:
    """List currently loaded models."""
    running = model_manager.list_running_models()
    
    running_models = []
    for info in running:
        # Get metadata
        meta = model_manager.show_model(info["name"])
        
        if meta:
            # Calculate expiry time (5 minutes from last use)
            last_used = datetime.fromisoformat(info["last_used"])
            expires_at = last_used + timedelta(minutes=5)
            
            running_models.append(
                RunningModel(
                    name=info["name"],
                    model=info["name"],
                    digest="",
                    size=0,
                    details=ModelDetails(
                        parent_model="",
                        format=meta.format,
                        family=meta.families[0] if meta.families else "",
                        families=meta.families,
                        parameter_size=meta.parameter_size,
                        quantization_level=meta.quantization_level,
                    ),
                    expires_at=expires_at.isoformat(),
                    size_vram=0,
                )
            )
    
    return ListRunningResponse(models=running_models)


@router.post("/create")
async def create_model():
    """Create model endpoint (not implemented)."""
    raise HTTPException(
        status_code=501,
        detail="Model creation not supported. Use HuggingFace models directly."
    )


@router.post("/push")
async def push_model():
    """Push model endpoint (not implemented)."""
    raise HTTPException(
        status_code=501,
        detail="Model pushing not supported. This is a local HuggingFace wrapper."
    )
