"""OpenAI-compatible API router."""

import logging
import time
from datetime import datetime
from typing import AsyncIterator
from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.schemas.openai import (
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionChoice, ChatCompletionChunk,
    ChatCompletionChunkChoice, ChatCompletionChunkDelta,
    ChatMessage, Usage,
    CompletionRequest, CompletionResponse, CompletionChoice,
    EmbeddingRequest, EmbeddingResponse, EmbeddingData,
    ModelList, Model,
)
from app.services.model_manager import model_manager
from app.services.generation import generation_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["openai"])


@router.post("/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Create a chat completion (OpenAI-compatible).
    
    Supports both streaming and non-streaming responses.
    """
    try:
        # Get model
        model_info = await model_manager.get_model(request.model)
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]
        
        # Convert messages
        messages = [msg.model_dump() for msg in request.messages]
        
        # Parse stop sequences
        stop = request.stop
        if isinstance(stop, str):
            stop = [stop]
        
        # Handle streaming
        if request.stream:
            async def chat_stream() -> AsyncIterator[str]:
                created_time = int(time.time())
                
                async with model_manager.generation_lock:
                    async for chunk in generation_service.stream_chat_completion(
                        model=model,
                        tokenizer=tokenizer,
                        messages=messages,
                        max_new_tokens=request.max_tokens,
                        temperature=request.temperature or 0.8,
                        top_p=request.top_p or 0.9,
                        seed=request.seed,
                        stop=stop,
                    ):
                        if not chunk["done"]:
                            # Content chunk
                            response = ChatCompletionChunk(
                                id=f"chatcmpl-{created_time}",
                                created=created_time,
                                model=request.model,
                                choices=[
                                    ChatCompletionChunkChoice(
                                        index=0,
                                        delta=ChatCompletionChunkDelta(
                                            content=chunk["token"]
                                        ),
                                        finish_reason=None,
                                    )
                                ]
                            )
                            yield response.model_dump_json()
                        else:
                            # Final chunk
                            response = ChatCompletionChunk(
                                id=f"chatcmpl-{created_time}",
                                created=created_time,
                                model=request.model,
                                choices=[
                                    ChatCompletionChunkChoice(
                                        index=0,
                                        delta=ChatCompletionChunkDelta(),
                                        finish_reason="stop",
                                    )
                                ]
                            )
                            yield response.model_dump_json()
            
            return EventSourceResponse(chat_stream())
        else:
            # Non-streaming
            async with model_manager.generation_lock:
                result = await generation_service.generate_chat_completion(
                    model=model,
                    tokenizer=tokenizer,
                    messages=messages,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature or 0.8,
                    top_p=request.top_p or 0.9,
                    seed=request.seed,
                    stop=stop,
                )
            
            created_time = int(time.time())
            
            response = ChatCompletionResponse(
                id=f"chatcmpl-{created_time}",
                created=created_time,
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=result["text"]
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=Usage(
                    prompt_tokens=result["prompt_tokens"],
                    completion_tokens=result["completion_tokens"],
                    total_tokens=result["prompt_tokens"] + result["completion_tokens"],
                )
            )
            return response
            
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/completions")
async def create_completion(request: CompletionRequest):
    """
    Create a text completion (OpenAI-compatible).
    
    Supports both streaming and non-streaming responses.
    """
    try:
        # Get model
        model_info = await model_manager.get_model(request.model)
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]
        
        # Handle prompt (currently only support single string)
        if isinstance(request.prompt, list):
            prompt = request.prompt[0] if request.prompt else ""
        else:
            prompt = request.prompt
        
        # Parse stop sequences
        stop = request.stop
        if isinstance(stop, str):
            stop = [stop]
        
        # Handle streaming
        if request.stream:
            async def completion_stream() -> AsyncIterator[str]:
                created_time = int(time.time())
                
                async with model_manager.generation_lock:
                    async for chunk in generation_service.stream_completion(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        max_new_tokens=request.max_tokens,
                        temperature=request.temperature or 0.8,
                        top_p=request.top_p or 0.9,
                        seed=request.seed,
                        stop=stop,
                    ):
                        if not chunk["done"]:
                            # Content chunk
                            response = {
                                "id": f"cmpl-{created_time}",
                                "object": "text_completion",
                                "created": created_time,
                                "model": request.model,
                                "choices": [{
                                    "text": chunk["token"],
                                    "index": 0,
                                    "finish_reason": None,
                                }]
                            }
                            yield f"data: {json.dumps(response)}\n\n"
                        else:
                            # Final chunk
                            yield "data: [DONE]\n\n"
            
            return EventSourceResponse(completion_stream())
        else:
            # Non-streaming
            async with model_manager.generation_lock:
                result = await generation_service.generate_completion(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature or 0.8,
                    top_p=request.top_p or 0.9,
                    seed=request.seed,
                    stop=stop,
                )
            
            created_time = int(time.time())
            
            response = CompletionResponse(
                id=f"cmpl-{created_time}",
                created=created_time,
                model=request.model,
                choices=[
                    CompletionChoice(
                        text=result["text"],
                        index=0,
                        finish_reason="stop",
                    )
                ],
                usage=Usage(
                    prompt_tokens=result["prompt_tokens"],
                    completion_tokens=result["completion_tokens"],
                    total_tokens=result["prompt_tokens"] + result["completion_tokens"],
                )
            )
            return response
            
    except Exception as e:
        logger.error(f"Completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embeddings")
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """Create embeddings (OpenAI-compatible)."""
    try:
        # Get model
        model_info = await model_manager.get_model(request.model)
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]
        
        # Parse input
        if isinstance(request.input, str):
            texts = [request.input]
        elif isinstance(request.input, list):
            if request.input and isinstance(request.input[0], int):
                # Token input - decode first
                texts = [tokenizer.decode(request.input)]
            elif request.input and isinstance(request.input[0], list):
                # Array of token arrays
                texts = [tokenizer.decode(tokens) for tokens in request.input]
            else:
                # Array of strings
                texts = request.input
        else:
            texts = []
        
        # Generate embeddings
        async with model_manager.generation_lock:
            embeddings = await generation_service.generate_embeddings(
                model=model,
                tokenizer=tokenizer,
                texts=texts,
            )
        
        # Build response
        data = [
            EmbeddingData(
                embedding=emb,
                index=i
            )
            for i, emb in enumerate(embeddings)
        ]
        
        total_tokens = sum(len(tokenizer.encode(t)) for t in texts)
        
        return EmbeddingResponse(
            data=data,
            model=request.model,
            usage=Usage(
                prompt_tokens=total_tokens,
                completion_tokens=0,
                total_tokens=total_tokens,
            )
        )
        
    except Exception as e:
        logger.error(f"Embeddings failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
async def list_models() -> ModelList:
    """List available models (OpenAI-compatible)."""
    models = model_manager.list_models()
    
    model_list = [
        Model(
            id=meta.name,
            created=int(datetime.fromisoformat(meta.modified_at).timestamp()) if meta.modified_at else 0,
            owned_by="airllm",
        )
        for meta in models
    ]
    
    return ModelList(data=model_list)


@router.get("/models/{model_id}")
async def retrieve_model(model_id: str) -> Model:
    """Retrieve model information (OpenAI-compatible)."""
    meta = model_manager.show_model(model_id)
    
    if not meta:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    return Model(
        id=meta.name,
        created=int(datetime.fromisoformat(meta.modified_at).timestamp()) if meta.modified_at else 0,
        owned_by="airllm",
    )
