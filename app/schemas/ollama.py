"""Pydantic schemas for Ollama native API."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# Generate endpoints
class GenerateRequest(BaseModel):
    """Request for /api/generate endpoint."""
    model: str
    prompt: str
    suffix: Optional[str] = None
    images: Optional[List[str]] = None
    format: Optional[str] = None  # "json" or JSON schema
    options: Optional[Dict[str, Any]] = None
    system: Optional[str] = None
    template: Optional[str] = None
    stream: bool = True
    raw: bool = False
    keep_alive: Optional[str] = "5m"
    context: Optional[List[int]] = None


class GenerateResponse(BaseModel):
    """Response for /api/generate endpoint (non-streaming)."""
    model: str
    created_at: str
    response: str
    done: bool
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class GenerateStreamResponse(BaseModel):
    """Streaming response chunk for /api/generate."""
    model: str
    created_at: str
    response: str
    done: bool
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


# Chat endpoints
class ChatMessage(BaseModel):
    """Chat message."""
    role: str  # system, user, assistant, tool
    content: str
    images: Optional[List[str]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatRequest(BaseModel):
    """Request for /api/chat endpoint."""
    model: str
    messages: List[ChatMessage]
    tools: Optional[List[Dict[str, Any]]] = None
    format: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    stream: bool = True
    keep_alive: Optional[str] = "5m"


class ChatResponseMessage(BaseModel):
    """Chat response message."""
    role: str
    content: str
    images: Optional[List[str]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatResponse(BaseModel):
    """Response for /api/chat endpoint (non-streaming)."""
    model: str
    created_at: str
    message: ChatResponseMessage
    done: bool
    done_reason: Optional[str] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class ChatStreamResponse(BaseModel):
    """Streaming response chunk for /api/chat."""
    model: str
    created_at: str
    message: ChatResponseMessage
    done: bool
    done_reason: Optional[str] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


# Embeddings endpoints
class EmbedRequest(BaseModel):
    """Request for /api/embed endpoint."""
    model: str
    input: str | List[str]  # Single text or list of texts
    truncate: bool = True
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[str] = "5m"


class EmbedResponse(BaseModel):
    """Response for /api/embed endpoint."""
    model: str
    embeddings: List[List[float]]
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None


# Legacy embeddings endpoint
class EmbeddingsRequest(BaseModel):
    """Request for /api/embeddings endpoint (legacy)."""
    model: str
    prompt: str
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[str] = "5m"


class EmbeddingsResponse(BaseModel):
    """Response for /api/embeddings endpoint (legacy)."""
    embedding: List[float]


# Model management endpoints
class ModelDetails(BaseModel):
    """Model details."""
    parent_model: str = ""
    format: str = "gguf"
    family: str = ""
    families: List[str] = []
    parameter_size: str = ""
    quantization_level: Optional[str] = None


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    model: str
    modified_at: str
    size: int = 0
    digest: str = ""
    details: ModelDetails


class ListModelsResponse(BaseModel):
    """Response for /api/tags endpoint."""
    models: List[ModelInfo]


class ShowModelRequest(BaseModel):
    """Request for /api/show endpoint."""
    model: str
    verbose: bool = False


class ShowModelResponse(BaseModel):
    """Response for /api/show endpoint."""
    modelfile: str = ""
    parameters: str = ""
    template: str = ""
    details: ModelDetails
    model_info: Optional[Dict[str, Any]] = None


class PullRequest(BaseModel):
    """Request for /api/pull endpoint."""
    model: str
    insecure: bool = False
    stream: bool = True


class PullResponse(BaseModel):
    """Response for /api/pull endpoint."""
    status: str
    digest: Optional[str] = None
    total: Optional[int] = None
    completed: Optional[int] = None


class CopyRequest(BaseModel):
    """Request for /api/copy endpoint."""
    source: str
    destination: str


class DeleteRequest(BaseModel):
    """Request for /api/delete endpoint."""
    model: str


class RunningModel(BaseModel):
    """Running model information."""
    name: str
    model: str
    size: int = 0
    digest: str = ""
    details: ModelDetails
    expires_at: str
    size_vram: int = 0


class ListRunningResponse(BaseModel):
    """Response for /api/ps endpoint."""
    models: List[RunningModel]


class VersionResponse(BaseModel):
    """Response for /api/version endpoint."""
    version: str
