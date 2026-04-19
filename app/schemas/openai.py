"""Pydantic schemas for OpenAI-compatible API."""

from typing import List, Optional, Dict, Any, Literal, Union
from pydantic import BaseModel, Field


# Common models
class Usage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class FunctionCall(BaseModel):
    """Function call information."""
    name: str
    arguments: str


class ToolCall(BaseModel):
    """Tool call information."""
    id: str = "call_0"
    type: str = "function"
    function: FunctionCall


# Chat Completion
class ChatMessage(BaseModel):
    """Chat message."""
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """Request for /v1/chat/completions."""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.9
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    seed: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None


class ChatCompletionChoice(BaseModel):
    """Choice in chat completion response."""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


class ChatCompletionResponse(BaseModel):
    """Response for /v1/chat/completions."""
    id: str = "chatcmpl-0"
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage
    system_fingerprint: Optional[str] = None


class ChatCompletionChunkDelta(BaseModel):
    """Delta in streaming chat completion."""
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionChunkChoice(BaseModel):
    """Choice in streaming chat completion."""
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


class ChatCompletionChunk(BaseModel):
    """Streaming chunk for chat completion."""
    id: str = "chatcmpl-0"
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]
    system_fingerprint: Optional[str] = None


# Text Completion
class CompletionRequest(BaseModel):
    """Request for /v1/completions."""
    model: str
    prompt: Union[str, List[str]]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.9
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    best_of: Optional[int] = 1
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    seed: Optional[int] = None


class CompletionChoice(BaseModel):
    """Choice in completion response."""
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    """Response for /v1/completions."""
    id: str = "cmpl-0"
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage
    system_fingerprint: Optional[str] = None


# Embeddings
class EmbeddingRequest(BaseModel):
    """Request for /v1/embeddings."""
    model: str
    input: Union[str, List[str], List[int], List[List[int]]]
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


class EmbeddingData(BaseModel):
    """Embedding data."""
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    """Response for /v1/embeddings."""
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Usage


# Models
class Model(BaseModel):
    """Model information."""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "airllm"


class ModelList(BaseModel):
    """List of models."""
    object: str = "list"
    data: List[Model]
