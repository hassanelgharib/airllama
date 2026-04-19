"""Text generation and inference utilities."""

import asyncio
import logging
import time
from threading import Thread
from typing import Dict, List, Any, Optional, AsyncIterator
from datetime import datetime

import torch
from transformers import TextIteratorStreamer

from app.config import settings

logger = logging.getLogger(__name__)


class GenerationService:
    """Service for handling text generation with AirLLM models."""
    
    @staticmethod
    def _apply_chat_template(
        tokenizer: Any,
        messages: List[Dict[str, str]],
    ) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}, using fallback")
        
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)

    @staticmethod
    def _tokenize(tokenizer, prompt):
        """Tokenize prompt and return input_ids + attention_mask on the right device."""
        input_tokens = tokenizer(
            prompt,
            return_tensors="pt",
            return_attention_mask=True,
            truncation=True,
            max_length=settings.default_max_length,
            padding=False,
        )
        input_ids = input_tokens["input_ids"]
        attention_mask = input_tokens.get("attention_mask")
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            if attention_mask is not None:
                attention_mask = attention_mask.cuda()
        return input_ids, attention_mask

    @staticmethod
    async def _token_stream(
        model: Any,
        tokenizer: Any,
        input_ids: Any,
        attention_mask: Any,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: Optional[int],
    ) -> AsyncIterator[str]:
        """
        Real streaming: run model.generate() in a background thread with
        TextIteratorStreamer and yield decoded tokens via an asyncio.Queue
        as they are produced — one 39-layer pass per token.
        """
        if seed is not None:
            torch.manual_seed(seed)

        do_sample = temperature > 0
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "use_cache": False,
            "do_sample": do_sample,
            "streamer": streamer,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            gen_kwargs["top_k"] = top_k

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _run_generate():
            try:
                model.generate(**gen_kwargs)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

        def _consume_streamer():
            for token_text in streamer:  # blocks until each token is ready
                loop.call_soon_threadsafe(queue.put_nowait, ("token", token_text))
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        Thread(target=_run_generate, daemon=True).start()
        Thread(target=_consume_streamer, daemon=True).start()

        while True:
            kind, val = await queue.get()
            if kind == "done":
                return
            if kind == "error":
                raise val
            yield val  # decoded token string

    @staticmethod
    async def stream_completion(
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 40,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream text completion token-by-token using TextIteratorStreamer.
        Each token is yielded as soon as its forward pass completes.
        """
        if max_new_tokens is None:
            max_new_tokens = settings.default_max_new_tokens

        input_ids, attention_mask = GenerationService._tokenize(tokenizer, prompt)
        prompt_tokens = input_ids.shape[1]
        start_time = time.time()
        eval_count = 0
        generated_so_far = ""

        async for token_text in GenerationService._token_stream(
            model, tokenizer, input_ids, attention_mask,
            max_new_tokens, temperature, top_p, top_k, seed,
        ):
            if not token_text:
                continue

            # Check stop sequences
            candidate = generated_so_far + token_text
            stop_hit = False
            if stop:
                for stop_seq in stop:
                    if stop_seq in candidate:
                        truncated = candidate[:candidate.index(stop_seq)]
                        emit = truncated[len(generated_so_far):]
                        if emit:
                            yield {"token": emit, "done": False}
                        stop_hit = True
                        break
            if stop_hit:
                break

            generated_so_far = candidate
            eval_count += 1
            yield {"token": token_text, "done": False}

        total_time = time.time() - start_time
        yield {
            "token": "",
            "done": True,
            "stats": {
                "total_duration": int(total_time * 1e9),
                "load_duration": 0,
                "prompt_eval_count": prompt_tokens,
                "prompt_eval_duration": 0,
                "eval_count": eval_count,
                "eval_duration": int(total_time * 1e9),
            },
        }

    @staticmethod
    async def generate_completion(
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 40,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Non-streaming completion: collects all tokens from the streamer then
        returns the full result.  Uses the same TextIteratorStreamer path so
        the background threads keep the event loop responsive.
        """
        if max_new_tokens is None:
            max_new_tokens = settings.default_max_new_tokens

        input_ids, attention_mask = GenerationService._tokenize(tokenizer, prompt)
        prompt_tokens = input_ids.shape[1]
        start_time = time.time()

        tokens: List[str] = []
        async for token_text in GenerationService._token_stream(
            model, tokenizer, input_ids, attention_mask,
            max_new_tokens, temperature, top_p, top_k, seed,
        ):
            tokens.append(token_text)

        output_text = "".join(tokens)

        # Apply stop sequences
        if stop:
            for stop_seq in stop:
                if stop_seq in output_text:
                    output_text = output_text[:output_text.index(stop_seq)]

        total_time = time.time() - start_time
        eval_count = len(tokens)

        return {
            "text": output_text,
            "stats": {
                "total_duration": int(total_time * 1e9),
                "load_duration": 0,
                "prompt_eval_count": prompt_tokens,
                "prompt_eval_duration": 0,
                "eval_count": eval_count,
                "eval_duration": int(total_time * 1e9),
            },
            "prompt_tokens": prompt_tokens,
            "completion_tokens": eval_count,
        }

    @staticmethod
    async def generate_chat_completion(
        model: Any,
        tokenizer: Any,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Dict[str, Any]:
        prompt = GenerationService._apply_chat_template(tokenizer, messages)
        return await GenerationService.generate_completion(
            model=model, tokenizer=tokenizer, prompt=prompt, **kwargs
        )

    @staticmethod
    async def stream_chat_completion(
        model: Any,
        tokenizer: Any,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        prompt = GenerationService._apply_chat_template(tokenizer, messages)
        async for chunk in GenerationService.stream_completion(
            model=model, tokenizer=tokenizer, prompt=prompt, **kwargs
        ):
            yield chunk


    @staticmethod
    async def generate_embeddings(
        model: Any,
        tokenizer: Any,
        texts: List[str],
    ) -> List[List[float]]:
        embeddings = []
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=settings.default_max_length,
                padding=True,
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            try:
                with torch.no_grad():
                    outputs = model.model(**inputs, output_hidden_states=True)
                    last_hidden = outputs.hidden_states[-1]
                    embedding = last_hidden.mean(dim=1).squeeze().cpu().numpy().tolist()
                    embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Embeddings not supported for this model: {e}")
                embeddings.append([0.0] * 768)
        return embeddings


# Global service instance
generation_service = GenerationService()
