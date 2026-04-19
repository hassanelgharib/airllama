"""Text generation and inference utilities."""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, AsyncIterator
from datetime import datetime

import torch

from app.config import settings

logger = logging.getLogger(__name__)


class GenerationService:
    """Service for handling text generation with AirLLM models."""
    
    @staticmethod
    def _apply_chat_template(
        tokenizer: Any,
        messages: List[Dict[str, str]],
    ) -> str:
        """
        Apply chat template to format messages into a prompt.
        
        Args:
            tokenizer: The model's tokenizer
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Formatted prompt string
        """
        # Try to use the tokenizer's chat template if available
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return prompt
            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}, using fallback")
        
        # Fallback: simple template
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
        
        # Add prompt for assistant response
        prompt_parts.append("Assistant:")
        
        return "\n".join(prompt_parts)
    
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
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text completion.
        
        Args:
            model: The AirLLM model instance
            tokenizer: The model's tokenizer
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stop: List of stop sequences
            seed: Random seed for reproducibility
            **kwargs: Additional generation parameters
            
        Returns:
            Dict with generated text and timing statistics
        """
        start_time = time.time()
        
        # Set max_new_tokens
        if max_new_tokens is None:
            max_new_tokens = settings.default_max_new_tokens
        
        # Tokenize input
        try:
            input_tokens = tokenizer(
                prompt,
                return_tensors="pt",
                return_attention_mask=True,
                truncation=True,
                max_length=settings.default_max_length,
                padding=False
            )
        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            raise
        
        prompt_tokens = input_tokens['input_ids'].shape[1]
        logger.info(f"Prompt tokens: {prompt_tokens}")
        
        # Prepare generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "use_cache": False,  # AirLLM disables KV cache internally; passing True causes NoneType errors
            "return_dict_in_generate": True,
            "do_sample": temperature > 0,
        }
        
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
            gen_kwargs["top_k"] = top_k
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Update with any additional kwargs
        gen_kwargs.update(kwargs)
        
        # Move to GPU if available
        input_ids = input_tokens['input_ids']
        attention_mask = input_tokens.get('attention_mask')
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            if attention_mask is not None:
                attention_mask = attention_mask.cuda()
        
        # Generate
        generation_start = time.time()
        try:
            generation_output = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
        
        generation_time = time.time() - generation_start
        
        # Decode output
        output_text = tokenizer.decode(
            generation_output.sequences[0],
            skip_special_tokens=True
        )
        
        # Remove the prompt from output if present
        if output_text.startswith(prompt):
            output_text = output_text[len(prompt):].strip()
        
        # Apply stop sequences
        if stop:
            for stop_seq in stop:
                if stop_seq in output_text:
                    output_text = output_text[:output_text.index(stop_seq)]
        
        eval_tokens = generation_output.sequences[0].shape[0] - prompt_tokens
        total_time = time.time() - start_time
        
        # Build timing statistics (in nanoseconds for Ollama compatibility)
        stats = {
            "total_duration": int(total_time * 1e9),
            "load_duration": 0,  # Model already loaded
            "prompt_eval_count": prompt_tokens,
            "prompt_eval_duration": 0,  # Not separately measured
            "eval_count": eval_tokens,
            "eval_duration": int(generation_time * 1e9),
        }
        
        return {
            "text": output_text,
            "stats": stats,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": eval_tokens,
        }
    
    @staticmethod
    async def generate_chat_completion(
        model: Any,
        tokenizer: Any,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate chat completion from messages.
        
        Args:
            model: The AirLLM model instance
            tokenizer: The model's tokenizer
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            Dict with generated text and timing statistics
        """
        # Format messages into a prompt
        prompt = GenerationService._apply_chat_template(tokenizer, messages)
        
        # Generate completion
        return await GenerationService.generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            **kwargs
        )
    
    @staticmethod
    def split_into_tokens(tokenizer: Any, text: str) -> List[str]:
        """
        Split generated text back into individual tokens for streaming.
        
        Args:
            tokenizer: The model's tokenizer
            text: Generated text to split
            
        Returns:
            List of token strings
        """
        # Tokenize the text
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        
        # Decode each token individually
        tokens = []
        for token_id in token_ids:
            token_text = tokenizer.decode([token_id], skip_special_tokens=True)
            tokens.append(token_text)
        
        return tokens
    
    @staticmethod
    async def stream_completion(
        model: Any,
        tokenizer: Any,
        prompt: str,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Simulate streaming by generating full response then yielding tokens.
        
        Args:
            model: The AirLLM model instance
            tokenizer: The model's tokenizer
            prompt: Input prompt text
            **kwargs: Additional generation parameters
            
        Yields:
            Dicts with incremental response tokens
        """
        # Generate full completion
        result = await GenerationService.generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            **kwargs
        )
        
        text = result["text"]
        stats = result["stats"]
        
        # Split into tokens
        tokens = GenerationService.split_into_tokens(tokenizer, text)
        
        # Yield tokens one by one
        for i, token in enumerate(tokens):
            chunk = {
                "token": token,
                "done": False,
            }
            yield chunk
            
            # Small delay to simulate streaming
            await asyncio.sleep(0.01)
        
        # Final chunk with stats
        yield {
            "token": "",
            "done": True,
            "stats": stats,
        }
    
    @staticmethod
    async def stream_chat_completion(
        model: Any,
        tokenizer: Any,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Simulate streaming chat completion.
        
        Args:
            model: The AirLLM model instance
            tokenizer: The model's tokenizer
            messages: List of message dicts
            **kwargs: Additional generation parameters
            
        Yields:
            Dicts with incremental response tokens
        """
        # Format messages
        prompt = GenerationService._apply_chat_template(tokenizer, messages)
        
        # Stream completion
        async for chunk in GenerationService.stream_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            **kwargs
        ):
            yield chunk
    
    @staticmethod
    async def generate_embeddings(
        model: Any,
        tokenizer: Any,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Generate embeddings for input texts.
        
        Note: This is a basic implementation using mean pooling of last hidden states.
        For production, consider using dedicated embedding models.
        
        Args:
            model: The AirLLM model instance
            tokenizer: The model's tokenizer
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for text in texts:
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=settings.default_max_length,
                padding=True
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Get model output
            # Note: This is a simplified approach. AirLLM might not expose
            # the full model architecture needed for proper embeddings.
            # For now, we'll return a placeholder or attempt basic extraction.
            try:
                with torch.no_grad():
                    # Try to get the last hidden state
                    # This might not work for all AirLLM model types
                    outputs = model.model(**inputs, output_hidden_states=True)
                    last_hidden = outputs.hidden_states[-1]
                    
                    # Mean pooling
                    embedding = last_hidden.mean(dim=1).squeeze().cpu().numpy().tolist()
                    embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Embeddings not supported for this model: {e}")
                # Return a zero vector as fallback
                embeddings.append([0.0] * 768)  # Standard BERT-like dimension
        
        return embeddings


# Global service instance
generation_service = GenerationService()
