"""Model lifecycle management for AirLLM models."""

import shutil
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from airllm import AutoModel
from transformers import AutoConfig

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    name: str
    architecture: str
    parameter_size: str
    quantization_level: Optional[str]
    format: str = "gguf"
    families: List[str] = None
    size: int = 0
    modified_at: str = ""
    digest: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = asdict(self)
        if self.families is None:
            data["families"] = []
        return data


class ModelManager:
    """Singleton manager for AirLLM model lifecycle."""
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the model manager."""
        if self._initialized:
            return
            
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        self.registry: Dict[str, ModelMetadata] = {}
        self.generation_lock = asyncio.Lock()  # Serialize generation requests
        self._initialized = True
        
        # Ensure cache directory exists
        settings.cache_path.mkdir(parents=True, exist_ok=True)
        
        # Load registry
        self._load_registry()
    
    def _load_registry(self):
        """Load the models registry from disk."""
        if settings.models_registry_path.exists():
            try:
                with open(settings.models_registry_path, "r") as f:
                    data = json.load(f)
                    for name, metadata in data.items():
                        self.registry[name] = ModelMetadata(**metadata)
                logger.info(f"Loaded {len(self.registry)} models from registry")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                self.registry = {}
        else:
            logger.info("No existing registry found, starting fresh")
    
    def _save_registry(self):
        """Save the models registry to disk."""
        try:
            data = {name: meta.to_dict() for name, meta in self.registry.items()}
            with open(settings.models_registry_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.registry)} models to registry")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    async def load_model(
        self,
        model_name: str,
        compression: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Load a model into memory.
        
        Args:
            model_name: HuggingFace model repo ID or local path
            compression: Compression type (4bit, 8bit, or None)
            **kwargs: Additional arguments for AutoModel.from_pretrained
            
        Returns:
            Dict containing model instance, tokenizer, and metadata
        """
        async with self._lock:
            # If model already loaded, return it
            if model_name in self.loaded_models:
                self.loaded_models[model_name]["last_used"] = datetime.now().isoformat()
                logger.info(f"Model {model_name} already loaded")
                return self.loaded_models[model_name]
            
            # Unload other models if we're at capacity
            if len(self.loaded_models) >= settings.max_loaded_models:
                await self._unload_oldest_model()
            
            logger.info(f"Loading model {model_name}...")
            
            try:
                # Prepare kwargs
                layer_shards_path = settings.cache_path / "layers" / model_name.replace("/", "_")
                layer_shards_path.mkdir(parents=True, exist_ok=True)
                model_kwargs = {
                    "layer_shards_saving_path": str(layer_shards_path),
                    "profiling_mode": False,
                }
                
                if compression:
                    model_kwargs["compression"] = compression
                elif settings.default_compression and settings.default_compression.lower() != "none":
                    model_kwargs["compression"] = settings.default_compression
                
                if settings.hf_token:
                    model_kwargs["hf_token"] = settings.hf_token
                
                model_kwargs.update(kwargs)

                # Run the blocking AutoModel.from_pretrained in a thread so the
                # event loop stays responsive during model initialisation.
                loop = asyncio.get_running_loop()

                async def _try_load() -> Any:
                    try:
                        return await loop.run_in_executor(
                            None,
                            lambda: AutoModel.from_pretrained(model_name, **model_kwargs),
                        )
                    except AssertionError as ae:
                        msg = str(ae)
                        if "safetensors.index.json should exist" in msg or "pytorch_model.bin.index.json" in msg:
                            raise RuntimeError(
                                f"Model '{model_name}' appears to be a single-file model (no shard index found). "
                                "AirLLM requires multi-shard models (typically 7B+ parameters). "
                                "Try a larger model such as mistralai/Mistral-7B-Instruct-v0.2."
                            ) from ae
                        raise

                def _preemptive_tied_lm_head_fix() -> bool:
                    """
                    Models with tie_word_embeddings=True (e.g. Qwen2.5, Phi) do not store
                    lm_head.weight as a separate tensor in the safetensors shards — it is
                    aliased to model.embed_tokens.weight. AirLLM crashes with IndexError
                    when it tries to split that missing tensor.

                    Pre-create lm_head.safetensors from the original HuggingFace model
                    files BEFORE AirLLM's splitting pass so that it sees the shard as
                    already existing and skips it entirely.
                    Returns True if the shard was created (or already existed).
                    """
                    split_dir = layer_shards_path / "splitted_model"
                    lm_head_file = split_dir / "lm_head.safetensors"
                    if lm_head_file.exists():
                        return True  # already handled

                    try:
                        # Check model config for tie_word_embeddings
                        cfg = AutoConfig.from_pretrained(
                            model_name,
                            token=settings.hf_token or None,
                            trust_remote_code=True,
                            local_files_only=True,
                        )
                        if not getattr(cfg, "tie_word_embeddings", False):
                            return False
                    except Exception as e:
                        logger.debug(f"Tied lm_head pre-check: could not load config: {e}")
                        return False

                    try:
                        # Locate the original HF cached model shards
                        from huggingface_hub import snapshot_download as _snap
                        local_path = _snap(
                            model_name,
                            local_files_only=True,
                            token=settings.hf_token or None,
                        )
                    except Exception as e:
                        logger.debug(f"Tied lm_head pre-check: snapshot_download failed: {e}")
                        return False

                    try:
                        import glob
                        from safetensors import safe_open
                        from safetensors.torch import save_file as _st_save

                        shard_files = sorted(
                            glob.glob(f"{local_path}/model-*.safetensors") +
                            glob.glob(f"{local_path}/model.safetensors")
                        )
                        if not shard_files:
                            logger.debug("Tied lm_head pre-check: no safetensors shards found")
                            return False

                        embed_weight = None
                        for shard_path in shard_files:
                            with safe_open(shard_path, framework="pt") as f:
                                for key in f.keys():
                                    if "embed_tokens.weight" in key:
                                        embed_weight = f.get_tensor(key)
                                        break
                            if embed_weight is not None:
                                break

                        if embed_weight is None:
                            logger.debug("Tied lm_head pre-check: embed_tokens.weight not found")
                            return False

                        split_dir.mkdir(parents=True, exist_ok=True)
                        _st_save({"lm_head.weight": embed_weight}, str(lm_head_file))
                        logger.info(
                            f"Pre-created tied lm_head shard at {lm_head_file} "
                            f"(tie_word_embeddings=True for {model_name})"
                        )
                        return True
                    except Exception as e:
                        logger.warning(f"Tied lm_head pre-check: failed to create shard: {e}", exc_info=True)
                        return False

                # Pre-create lm_head shard for weight-tied models before AirLLM's
                # splitting pass so the IndexError never occurs.
                await loop.run_in_executor(None, _preemptive_tied_lm_head_fix)

                try:
                    model = await _try_load()
                except IndexError as first_err:
                    # Fallback: post-crash attempt using the already-split embed shard.
                    # This handles the case where the preemptive fix couldn't run (e.g.
                    # model config not in local cache yet on the very first pull).
                    logger.warning(
                        f"IndexError during layer splitting ({first_err}); "
                        "attempting post-crash tied lm_head fix..."
                    )
                    split_dir = layer_shards_path / "splitted_model"
                    lm_head_file = split_dir / "lm_head.safetensors"
                    embed_file = split_dir / "model.embed_tokens.safetensors"
                    fixed = False
                    if embed_file.exists() and not lm_head_file.exists():
                        try:
                            from safetensors.torch import load_file as _lt_load
                            from safetensors.torch import save_file as _lt_save
                            tensors = _lt_load(str(embed_file))
                            # Key may be full path or just "weight" depending on AirLLM version
                            weight = (
                                tensors.get("model.embed_tokens.weight")
                                or tensors.get("weight")
                                or next(iter(tensors.values()), None)
                            )
                            if weight is not None:
                                _lt_save({"lm_head.weight": weight}, str(lm_head_file))
                                logger.info(f"Post-crash tied lm_head fix: created {lm_head_file}")
                                fixed = True
                        except Exception as fix_err:
                            logger.warning(f"Post-crash tied lm_head fix failed: {fix_err}", exc_info=True)

                    if fixed:
                        logger.info("Retrying load after tied lm_head fix...")
                        model = await _try_load()
                    else:
                        raise
                except RuntimeError:
                    # Our own descriptive errors (e.g. single-file model) — propagate as-is.
                    raise
                except Exception as first_err:
                    # Other unexpected failures: clear cache once and retry.
                    if layer_shards_path.exists() and any(layer_shards_path.iterdir()):
                        logger.warning(
                            f"Load failed ({first_err}); clearing stale layer cache at "
                            f"{layer_shards_path} and retrying..."
                        )
                        shutil.rmtree(layer_shards_path)
                        layer_shards_path.mkdir(parents=True, exist_ok=True)
                        model = await _try_load()
                    else:
                        raise

                # Store model info
                model_info = {
                    "model": model,
                    "tokenizer": model.tokenizer,
                    "loaded_at": datetime.now().isoformat(),
                    "last_used": datetime.now().isoformat(),
                    "compression": compression or settings.default_compression,
                }
                
                self.loaded_models[model_name] = model_info
                
                # Register model if not already registered
                if model_name not in self.registry:
                    await self._register_model(model_name, model)
                
                logger.info(f"Successfully loaded model {model_name}")
                return model_info
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise
    
    async def _unload_oldest_model(self):
        """Unload the least recently used model."""
        if not self.loaded_models:
            return
        
        # Find oldest model by last_used timestamp
        oldest_name = min(
            self.loaded_models.keys(),
            key=lambda k: self.loaded_models[k]["last_used"]
        )
        
        await self.unload_model(oldest_name)
    
    async def unload_model(self, model_name: str):
        """Unload a model from memory."""
        if model_name in self.loaded_models:
            logger.info(f"Unloading model {model_name}")
            # Delete the model to free memory
            del self.loaded_models[model_name]
            # Force garbage collection could be added here if needed
            logger.info(f"Model {model_name} unloaded")
    
    async def get_model(self, model_name: str) -> Dict[str, Any]:
        """
        Get a loaded model or load it if not loaded.
        
        Args:
            model_name: HuggingFace model repo ID
            
        Returns:
            Dict containing model instance and metadata
        """
        if model_name in self.loaded_models:
            self.loaded_models[model_name]["last_used"] = datetime.now().isoformat()
            return self.loaded_models[model_name]
        
        return await self.load_model(model_name)
    
    async def _register_model(self, model_name: str, model: Any):
        """Register a model in the registry."""
        try:
            # Get model config from HuggingFace
            config = AutoConfig.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=settings.hf_token
            )
            
            # Extract metadata
            architecture = config.architectures[0] if hasattr(config, "architectures") else "unknown"
            
            # Determine family based on architecture
            families = []
            if "Llama" in architecture:
                families.append("llama")
            elif "Mistral" in architecture:
                families.append("mistral")
            elif "Qwen" in architecture:
                families.append("qwen")
            elif "ChatGLM" in architecture:
                families.append("chatglm")
            elif "Baichuan" in architecture:
                families.append("baichuan")
            
            # Try to get parameter size
            param_size = "unknown"
            if hasattr(config, "num_parameters"):
                params = config.num_parameters
                if params > 1e9:
                    param_size = f"{params / 1e9:.1f}B"
                elif params > 1e6:
                    param_size = f"{params / 1e6:.1f}M"
            
            metadata = ModelMetadata(
                name=model_name,
                architecture=architecture,
                parameter_size=param_size,
                quantization_level=settings.default_compression,
                families=families,
                modified_at=datetime.now().isoformat(),
            )
            
            self.registry[model_name] = metadata
            self._save_registry()
            
            logger.info(f"Registered model {model_name}")
            
        except Exception as e:
            logger.warning(f"Failed to register model metadata: {e}")
            # Create minimal metadata
            self.registry[model_name] = ModelMetadata(
                name=model_name,
                architecture="unknown",
                parameter_size="unknown",
                quantization_level=settings.default_compression,
                families=[],
                modified_at=datetime.now().isoformat(),
            )
            self._save_registry()
    
    async def pull_model(self, model_name: str, stream_progress: bool = False):
        """
        Download a model from HuggingFace.
        
        Args:
            model_name: HuggingFace model repo ID
            stream_progress: Whether to stream download progress
            
        Yields:
            Progress updates if stream_progress is True
        """
        from huggingface_hub import snapshot_download

        logger.info(f"Pulling model {model_name}")

        if stream_progress:
            yield {"status": "pulling manifest"}

        try:
            loop = asyncio.get_running_loop()

            # Download all model files from HuggingFace Hub.
            # Runs in a thread executor so the event loop is not blocked and
            # huggingface_hub's tqdm progress bars are printed to the terminal.
            download_kwargs = {"repo_id": model_name}
            if settings.hf_token:
                download_kwargs["token"] = settings.hf_token

            if stream_progress:
                yield {"status": "downloading"}

            await loop.run_in_executor(
                None,
                lambda: snapshot_download(**download_kwargs),
            )

            if stream_progress:
                yield {"status": "verifying sha256"}

            # Register the model in the local registry without loading it into
            # memory.  Loading (splitting into layers) happens lazily on first
            # inference, matching Ollama's pull behaviour.
            if model_name not in self.registry:
                await self._register_model(model_name, None)

            if stream_progress:
                yield {"status": "success"}

            logger.info(f"Successfully pulled model {model_name}")

        except Exception as e:
            err_msg = str(e)
            # Provide a clearer message for gated / private models
            if "cannot find the requested files" in err_msg or "401" in err_msg or "403" in err_msg:
                friendly = (
                    f"Cannot download '{model_name}': access denied. "
                    "This is likely a gated model. "
                    "1) Accept the model license at https://huggingface.co/{model_name} "
                    "2) Set HF_TOKEN=<your_token> in .env and restart the server."
                ).replace("{model_name}", model_name)
                logger.error(f"Failed to pull model {model_name}: {friendly}")
                if stream_progress:
                    yield {"status": "error", "error": friendly}
                raise RuntimeError(friendly) from e
            logger.error(f"Failed to pull model {model_name}: {e}")
            if stream_progress:
                yield {"status": "error", "error": err_msg}
            raise
    
    async def delete_model(self, model_name: str):
        """Delete a model from the registry."""
        # Unload if loaded
        await self.unload_model(model_name)
        
        # Remove from registry
        if model_name in self.registry:
            del self.registry[model_name]
            self._save_registry()
            logger.info(f"Deleted model {model_name} from registry")
    
    async def copy_model(self, source: str, destination: str):
        """Copy a model entry in the registry."""
        if source not in self.registry:
            raise ValueError(f"Source model {source} not found")
        
        # Copy metadata
        self.registry[destination] = ModelMetadata(
            **{**self.registry[source].to_dict(), "name": destination}
        )
        self._save_registry()
        logger.info(f"Copied model {source} to {destination}")
    
    def list_models(self) -> List[ModelMetadata]:
        """List all registered models."""
        return list(self.registry.values())
    
    def show_model(self, model_name: str) -> Optional[ModelMetadata]:
        """Get metadata for a specific model."""
        return self.registry.get(model_name)
    
    def list_running_models(self) -> List[Dict[str, Any]]:
        """List currently loaded models."""
        return [
            {
                "name": name,
                "loaded_at": info["loaded_at"],
                "last_used": info["last_used"],
                "compression": info["compression"],
            }
            for name, info in self.loaded_models.items()
        ]


# Global model manager instance
model_manager = ModelManager()
