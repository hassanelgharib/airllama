"""Model lifecycle management for AirLLM models."""

import shutil
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

import torch
from airllm import AutoModel
from airllm.airllm_base import AirLLMBaseModel
from transformers import AutoConfig

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AirLLM ↔ newer-transformers compatibility patches
# Applied once at import time; safe to apply even if the model is not Qwen2.
# ---------------------------------------------------------------------------

# 1. _is_stateful — required by GenerationMixin in transformers ≥ 4.40.
if not hasattr(AirLLMBaseModel, "_is_stateful"):
    AirLLMBaseModel._is_stateful = False

# 2. Qwen2DecoderLayer patches for transformers ≥ 4.45:
#    a) position_embeddings (cos/sin) is now a required argument computed at the
#       Qwen2Model level and shared across all layers.  AirLLM calls each layer
#       individually without this pre-computation, so we inject a reference to
#       the model's Qwen2RotaryEmbedding (_rotary_emb) on each layer and compute
#       it on the fly when it is absent.
#    b) forward() now returns a plain Tensor instead of a tuple.  AirLLM's loop
#       does `layer(...)[0]`, which on a plain Tensor strips the batch dimension.
#       Wrapping in a 1-tuple restores the old behaviour.
#    c) The 4-D boolean mask AirLLM builds is only correct for SDPA; for eager
#       attention it must be a float mask (0 = attend, -inf = mask).
try:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer as _Qwen2DL
    _orig_qwen2_dl_fwd = _Qwen2DL.forward

    def _qwen2_dl_fwd_compat(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        # a) compute RoPE embeddings when the caller didn't provide them.
        # _rotary_emb_list is a plain Python list (NOT an nn.Module attribute) so
        # layer.to("meta") will NOT register it as a submodule and will NOT move
        # the shared rotary_emb to meta when adjacent layers are processed.
        if position_embeddings is None:
            _re_list = getattr(self, "_rotary_emb_list", None)
            _re = _re_list[0] if _re_list else None
            if _re is not None and position_ids is not None:
                position_embeddings = _re(hidden_states, position_ids)

        # c) convert boolean causal mask → float additive mask so that both
        #    SDPA and eager attention paths receive a compatible format
        if attention_mask is not None and attention_mask.dtype == torch.bool:
            float_mask = torch.zeros(
                attention_mask.shape, dtype=hidden_states.dtype, device=hidden_states.device
            )
            float_mask = float_mask.masked_fill(~attention_mask, float("-inf"))
            attention_mask = float_mask

        result = _orig_qwen2_dl_fwd(
            self,
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # b) wrap plain-Tensor return so that layer(...)[0] gives the full
        #    3-D hidden-state tensor rather than stripping the batch dimension
        return (result,) if isinstance(result, torch.Tensor) else result

    _Qwen2DL.forward = _qwen2_dl_fwd_compat
    logger.debug("Patched Qwen2DecoderLayer.forward for AirLLM compatibility")
except Exception as _patch_err:
    logger.debug(f"Qwen2DecoderLayer patch skipped: {_patch_err}")

# 3. After AirLLM rebuilds its model on every forward() call (it calls
#    init_model() each time), inject a _rotary_emb_list reference into every
#    decoder layer so patch #2 can compute position_embeddings.
#    Key details:
#    - Recreate rotary_emb OUTSIDE init_empty_weights() so inv_freq has real data
#      (under init_empty_weights, inv_freq is a meta tensor with no values).
#    - Store as _rotary_emb_list (a plain Python list, NOT an nn.Module attribute)
#      so that layer.to("meta") does NOT register it as a submodule and does NOT
#      wipe the shared object when each layer is evicted after use.
_orig_airllm_init_model = AirLLMBaseModel.init_model

def _patched_airllm_init_model(self):
    _orig_airllm_init_model(self)
    try:
        inner = getattr(self.model, "model", None)
        if inner is None or not hasattr(inner, "rotary_emb") or not hasattr(inner, "layers"):
            return
        # Recreate rotary embedding with real (non-meta) inv_freq data
        old_rotary = inner.rotary_emb
        new_rotary = type(old_rotary)(config=self.config)
        new_rotary.to(self.running_device)
        inner.rotary_emb = new_rotary
        for _dl in inner.layers:
            # plain list — PyTorch does NOT register list attributes as submodules,
            # so layer.to("meta") will leave this list (and its contents) untouched
            _dl._rotary_emb_list = [new_rotary]
    except Exception as e:
        logger.debug(f"rotary_emb reinit skipped: {e}")

AirLLMBaseModel.init_model = _patched_airllm_init_model


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

                def _find_embed_weight_from_hf_cache():
                    """
                    Scan the local HuggingFace hub cache for embed_tokens.weight.
                    Returns the tensor or None. Does not call HF API.
                    """
                    import os as _os
                    from safetensors import safe_open as _safe_open

                    hf_home = Path(_os.environ.get(
                        "HF_HOME",
                        str(Path.home() / ".cache" / "huggingface")
                    ))
                    hub_dir = hf_home / "hub"
                    model_cache_name = "models--" + model_name.replace("/", "--")
                    snapshots_dir = hub_dir / model_cache_name / "snapshots"

                    if not snapshots_dir.exists():
                        # Fallback: XDG_CACHE_HOME
                        xdg = Path(_os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
                        snapshots_dir = xdg / "huggingface" / "hub" / model_cache_name / "snapshots"

                    if not snapshots_dir.exists():
                        logger.warning(f"HF hub snapshots dir not found: {snapshots_dir}")
                        return None

                    snapshot_dirs = sorted([p for p in snapshots_dir.iterdir() if p.is_dir()])
                    if not snapshot_dirs:
                        logger.warning(f"No snapshot dirs in {snapshots_dir}")
                        return None

                    snapshot_path = snapshot_dirs[-1]
                    shard_files = sorted(
                        list(snapshot_path.glob("model-*.safetensors")) +
                        list(snapshot_path.glob("model.safetensors"))
                    )

                    if not shard_files:
                        logger.warning(f"No safetensors shards in {snapshot_path}")
                        return None

                    for shard_path in shard_files:
                        try:
                            with _safe_open(str(shard_path), framework="pt", device="cpu") as f:
                                for key in f.keys():
                                    if "embed_tokens.weight" in key:
                                        logger.info(f"Found embed weight key '{key}' in {shard_path.name}")
                                        return f.get_tensor(key)
                        except Exception as e:
                            logger.warning(f"Could not read shard {shard_path}: {e}")
                    return None

                def _create_lm_head_shard(split_dir: Path, embed_weight) -> bool:
                    """Save embed_weight as lm_head.safetensors + lm_head.safetensors.done marker.
                    AirLLM's SafetensorModelPersister.model_persist_exist() requires BOTH files."""
                    try:
                        from safetensors.torch import save_file as _st_save
                        lm_head_file = split_dir / "lm_head.safetensors"
                        done_marker = split_dir / "lm_head.safetensors.done"
                        split_dir.mkdir(parents=True, exist_ok=True)
                        _st_save({"lm_head.weight": embed_weight}, str(lm_head_file))
                        done_marker.touch()
                        logger.info(f"Created tied lm_head shard + .done marker: {lm_head_file}")
                        return True
                    except Exception as e:
                        logger.warning(f"Failed to save lm_head shard: {e}", exc_info=True)
                        return False

                def _preemptive_tied_lm_head_fix() -> bool:
                    """
                    Models with tie_word_embeddings=True (e.g. Qwen2.5, Phi) do not store
                    lm_head.weight as a separate tensor — it is aliased to embed_tokens.weight.
                    AirLLM crashes with IndexError when it tries to split that missing tensor.

                    Pre-create lm_head.safetensors from the HF hub cache BEFORE AirLLM's
                    splitting pass so it sees the shard as already present and skips it.
                    """
                    split_dir = layer_shards_path / "splitted_model"
                    lm_head_file = split_dir / "lm_head.safetensors"
                    done_marker = split_dir / "lm_head.safetensors.done"

                    if lm_head_file.exists() and done_marker.exists():
                        logger.info(f"Tied lm_head shard already exists: {lm_head_file}")
                        return True

                    # Check config for tie_word_embeddings
                    try:
                        cfg = AutoConfig.from_pretrained(
                            model_name,
                            token=settings.hf_token or None,
                            trust_remote_code=True,
                            local_files_only=True,
                        )
                        if not getattr(cfg, "tie_word_embeddings", False):
                            logger.info(f"Model {model_name}: tie_word_embeddings=False, skipping lm_head fix")
                            return False
                        logger.info(f"Model {model_name}: tie_word_embeddings=True — pre-creating lm_head shard")
                    except Exception as e:
                        logger.warning(f"Could not check tie_word_embeddings for {model_name}: {e}")
                        return False

                    embed_weight = _find_embed_weight_from_hf_cache()
                    if embed_weight is None:
                        logger.warning(f"Could not find embed_tokens.weight in HF cache for {model_name}")
                        return False

                    return _create_lm_head_shard(split_dir, embed_weight)

                # Pre-create lm_head shard for weight-tied models before AirLLM's
                # splitting pass so the IndexError never occurs.
                await loop.run_in_executor(None, _preemptive_tied_lm_head_fix)

                try:
                    model = await _try_load()
                except IndexError as first_err:
                    # Post-crash fallback: all 38 layers were split+saved before the crash.
                    # Build lm_head.safetensors from the already-split embed_tokens shard.
                    logger.warning(
                        f"IndexError during layer splitting ({first_err}); "
                        "attempting post-crash tied lm_head fix..."
                    )
                    split_dir = layer_shards_path / "splitted_model"
                    lm_head_file = split_dir / "lm_head.safetensors"
                    embed_file = split_dir / "model.embed_tokens.safetensors"
                    fixed = False

                    logger.warning(
                        f"Post-crash state: embed_file.exists()={embed_file.exists()}, "
                        f"lm_head_file.exists()={lm_head_file.exists()}"
                    )

                    if embed_file.exists() and not lm_head_file.exists():
                        try:
                            from safetensors.torch import load_file as _lt_load
                            tensors = _lt_load(str(embed_file))
                            logger.warning(f"Post-crash: embed shard keys={list(tensors.keys())}")

                            # Deliberately avoid `or` between tensors — bool(tensor) raises
                            # RuntimeError for multi-element tensors.
                            embed_weight = None
                            for key_candidate in ["model.embed_tokens.weight", "weight"]:
                                t = tensors.get(key_candidate)
                                if t is not None:
                                    embed_weight = t
                                    break
                            if embed_weight is None and tensors:
                                embed_weight = next(iter(tensors.values()))

                            if embed_weight is not None:
                                fixed = _create_lm_head_shard(split_dir, embed_weight)
                            else:
                                logger.warning(f"Post-crash fix: embed shard has no tensors in {embed_file}")
                        except Exception as fix_err:
                            logger.warning(f"Post-crash tied lm_head fix failed: {fix_err}", exc_info=True)
                    elif lm_head_file.exists():
                        # Preemptive fix created the shard but AirLLM still crashed — the
                        # "re-save all" path overwrote then deleted it. Delete and recreate.
                        logger.warning("Post-crash: lm_head shard exists but load still crashed — recreating from HF cache")
                        lm_head_file.unlink(missing_ok=True)
                        (split_dir / "lm_head.safetensors.done").unlink(missing_ok=True)
                        embed_weight = _find_embed_weight_from_hf_cache()
                        if embed_weight is not None:
                            fixed = _create_lm_head_shard(split_dir, embed_weight)
                    else:
                        logger.warning(f"Post-crash fix: embed_file missing at {embed_file}")

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
