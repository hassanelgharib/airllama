"""Configuration management using pydantic-settings."""

from pathlib import Path
from typing import Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 11434
    
    # Model Configuration
    model_cache_dir: str = "~/.cache/airllm"
    default_compression: Optional[str] = None  # 4bit or 8bit require bitsandbytes + CUDA
    max_loaded_models: int = 1

    @field_validator("default_compression", mode="before")
    @classmethod
    def normalise_compression(cls, v):
        """Treat empty string, 'none', 'null' as no compression."""
        if v is None:
            return None
        if str(v).strip().lower() in ("", "none", "null"):
            return None
        return v
    
    # HuggingFace Configuration
    hf_token: Optional[str] = None
    
    # Performance Settings
    default_max_length: int = 2048
    default_max_new_tokens: int = 512
    
    @property
    def cache_path(self) -> Path:
        """Get the expanded cache directory path."""
        return Path(self.model_cache_dir).expanduser()
    
    @property
    def models_registry_path(self) -> Path:
        """Get the path to the models registry file."""
        return self.cache_path / "models_registry.json"


# Global settings instance
settings = Settings()
