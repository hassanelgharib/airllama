"""Configuration management using pydantic-settings."""

from pathlib import Path
from typing import Optional
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
    default_compression: Optional[str] = "4bit"  # none, 4bit, or 8bit
    max_loaded_models: int = 1
    
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
