"""
Configuration for CacheFlow: model paths, context size, and defaults.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CacheFlowConfig(BaseModel):
    """Configuration for an agentgit project."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_path: Path
    model_path: str
    model_name: str
    model_hash: str
    ctx_size: int = 8192
    n_gpu_layers: int = 99
    slot_save_path: Path = Field(default_factory=lambda: Path(".cacheflow/snapshots"))

    @field_validator("ctx_size")
    @classmethod
    def validate_ctx_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("ctx_size must be positive")
        if v > 32768:
            raise ValueError("ctx_size is unreasonably large (> 32768)")
        return v

    @field_validator("n_gpu_layers")
    @classmethod
    def validate_n_gpu_layers(cls, v: int) -> int:
        if v < -1:
            raise ValueError("n_gpu_layers must be >= -1 (-1 = CPU only, >= 0 = GPU layers)")
        return v

    def save(self) -> None:
        """Save config to .cacheflow/config.json."""
        config_file = self.base_path / ".cacheflow" / "config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert paths to strings for JSON serialization
        data = {
            "model_path": self.model_path,
            "model_name": self.model_name,
            "model_hash": self.model_hash,
            "ctx_size": self.ctx_size,
            "n_gpu_layers": self.n_gpu_layers,
            "slot_save_path": str(self.slot_save_path),
        }

        with open(config_file, "w") as f:
            json.dump(data, f, indent=2)


def compute_model_hash(model_path: str, bytes_to_read: int = 10 * 1024 * 1024) -> str:
    """
    Compute SHA256 hash of first N bytes of model file.

    Args:
        model_path: Path to GGUF model file
        bytes_to_read: Number of bytes to hash (default: 10MB for speed)

    Returns:
        Hex digest of SHA256 hash
    """
    sha256 = hashlib.sha256()
    with open(model_path, "rb") as f:
        data = f.read(bytes_to_read)
        sha256.update(data)
    return sha256.hexdigest()


def save_config(config: CacheFlowConfig) -> None:
    """
    Save config to .cacheflow/config.json.

    Args:
        config: CacheFlowConfig instance to save
    """
    config.save()


def load_config(base_path: Path) -> CacheFlowConfig:
    """
    Load config from .cacheflow/config.json.

    Args:
        base_path: Project root path

    Returns:
        CacheFlowConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    config_file = base_path / ".cacheflow" / "config.json"

    if not config_file.exists():
        raise FileNotFoundError(
            f"Config not found at {config_file}. Run 'agentgit init' first."
        )

    with open(config_file) as f:
        data = json.load(f)

    return CacheFlowConfig(
        base_path=base_path,
        model_path=data["model_path"],
        model_name=data["model_name"],
        model_hash=data["model_hash"],
        ctx_size=data.get("ctx_size", 8192),
        n_gpu_layers=data.get("n_gpu_layers", 99),
        slot_save_path=Path(data.get("slot_save_path", ".cacheflow/snapshots")),
    )


def find_gguf_for_model(model_name: str) -> Optional[str]:
    """
    Find GGUF file for a model name.

    Searches common paths:
    - ~/.ollama/models/blobs/
    - ~/Library/Caches/llama.cpp/
    - ~/.cache/lm-studio/models/
    - Current directory

    Args:
        model_name: Model name (e.g., "llama3.1:8b")

    Returns:
        Path to GGUF file or None if not found
    """
    search_paths = [
        Path.home() / ".ollama/models/blobs",
        Path.home() / "Library/Caches/llama.cpp",
        Path.home() / ".cache/lm-studio/models",
        Path.cwd(),
    ]

    for base_path in search_paths:
        if not base_path.exists():
            continue

        # Look for GGUF files with model name in them
        for gguf_file in base_path.rglob("*.gguf"):
            if model_name.lower() in gguf_file.name.lower():
                return str(gguf_file)

        # Also look for ollama blob symlinks (files with model name)
        for file_path in base_path.iterdir():
            if file_path.is_file() and model_name.lower() in file_path.name.lower():
                return str(file_path)

    return None
