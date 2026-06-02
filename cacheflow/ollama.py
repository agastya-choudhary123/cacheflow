"""Ollama integration: detect installed models and get their paths."""

import subprocess
from pathlib import Path
from typing import Optional, List


def get_ollama_models_dir() -> Path:
    """Get ollama models directory."""
    return Path.home() / ".ollama" / "models"


def list_ollama_models() -> List[str]:
    """
    List all installed ollama models.

    Returns:
        List of model names (e.g., ['llama3.1:8b', 'mistral:7b'])
    """
    models_dir = get_ollama_models_dir()

    if not models_dir.exists():
        return []

    models = []
    # Ollama stores manifests in manifests/library/
    manifests_dir = models_dir / "manifests" / "library"

    if manifests_dir.exists():
        for model_dir in manifests_dir.iterdir():
            if model_dir.is_dir():
                # Model name is directory name, check for latest version
                latest_file = model_dir / "latest"
                if latest_file.exists():
                    models.append(model_dir.name)

    return sorted(models)


def get_ollama_model_path(model_name: str) -> Optional[Path]:
    """
    Get the .gguf file path for an installed ollama model.

    Args:
        model_name: Model name (e.g., 'llama3.1:8b')

    Returns:
        Path to .gguf file, or None if not found
    """
    models_dir = get_ollama_models_dir()

    if not models_dir.exists():
        return None

    # Ollama stores blobs in blobs/sha256-<hash>
    # The manifest points to which blob to use
    # For simplicity, look for .gguf files in the models directory structure

    # First try: direct model name as directory
    model_dir = models_dir / "models" / model_name
    if model_dir.exists():
        for gguf_file in model_dir.rglob("*.gguf"):
            return gguf_file

    # Fallback: search all .gguf files in blobs
    blobs_dir = models_dir / "blobs"
    if blobs_dir.exists():
        for gguf_file in blobs_dir.glob("sha256-*"):
            if gguf_file.is_file():
                return gguf_file

    return None


def ollama_is_installed() -> bool:
    """Check if ollama CLI is available."""
    try:
        subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            timeout=2,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_running_ollama_url() -> Optional[str]:
    """Check if ollama server is running and return its URL."""
    try:
        subprocess.run(
            ["curl", "-s", "http://127.0.0.1:11434/api/tags"],
            capture_output=True,
            timeout=2,
        )
        return "http://127.0.0.1:11434"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
