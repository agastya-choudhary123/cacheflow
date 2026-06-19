"""Ollama integration: detect installed models and get their paths."""

import json
import subprocess
from pathlib import Path
from typing import Optional, List

# Ollama lays manifests out as manifests/<registry>/<namespace>/<model>/<tag>,
# e.g. manifests/registry.ollama.ai/library/qwen2.5-coder/7b. Default registry
# pulls (no explicit host/namespace, e.g. `ollama pull qwen3:8b`) always land
# under registry.ollama.ai/library — that's the only root actually present on
# a normal install, so it's the one we scan.
_DEFAULT_REGISTRY = "registry.ollama.ai"
_DEFAULT_NAMESPACE = "library"


def get_ollama_models_dir() -> Path:
    """Get ollama models directory."""
    return Path.home() / ".ollama" / "models"


def list_ollama_models() -> List[str]:
    """
    List all installed ollama models.

    Returns:
        List of model names (e.g., ['qwen2.5-coder:7b', 'mistral:7b'])
    """
    library_dir = get_ollama_models_dir() / "manifests" / _DEFAULT_REGISTRY / _DEFAULT_NAMESPACE
    if not library_dir.exists():
        return []

    models = []
    for model_dir in library_dir.iterdir():
        if not model_dir.is_dir():
            continue
        # Each file under the model dir is a tag (e.g. "7b", "latest"), not
        # just "latest" — the old code only ever matched models pulled
        # without a tag, which silently hid every `name:tag` install (the
        # normal case: `ollama pull qwen2.5-coder:7b`, `ollama pull qwen3:8b`).
        for tag_file in model_dir.iterdir():
            if tag_file.is_file():
                models.append(f"{model_dir.name}:{tag_file.name}")

    return sorted(models)


def get_ollama_model_path(model_name: str) -> Optional[Path]:
    """
    Get the .gguf file path for an installed ollama model.

    Args:
        model_name: Model name (e.g., 'qwen2.5-coder:7b')

    Returns:
        Path to .gguf file, or None if not found
    """
    models_dir = get_ollama_models_dir()
    name, _, tag = model_name.partition(":")
    tag = tag or "latest"

    manifest_path = models_dir / "manifests" / _DEFAULT_REGISTRY / _DEFAULT_NAMESPACE / name / tag
    if not manifest_path.is_file():
        return None

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    for layer in manifest.get("layers", []):
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            digest = layer.get("digest", "")
            blob = models_dir / "blobs" / digest.replace(":", "-")
            if blob.is_file():
                return blob

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
