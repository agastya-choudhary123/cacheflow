"""Tests for ollama model discovery (manifest layout, tag handling, blob lookup)."""

import json
import tempfile
from pathlib import Path

import pytest

from cacheflow import ollama


@pytest.fixture
def fake_ollama_home(monkeypatch):
    """Build a real `~/.ollama/models` layout under a temp dir and point at it."""
    with tempfile.TemporaryDirectory() as tmp:
        models_dir = Path(tmp) / "models"
        library = models_dir / "manifests" / "registry.ollama.ai" / "library"

        def make_model(name: str, tag: str, digest: str, size: int = 123):
            model_dir = library / name
            model_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "schemaVersion": 2,
                "layers": [
                    {"mediaType": "application/vnd.ollama.image.model", "digest": f"sha256:{digest}", "size": size},
                    {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:deadbeef", "size": 1},
                ],
            }
            (model_dir / tag).write_text(json.dumps(manifest))
            blobs = models_dir / "blobs"
            blobs.mkdir(parents=True, exist_ok=True)
            (blobs / f"sha256-{digest}").write_bytes(b"fake-gguf-bytes")

        make_model("qwen2.5-coder", "7b", "aaa111")
        make_model("qwen3", "8b", "bbb222")

        monkeypatch.setattr(ollama, "get_ollama_models_dir", lambda: models_dir)
        yield models_dir


def test_list_ollama_models_finds_tagged_models(fake_ollama_home):
    # Regression: the old implementation only ever looked for a file literally
    # named "latest", so every normal `ollama pull name:tag` install (the
    # common case) was invisible to `cf init`'s model discovery.
    models = ollama.list_ollama_models()
    assert models == ["qwen2.5-coder:7b", "qwen3:8b"]


def test_get_ollama_model_path_resolves_blob_via_manifest(fake_ollama_home):
    path = ollama.get_ollama_model_path("qwen3:8b")
    assert path is not None
    assert path.name == "sha256-bbb222"
    assert path.read_bytes() == b"fake-gguf-bytes"


def test_get_ollama_model_path_unknown_model_returns_none(fake_ollama_home):
    assert ollama.get_ollama_model_path("nonexistent:1b") is None


def test_get_ollama_model_path_no_models_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(ollama, "get_ollama_models_dir", lambda: tmp_path / "missing")
    assert ollama.list_ollama_models() == []
    assert ollama.get_ollama_model_path("qwen3:8b") is None
