"""Tests for AgentGit configuration management."""

import json
import tempfile
from pathlib import Path

import pytest

from agentgit.config import (
    AgentGitConfig,
    compute_model_hash,
    find_gguf_for_model,
    load_config,
    save_config,
)


def test_compute_model_hash():
    """Test model hash computation on a temporary file."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        # Write 20MB of data
        f.write(b"x" * (20 * 1024 * 1024))
        temp_path = f.name

    try:
        hash1 = compute_model_hash(temp_path)
        hash2 = compute_model_hash(temp_path)
        assert hash1 == hash2  # Should be deterministic
        assert len(hash1) == 64  # SHA256 hex is 64 chars
    finally:
        Path(temp_path).unlink()


def test_agentgit_config_creation():
    """Test creating an AgentGitConfig instance."""
    config = AgentGitConfig(
        base_path=Path("/tmp/test"),
        model_path="/path/to/model.gguf",
        model_name="llama3.1:8b",
        model_hash="abc123",
        ctx_size=8192,
        n_gpu_layers=99,
    )
    assert config.base_path == Path("/tmp/test")
    assert config.model_name == "llama3.1:8b"
    assert config.ctx_size == 8192
    assert config.n_gpu_layers == 99


def test_config_save_and_load():
    """Test saving and loading configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)
        (base_path / ".agentgit").mkdir(parents=True)

        # Create and save config
        config = AgentGitConfig(
            base_path=base_path,
            model_path="/path/to/model.gguf",
            model_name="llama3.1:8b",
            model_hash="abc123def456",
            ctx_size=4096,
            n_gpu_layers=50,
            slot_save_path=base_path / ".agentgit/snapshots",
        )
        save_config(config)

        # Verify file was written
        config_file = base_path / ".agentgit" / "config.json"
        assert config_file.exists()

        # Load and verify
        loaded = load_config(base_path)
        assert loaded.model_path == "/path/to/model.gguf"
        assert loaded.model_name == "llama3.1:8b"
        assert loaded.model_hash == "abc123def456"
        assert loaded.ctx_size == 4096
        assert loaded.n_gpu_layers == 50


def test_load_config_missing_file():
    """Test that loading from missing config raises FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            load_config(Path(tmpdir))


def test_find_gguf_for_model_not_found():
    """Test that find_gguf_for_model returns None when not found."""
    result = find_gguf_for_model("nonexistent_model_xyz123")
    assert result is None


def test_default_values():
    """Test that default values are set correctly."""
    config = AgentGitConfig(
        base_path=Path("/tmp"),
        model_path="/path/to/model.gguf",
        model_name="test",
        model_hash="test",
    )
    assert config.ctx_size == 8192
    assert config.n_gpu_layers == 99
