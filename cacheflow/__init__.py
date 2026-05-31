"""CacheFlow: Persistent KV cache memory for AI agents."""

__version__ = "0.1.0"


# Exception hierarchy
class CacheFlowError(Exception):
    """Base exception for all CacheFlow errors."""
    pass


class ServerError(CacheFlowError):
    """Errors related to llama-server process management."""
    pass


class SnapshotError(CacheFlowError):
    """Errors related to snapshot save/restore/validation."""
    pass


class ModelMismatchError(CacheFlowError):
    """Model hash or version mismatch error."""
    pass


class VersionMismatchError(CacheFlowError):
    """llama.cpp version mismatch error."""
    pass


class AgentSessionError(CacheFlowError):
    """Errors during agent session execution."""
    pass


__all__ = [
    "CacheFlowError",
    "ServerError",
    "SnapshotError",
    "ModelMismatchError",
    "VersionMismatchError",
    "AgentSessionError",
]
