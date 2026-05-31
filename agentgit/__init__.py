"""AgentGit: Persistent KV cache memory for AI agents."""

__version__ = "0.1.0"


# Exception hierarchy
class AgentGitError(Exception):
    """Base exception for all AgentGit errors."""
    pass


class ServerError(AgentGitError):
    """Errors related to llama-server process management."""
    pass


class SnapshotError(AgentGitError):
    """Errors related to snapshot save/restore/validation."""
    pass


class ModelMismatchError(AgentGitError):
    """Model hash or version mismatch error."""
    pass


class VersionMismatchError(AgentGitError):
    """llama.cpp version mismatch error."""
    pass


class AgentSessionError(AgentGitError):
    """Errors during agent session execution."""
    pass


__all__ = [
    "AgentGitError",
    "ServerError",
    "SnapshotError",
    "ModelMismatchError",
    "VersionMismatchError",
    "AgentSessionError",
]
