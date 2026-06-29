"""
Shared llama-cpp-python primitives used by the in-process engine (engine.py):
the binary KV-cache snapshot format (_write_snapshot/_read_snapshot) and
CooperativeSlotManager, which time-multiplexes one loaded model across
multiple virtual slots by swapping KV state on context switch.
"""

import ctypes
import struct
from pathlib import Path
from typing import Optional, Dict
import threading
from dataclasses import dataclass

import numpy as np

try:
    import llama_cpp
    from llama_cpp import Llama, LlamaState
except ImportError:
    raise ImportError("llama-cpp-python not installed. Run: pip install llama-cpp-python")


# ── Binary snapshot format ────────────────────────────────────────────────────
# Version 4 layout (no pickle — pure struct + raw arrays):
#
#   4 bytes  magic = b"CFKV"
#   4 bytes  version = 4 (LE uint32)
#   8 bytes  n_tokens (LE uint64)
#   8 bytes  seed (LE uint64)
#   8 bytes  seq_size (LE uint64)
#   <seq_size bytes>  per-sequence KV state (llama_state_seq_get_data, seq 0)
#   8 bytes  input_ids_len (LE uint64)  — number of int32 elements
#   <input_ids_len * 4 bytes>  int32 LE array
#
# Why seq state instead of full state (v3): Llama.save_state() serializes
# llama_get_state_size(), which is sized to the *entire* context buffer (n_ctx,
# e.g. 16384 tokens) regardless of how many tokens are actually populated. The
# per-sequence API serializes only the n_tokens of KV that are live, so a snapshot
# for a 9k-token prime drops from the full 16384-ctx buffer to ~9k worth of KV —
# shrinking both the save write and the restore read on the cold/prime path.
#
# scores are not stored at all: they are last-batch logit outputs that the next
# forward pass overwrites, so they're reconstructed as zeros on apply.
#
_SNAPSHOT_MAGIC = b"CFKV"
_SNAPSHOT_VERSION = 4


@dataclass
class _Snapshot:
    """A deserialized snapshot. Knows how to apply itself onto a live model.

    Carries either compact per-sequence KV bytes (v4) or a full LlamaState (v3,
    read-only backward compat). Apply clears the model's KV and re-seeds it.
    """
    n_tokens: int
    seed: int
    input_ids: np.ndarray
    seq_data: Optional[bytes] = None          # v4: compact per-sequence KV
    full_state: Optional["LlamaState"] = None  # v3: legacy full-context state

    def apply_to(self, model: "Llama") -> None:
        if self.full_state is not None:
            # Legacy v3 snapshot: full-context state set via the high-level wrapper.
            model.load_state(self.full_state)
            return

        # v4: clear KV, then splice the compact per-sequence state back into seq 0.
        model._ctx.kv_cache_clear()
        size = len(self.seq_data)
        buf = (ctypes.c_uint8 * size).from_buffer_copy(self.seq_data)
        nread = llama_cpp.llama_state_seq_set_data(model._ctx.ctx, buf, size, 0)
        if nread == 0:
            raise RuntimeError("llama_state_seq_set_data failed to restore KV cache")
        # Re-sync the high-level wrapper bookkeeping so prefix-matching works.
        model.input_ids[: self.n_tokens] = self.input_ids[: self.n_tokens]
        model.scores[:, :] = 0.0
        model.n_tokens = self.n_tokens
        model._seed = self.seed


def _capture_compact(model: "Llama") -> "_Snapshot":
    """Capture the active slot's KV into a compact in-memory _Snapshot.

    Uses the per-sequence API (llama_state_seq_get_data) — the same path as
    v4 disk snapshots — so only the live tokens are captured, not the full
    n_ctx-sized buffer that model.save_state() / llama_copy_state_data returns.
    For a typical primed codebase (several thousand tokens) this is ~8 MB
    instead of several hundred MB per slot, which matters a lot when 8 agents
    are active concurrently.
    """
    ctx = model._ctx.ctx
    seq_size = llama_cpp.llama_state_seq_get_size(ctx, 0)
    buf = (ctypes.c_uint8 * seq_size)()
    nwritten = llama_cpp.llama_state_seq_get_data(ctx, buf, seq_size, 0)
    if nwritten == 0:
        raise RuntimeError("llama_state_seq_get_data returned no data")
    n = model.n_tokens
    return _Snapshot(
        n_tokens=n,
        seed=model._seed,
        input_ids=model.input_ids[:n].copy(),
        seq_data=bytes(buf[:nwritten]),
    )


def _write_snapshot(filepath: Path, snap: "_Snapshot") -> None:
    """Write a compact _Snapshot to disk.

    `snap` must have been captured via _capture_compact (seq_data populated).
    Writing from the already-captured bytes avoids a second C-API round-trip.
    """
    input_ids_bytes = snap.input_ids[:snap.n_tokens].astype("<i4").tobytes()

    with open(filepath, "wb") as f:
        f.write(_SNAPSHOT_MAGIC)
        f.write(struct.pack("<I", _SNAPSHOT_VERSION))
        f.write(struct.pack("<Q", snap.n_tokens))
        f.write(struct.pack("<Q", snap.seed))
        f.write(struct.pack("<Q", len(snap.seq_data)))
        f.write(snap.seq_data)
        f.write(struct.pack("<Q", snap.n_tokens))
        f.write(input_ids_bytes)


def _read_snapshot(filepath: Path) -> "_Snapshot":
    """Deserialize a snapshot from disk. No code execution — pure binary reads."""
    with open(filepath, "rb") as f:
        magic = f.read(4)
        if magic != _SNAPSHOT_MAGIC:
            raise ValueError(f"Not a CacheFlow snapshot (magic={magic!r}).")
        version = struct.unpack("<I", f.read(4))[0]

        if version == 4:
            n_tokens = struct.unpack("<Q", f.read(8))[0]
            seed = struct.unpack("<Q", f.read(8))[0]
            seq_size = struct.unpack("<Q", f.read(8))[0]
            seq_bytes = f.read(seq_size)
            if len(seq_bytes) != seq_size:
                raise ValueError(f"Truncated seq state: expected {seq_size}, got {len(seq_bytes)}")
            input_ids_len = struct.unpack("<Q", f.read(8))[0]
            input_ids_bytes = f.read(input_ids_len * 4)
            if len(input_ids_bytes) != input_ids_len * 4:
                raise ValueError("Truncated input_ids")
            input_ids = np.frombuffer(input_ids_bytes, dtype="<i4").copy()
            return _Snapshot(
                n_tokens=int(n_tokens), seed=int(seed),
                input_ids=input_ids, seq_data=seq_bytes,
            )

        if version == 3:
            # Legacy full-context snapshot — readable for backward compat.
            n_tokens = struct.unpack("<Q", f.read(8))[0]
            seed = struct.unpack("<Q", f.read(8))[0]
            llama_state_size = struct.unpack("<Q", f.read(8))[0]
            llama_state_bytes = f.read(llama_state_size)
            if len(llama_state_bytes) != llama_state_size:
                raise ValueError(f"Truncated llama_state: expected {llama_state_size}, got {len(llama_state_bytes)}")
            input_ids_len = struct.unpack("<Q", f.read(8))[0]
            input_ids_bytes = f.read(input_ids_len * 4)
            if len(input_ids_bytes) != input_ids_len * 4:
                raise ValueError("Truncated input_ids")
            input_ids = np.frombuffer(input_ids_bytes, dtype="<i4").copy()
            scores_rows, scores_cols = struct.unpack("<QQ", f.read(16))
            scores = np.zeros((scores_rows, scores_cols), dtype=np.float32)
            full_state = LlamaState(
                input_ids=input_ids, scores=scores, n_tokens=int(n_tokens),
                llama_state=llama_state_bytes, llama_state_size=llama_state_size,
                seed=int(seed),
            )
            return _Snapshot(
                n_tokens=int(n_tokens), seed=int(seed),
                input_ids=input_ids, full_state=full_state,
            )

        raise ValueError(
            f"Unsupported snapshot version {version} (expected {_SNAPSHOT_VERSION}). "
            "Re-prime the agent to generate a new snapshot."
        )


# ── Cooperative slot manager ──────────────────────────────────────────────────

class CooperativeSlotManager:
    """Time-multiplexes multiple agents onto a single model via state swapping.

    Analogous to an OS scheduler: each agent's KV cache is its "process state,"
    saved/restored on context switch. Only one agent runs at a time.

    _slot_states stores compact _Snapshot objects (per-sequence KV bytes, same
    format as v4 disk snapshots) rather than full LlamaState objects from
    model.save_state(). model.save_state() copies the entire n_ctx-sized KV
    buffer regardless of how many tokens are populated; _capture_compact only
    serializes the live tokens via llama_state_seq_get_data. For 8 primed agents
    at 8192 ctx this is ~64 MB total in Python heap instead of several GB.
    """

    def __init__(self, model: "Llama"):
        self.model = model
        self._active_slot: Optional[int] = None
        self._slot_states: Dict[int, Optional["_Snapshot"]] = {}
        self._lock = threading.Lock()

    def switch_to(self, slot_id: int) -> None:
        """Context-switch: flush current slot, restore target slot."""
        with self._lock:
            if self._active_slot == slot_id:
                return
            if self._active_slot is not None:
                self._slot_states[self._active_slot] = _capture_compact(self.model)
            target = self._slot_states.get(slot_id)
            if target is not None:
                target.apply_to(self.model)
            else:
                self.model.reset()
            self._active_slot = slot_id

    def invalidate(self, slot_id: int) -> None:
        """Discard saved state for a slot (after explicit reset/erase)."""
        with self._lock:
            self._slot_states.pop(slot_id, None)
            if self._active_slot == slot_id:
                self._active_slot = None

    def snapshot_state(self, slot_id: int) -> Optional["_Snapshot"]:
        """Return the current compact in-memory snapshot for a slot."""
        with self._lock:
            if self._active_slot == slot_id:
                snap = _capture_compact(self.model)
                self._slot_states[slot_id] = snap
                return snap
            return self._slot_states.get(slot_id)


@dataclass
class Slot:
    """Represents a model slot/context."""
    id: int
    n_ctx: int
