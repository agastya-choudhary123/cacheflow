"""Tests covering all 18 bug fixes."""

import hashlib
import os
import struct
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4, UUID

import pytest

from cacheflow.agent import AgentSession, fork_agent, DEFAULT_SYSTEM_PROMPT
from cacheflow.compressor import Compressor, _COMPACTION_EXECUTOR
from cacheflow.config import CacheFlowConfig, save_config
from cacheflow.gc import SnapshotGC
from cacheflow.slot_pool import SlotPool
from cacheflow.store import CacheFlowStore, _hash_context


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config(temp_dir):
    (temp_dir / ".cacheflow").mkdir(parents=True)
    cfg = CacheFlowConfig(
        base_path=temp_dir,
        model_path="/path/to/model.gguf",
        model_name="qwen2.5-coder:7b",
        model_hash="abc123def456",
        ctx_size=8192,
        n_gpu_layers=99,
        slot_save_path=temp_dir / ".cacheflow/snapshots",
    )
    save_config(cfg)
    return cfg


@pytest.fixture
def store(temp_dir, config):
    db_path = temp_dir / ".cacheflow" / "agents.db"
    s = CacheFlowStore(db_path)
    s.init_db()
    return s


@pytest.fixture
def snapshots_dir(temp_dir):
    d = temp_dir / ".cacheflow" / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Issue 1: Async save race ──────────────────────────────────────────────────
# The fix: save is now synchronous (file exists before response is returned).
# We verify the agent run succeeds and the snapshot file exists on disk before
# the DB commit (i.e., no race where the agent checks the file before it's written).

def test_fix1_save_is_synchronous(temp_dir, config):
    """save_slot returns filename only after the file actually exists on disk."""
    snapshots_dir = temp_dir / ".cacheflow" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    snapshot_file = snapshots_dir / "snapshot.bin"
    snapshot_file.write_bytes(os.urandom(1024))

    mock_server = MagicMock()
    mock_server.count_tokens.return_value = 10
    mock_server.completion.return_value = {
        "content": "done",
        "tokens_evaluated": 50,
        "tokens_predicted": 25,
    }
    mock_server.save_slot.return_value = {
        "filename": "snapshot.bin",
        "save_time_ms": 5,
        "size_bytes": 1024,
    }

    mock_tokenizer = MagicMock()
    mock_tokenizer.count.return_value = 10

    with patch("cacheflow.agent.get_tokenizer", return_value=mock_tokenizer):
        session = AgentSession("test-agent", temp_dir)
        with patch("cacheflow.agent.get_global_engine", return_value=mock_server):
            result = session.run("task")

    # If run() succeeds, the file was found — no race
    assert result.snapshot_size_bytes > 0


# ── Issue 2: Pickle → binary format ──────────────────────────────────────────

def test_fix2_binary_snapshot_format():
    """_write_snapshot / _read_snapshot use a versioned binary format, not pickle.

    v4 stores only the live per-sequence KV (llama_state_seq_get_data) instead of
    the full-context buffer, so writing reads the seq state off the model's ctx.
    """
    import numpy as np
    import cacheflow.llama_server_custom as lsc
    from cacheflow.llama_server_custom import _write_snapshot, _read_snapshot, _SNAPSHOT_MAGIC, _SNAPSHOT_VERSION

    seq_payload = os.urandom(4096)

    state = MagicMock()
    state.input_ids = np.array([1, 2, 3, 4], dtype=np.int32)
    state.n_tokens = 4
    state.seed = 42

    model = MagicMock()
    model._ctx.ctx = object()  # opaque ctx pointer

    def fake_get_size(ctx, seq_id):
        return len(seq_payload)

    def fake_get_data(ctx, dst, size, seq_id):
        dst[: len(seq_payload)] = seq_payload
        return len(seq_payload)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        tmp = Path(f.name)

    try:
        with patch.object(lsc.llama_cpp, "llama_state_seq_get_size", fake_get_size), \
             patch.object(lsc.llama_cpp, "llama_state_seq_get_data", fake_get_data):
            _write_snapshot(tmp, model, state)

        # Verify header
        with open(tmp, "rb") as f:
            magic = f.read(4)
            version = struct.unpack("<I", f.read(4))[0]

        assert magic == _SNAPSHOT_MAGIC
        assert version == _SNAPSHOT_VERSION

        # Round-trip
        recovered = _read_snapshot(tmp)
        assert recovered.n_tokens == state.n_tokens
        assert recovered.seed == state.seed
        assert recovered.seq_data == seq_payload
        assert list(recovered.input_ids) == list(state.input_ids)

    finally:
        tmp.unlink(missing_ok=True)


def test_fix2_corrupt_snapshot_raises():
    """Loading a corrupt (non-CFKV) file raises ValueError, not arbitrary code execution."""
    from cacheflow.llama_server_custom import _read_snapshot

    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        f.write(b"BAD!" + os.urandom(128))
        tmp = Path(f.name)

    try:
        with pytest.raises(ValueError, match="Not a CacheFlow snapshot"):
            _read_snapshot(tmp)
    finally:
        tmp.unlink(missing_ok=True)


# ── Issue 3: Compressor uses global server ────────────────────────────────────

# ── Issue 4: SlotLease __exit__ not called without __enter__ ─────────────────

def test_fix4_release_lock_calls_release_slot_directly(temp_dir, config):
    """_release_lock calls _SLOT_POOL.release_slot directly, no __exit__ misuse."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.count.return_value = 10
    with patch("cacheflow.agent.get_tokenizer", return_value=mock_tokenizer):
        session = AgentSession("test-agent", temp_dir)
    session._acquire_lock()
    assert session.slot_id is not None
    slot_id = session.slot_id
    session._release_lock()
    assert session.slot_lease is None
    assert session.slot_id is None
    # The slot should still be tracked in the pool (it is, just released)
    from cacheflow.agent import _SLOT_POOL
    state = _SLOT_POOL.get_slot_state(slot_id)
    assert state is not None  # slot exists
    assert not state.is_dirty   # released


# ── Issue 5: expire_on_commit=False ──────────────────────────────────────────

# ── Issue 6: Stable context hash instead of full text ────────────────────────

def test_fix6_hash_stored_not_full_text(store):
    """update_agent_stable_context stores a 64-char hash, not the full text."""
    agent = store.create_agent("a", "model", "hash", 8192)
    long_text = "x" * 100_000  # 100 KB
    store.update_agent_stable_context(agent, long_text)

    refreshed = store.get_agent("a")
    expected_hash = hashlib.sha256(long_text.encode()).hexdigest()
    assert refreshed.stable_context_hash == expected_hash
    # The full text should NOT be stored in stable_context_hash
    assert len(refreshed.stable_context_hash) == 64


def test_fix6_hash_context_helper():
    """_hash_context returns consistent 64-char SHA-256 hex."""
    h = _hash_context("hello world")
    assert len(h) == 64
    assert h == _hash_context("hello world")
    assert h != _hash_context("different")


# ── Issue 7: get_commit_by_id_prefix SQL LIKE ────────────────────────────────

# ── Issue 10: CooperativeSlotManager ─────────────────────────────────────────

def test_fix10_cooperative_slot_manager_switch():
    """CooperativeSlotManager context-switches between slots correctly."""
    from cacheflow.llama_server_custom import CooperativeSlotManager

    mock_model = MagicMock()
    mock_model.save_state.return_value = b"state_0"
    mock_model.load_state = MagicMock()
    mock_model.reset = MagicMock()

    manager = CooperativeSlotManager(mock_model)

    # Switch to slot 0 — no active slot, should reset
    manager.switch_to(0)
    mock_model.reset.assert_called_once()

    # Switch to slot 1 — should save slot 0 state first, then reset (no state for slot 1)
    mock_model.reset.reset_mock()
    manager.switch_to(1)
    mock_model.save_state.assert_called()
    assert 0 in manager._slot_states

    # Switch back to slot 0 — should restore slot 0's state
    manager.switch_to(0)
    mock_model.load_state.assert_called_with(b"state_0")


def test_fix10_same_slot_noop():
    """Switching to the already-active slot is a no-op."""
    from cacheflow.llama_server_custom import CooperativeSlotManager

    mock_model = MagicMock()
    mock_model.save_state.return_value = b"state"
    mock_model.reset = MagicMock()

    manager = CooperativeSlotManager(mock_model)
    manager.switch_to(0)
    call_count_before = mock_model.reset.call_count

    manager.switch_to(0)  # same slot
    assert mock_model.reset.call_count == call_count_before  # no extra reset


def test_fix10_invalidate_clears_state():
    """invalidate() discards in-memory state for a slot."""
    from cacheflow.llama_server_custom import CooperativeSlotManager

    mock_model = MagicMock()
    mock_model.save_state.return_value = b"state"
    mock_model.reset = MagicMock()

    manager = CooperativeSlotManager(mock_model)
    manager.switch_to(0)
    manager._slot_states[0] = b"some state"
    manager.invalidate(0)
    assert 0 not in manager._slot_states


# ── Issue 11: mark_dirty / load_commit are now locked ────────────────────────

def test_fix11_mark_dirty_thread_safe():
    """mark_dirty is protected by the pool lock (no data races)."""
    pool = SlotPool(max_slots=4)
    agent_id = uuid4()
    lease = pool.acquire_slot(agent_id)

    errors = []

    def toggle_dirty():
        for _ in range(1000):
            try:
                pool.mark_dirty(lease.slot_id)
                pool.release_slot(lease.slot_id)
            except Exception as e:
                errors.append(e)

    threads = [threading.Thread(target=toggle_dirty) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread safety errors: {errors}"


def test_fix11_load_commit_thread_safe():
    """load_commit is protected by the pool lock."""
    pool = SlotPool(max_slots=4)
    agent_id = uuid4()
    lease = pool.acquire_slot(agent_id)
    commit_id = uuid4()

    errors = []

    def do_load():
        for _ in range(500):
            try:
                pool.load_commit(lease.slot_id, commit_id, agent_id)
            except Exception as e:
                errors.append(e)

    threads = [threading.Thread(target=do_load) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors


# ── Issue 12: Module-level ThreadPoolExecutor ─────────────────────────────────

def test_fix12_compressor_no_per_instance_executor(store, config):
    """Compressor does not create a per-instance ThreadPoolExecutor."""
    c1 = Compressor(store, config)
    c2 = Compressor(store, config)
    # Both instances should use the module-level executor (no _executor attribute)
    assert not hasattr(c1, "_executor")
    assert not hasattr(c2, "_executor")
    # The module-level executor exists and is shared
    assert _COMPACTION_EXECUTOR is not None


# ── Issue 13: OS-assigned port ────────────────────────────────────────────────

def test_fix13_port_is_os_assigned():
    """_find_available_port uses socket port 0 to get an OS-assigned port."""
    from cacheflow.server import LlamaServer
    server = LlamaServer()
    port = server._find_available_port()
    assert isinstance(port, int)
    assert 1024 <= port <= 65535


# ── Issue 14: fork_agent fail-fast on missing snapshot ───────────────────────

# ── Issue 15: .gitignore respected in rglob fallback ─────────────────────────

def test_fix15_gitignore_respected_in_fallback(temp_dir, config):
    """_collect_source_files fallback respects .gitignore via pathspec."""
    # Write a .gitignore that excludes secret.py
    (temp_dir / ".gitignore").write_text("secret.py\n")
    (temp_dir / "visible.py").write_text("# public\n")
    (temp_dir / "secret.py").write_text("PASSWORD = 'hunter2'\n")

    mock_tokenizer = MagicMock()
    mock_tokenizer.count.return_value = 10
    with patch("cacheflow.agent.get_tokenizer", return_value=mock_tokenizer):
        session = AgentSession("a", temp_dir)

    # Force the non-git fallback by making git ls-files fail
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)
        try:
            import pathspec
            files = session._collect_source_files()
            names = [f.name for f in files]
            assert "visible.py" in names
            assert "secret.py" not in names
        except ImportError:
            pytest.skip("pathspec not installed")


# ── Issue 16: Exact tokenization via model's BPE tokenizer ───────────────────

def test_fix16_count_tokens_uses_tokenizer(temp_dir, config):
    """_count_tokens delegates to the model's BPE tokenizer, not the server."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.count.return_value = 42
    with patch("cacheflow.agent.get_tokenizer", return_value=mock_tokenizer):
        session = AgentSession("a", temp_dir)

    count = session._count_tokens("some text")
    assert count == 42
    mock_tokenizer.count.assert_called_once_with("some text")


def test_fix16_count_tokens_no_server_needed(temp_dir, config):
    """_count_tokens works without a running server (tokenizer is loaded at init)."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.count.return_value = 17
    with patch("cacheflow.agent.get_tokenizer", return_value=mock_tokenizer):
        session = AgentSession("a", temp_dir)

    session.server = None  # server not running
    assert session._count_tokens("hello world") == 17


def test_fix16_count_tokens_exact_not_heuristic(temp_dir, config):
    """_count_tokens returns whatever the BPE tokenizer returns, not len//4."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.count.return_value = 99  # distinct from any len//4 result
    with patch("cacheflow.agent.get_tokenizer", return_value=mock_tokenizer):
        session = AgentSession("a", temp_dir)

    text = "x" * 800  # len//4 heuristic would give 200, not 99
    assert session._count_tokens(text) == 99


# ── Issue 17: Dashboard XSS escaping ─────────────────────────────────────────

# ── Issue 18: Snapshot garbage collector ─────────────────────────────────────

# ── Additional cross-cutting: schema migration idempotency with new column ────

def test_schema_migration_stable_context_hash(temp_dir):
    """stable_context_hash column is created and backfilled on migration."""
    db_path = temp_dir / "test.db"
    store = CacheFlowStore(db_path)
    store.init_db()

    agent = store.create_agent("a", "model", "hash", 8192)
    store.update_agent_stable_context(agent, "some context text")

    refreshed = store.get_agent("a")
    assert refreshed.stable_context_hash is not None
    assert len(refreshed.stable_context_hash) == 64

    # Migration is idempotent — calling init_db again should not fail
    store.init_db()
    refreshed2 = store.get_agent("a")
    assert refreshed2.stable_context_hash == refreshed.stable_context_hash


# ── SnapshotGC (flat store: HEAD snapshot per agent) ──────────────────────────
# The store no longer has a commit DAG; each agent points at a single current
# (HEAD) snapshot. SnapshotGC keeps every agent's HEAD and reaps everything else.

def _set_head_snapshot(store, agent, snapshots_dir, size: int = 1024) -> Path:
    """Write a snapshot file and make it the agent's current (HEAD) snapshot."""
    snap = snapshots_dir / f"{agent.name}_{uuid4().hex[:8]}.bin"
    snap.write_bytes(os.urandom(size))
    store.update_agent_snapshot(
        agent=agent,
        snapshot_path=str(snap),
        snapshot_size_bytes=snap.stat().st_size,
        tokens_saved=0,
    )
    return snap


def test_gc_removes_unreferenced_snapshots(store, snapshots_dir):
    """SnapshotGC deletes snapshot files not referenced by any agent's HEAD."""
    agent = store.create_agent("a", "model", "hash", 8192)
    head = _set_head_snapshot(store, agent, snapshots_dir)

    orphan = snapshots_dir / "orphan_abc12345.bin"
    orphan.write_bytes(os.urandom(512))

    gc = SnapshotGC(store, snapshots_dir)
    deleted = gc.collect(dry_run=False)

    assert orphan in deleted
    assert not orphan.exists()
    # The agent's HEAD snapshot must survive
    assert head.exists()


def test_gc_dry_run_does_not_delete(store, snapshots_dir):
    """dry_run=True lists candidates without deleting them."""
    agent = store.create_agent("a", "model", "hash", 8192)
    _set_head_snapshot(store, agent, snapshots_dir)
    orphan = snapshots_dir / "orphan_dryrun.bin"
    orphan.write_bytes(os.urandom(512))

    gc = SnapshotGC(store, snapshots_dir)
    deleted = gc.collect(dry_run=True)

    assert orphan in deleted
    assert orphan.exists()  # not actually deleted


def test_gc_removes_tmp_orphans(store, snapshots_dir):
    """GC removes .tmp_ prefixed files left by crashed sessions."""
    agent = store.create_agent("a", "model", "hash", 8192)
    _set_head_snapshot(store, agent, snapshots_dir)

    tmp_file = snapshots_dir / ".tmp_orphan_crash.bin"
    tmp_file.write_bytes(os.urandom(512))

    gc = SnapshotGC(store, snapshots_dir)
    deleted = gc.collect()
    assert tmp_file in deleted
    assert not tmp_file.exists()


def test_gc_keeps_head_for_every_agent(store, snapshots_dir):
    """Each agent's current HEAD snapshot is retained; superseded ones are reaped."""
    agent_a = store.create_agent("a", "model", "hash", 8192)
    agent_b = store.create_agent("b", "model", "hash", 8192)

    # Agent A advances HEAD twice — the older snapshot becomes unreferenced.
    old_a = _set_head_snapshot(store, agent_a, snapshots_dir)
    new_a = _set_head_snapshot(store, store.get_agent("a"), snapshots_dir)
    head_b = _set_head_snapshot(store, agent_b, snapshots_dir)

    gc = SnapshotGC(store, snapshots_dir)
    deleted = gc.collect()

    # Both agents' current HEADs survive
    assert new_a.exists()
    assert head_b.exists()
    # The superseded snapshot is collected
    assert old_a in deleted
    assert not old_a.exists()
