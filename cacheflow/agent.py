"""Agent loop: completion, save, commit."""

import logging
import resource
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
from typing import Optional, Callable

logger = logging.getLogger(__name__)

from cacheflow.config import load_config, CacheFlowConfig
from cacheflow.store import CacheFlowStore, Agent, _hash_context
from cacheflow.engine import LlamaEngine, get_global_engine
from cacheflow.compressor import Compressor
from cacheflow.tokenizer import ModelTokenizer, get_tokenizer
from cacheflow.slot_pool import SlotPool, SlotLease
from cacheflow.gc import SnapshotGC
from cacheflow.templates import ChatTemplate, detect_template


DEFAULT_SYSTEM_PROMPT = """You are an expert software engineer with deep knowledge of the codebase you've been given access to. You help with coding tasks efficiently and precisely. When you complete a task, briefly summarize what you did and what you learned about the codebase."""

# Global slot pool for managing concurrent agent execution
_SLOT_POOL = SlotPool(max_slots=8)

# Serializes concurrent init_db calls to prevent SQLite locking races
_DB_INIT_LOCK = threading.Lock()

def _compute_flops_avoided(
    param_count: Optional[int],
    tokens_skipped: int,
    arch_info: Optional[dict] = None,
) -> Optional[float]:
    """FLOPs avoided by skipping prefill on `tokens_skipped` tokens.

    `param_count` must be the model's exact parameter count, read directly off
    the loaded GGUF via llama.cpp's own `llama_model_n_params` (engine.py) —
    never parsed/guessed from a model name or file size. Given that exact
    count, `2 * param_count` FLOPs/token is the standard, analytically-derived
    cost of every QKVO projection + FFN matmul in a forward pass (one
    multiply-add per parameter per token) — but it's a *floor*, not the whole
    cost: it has no notion of context length, while causal self-attention's
    score (Q·K^T) and weighted-sum (softmax·V) steps are NOT covered by
    param_count at all (attention weights have no parameters) and scale with
    how many prior tokens each token attends to. Summed over a prefill of T
    tokens evaluated together (token i attends to i prior tokens), that extra
    cost is `2 * n_layer * n_embd * T^2` — quadratic in T, derived from
    n_layer attention blocks each doing two T-by-n_embd matmuls per token,
    summed over T. (Residual adds and layernorm are elementwise — O(n_embd)
    each, negligible next to either term above, so deliberately not modeled.)

    This quadratic term is exactly the kind of thing the flat 2*param_count*T
    estimate hides, and it's not negligible for CacheFlow's workload: prefill
    lengths here run into the thousands of tokens, often comparable to or
    larger than the model's n_embd, at which point T^2 stops being dominated
    by the linear term. `arch_info` (engine.py's `get_arch_info()`) supplies
    n_layer/n_embd read straight off the GGUF's own metadata; when it's
    unavailable (e.g. an architecture whose metadata keys don't match the
    standard `{arch}.block_count`/`{arch}.embedding_length` convention), this
    falls back to the param-count-only floor rather than guess dimensions.

    Returns None (rather than fabricating a number) if `tokens_skipped` or
    `param_count` is unavailable.
    """
    if tokens_skipped <= 0 or param_count is None:
        return None
    flops = 2.0 * param_count * tokens_skipped
    if arch_info:
        n_layer = arch_info.get("n_layer")
        n_embd = arch_info.get("n_embd")
        if n_layer and n_embd:
            flops += 2.0 * n_layer * n_embd * (tokens_skipped ** 2)
    return flops


def _cpu_time_ms() -> float:
    """Total CPU time (user + system) consumed by this process so far, in ms.

    Sourced from the OS via `resource.getrusage` — the kernel's own per-process
    CPU accounting (sums all threads under RUSAGE_SELF), not derived or
    estimated. Since the model runs in-process (engine.py), this captures the
    actual CPU work llama.cpp performs during prime/restore/completion.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return (usage.ru_utime + usage.ru_stime) * 1000.0


@dataclass
class SessionResult:
    """Result of a single agent session."""

    agent_name: str
    task: str
    response: str
    tokens_this_session: int
    tokens_saved: int
    snapshot_size_bytes: int
    duration_ms: int
    is_first_session: bool
    tokens_per_sec: float = 0.0
    prime_time_ms: int = 0
    restore_time_ms: int = 0
    time_saved_ms: int = 0
    prime_cpu_ms: float = 0.0
    restore_cpu_ms: float = 0.0
    cpu_time_saved_ms: float = 0.0
    flops_avoided: Optional[float] = None


class AgentSession:
    """Manages a single agent session: load, run, save, commit."""

    def __init__(self, agent_name: str, base_path: Path):
        self.agent_name = agent_name
        self.base_path = Path(base_path)
        self.config: Optional[CacheFlowConfig] = None
        self.store: Optional[CacheFlowStore] = None
        self.server: Optional[LlamaEngine] = None
        self.slot_lease: Optional[SlotLease] = None
        self.slot_id: Optional[int] = None
        self._tokenizer: Optional[ModelTokenizer] = None
        self._template: Optional[ChatTemplate] = None
        self._setup()

    def _setup(self) -> None:
        """Load config and initialize store."""
        self.config = load_config(self.base_path)
        db_path = self.base_path / ".cacheflow" / "agents.db"
        self.store = CacheFlowStore(db_path)
        with _DB_INIT_LOCK:
            self.store.init_db()
        # get_tokenizer() returns a ModelTokenizer handle, but the underlying
        # vocab-only Llama load is itself deferred until the first .count()/
        # .encode() call (see ModelTokenizer below) — so this is cheap, and
        # AgentSession construction no longer blocks on a model load before
        # it's actually needed.
        self._tokenizer = get_tokenizer(self.config.model_path)

    def _acquire_lock(self) -> None:
        """Acquire a KV cache slot for this agent."""
        agent = self.store.get_agent(self.agent_name)
        if not agent:
            agent = self.store.create_agent(
                self.agent_name,
                self.config.model_name,
                self.config.model_hash,
                self.config.ctx_size,
            )
        self.slot_lease = _SLOT_POOL.acquire_slot(agent.id)
        self.slot_id = self.slot_lease.slot_id

    def _release_lock(self) -> None:
        """Release the KV cache slot."""
        if self.slot_lease is not None:
            # Call release_slot directly; __exit__ misuse avoided
            _SLOT_POOL.release_slot(self.slot_lease.slot_id)
            self.slot_lease = None
            self.slot_id = None

    def _collect_source_files(self) -> list[Path]:
        """Return all source files in the project, skipping generated/vendor dirs."""
        SOURCE_EXTS = {
            ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
            ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
            ".kt", ".scala", ".sh", ".bash", ".yaml", ".yml", ".toml",
            ".json", ".md", ".txt", ".sql", ".html", ".css", ".env.example",
        }
        SKIP_DIRS = {".git", ".cacheflow", "__pycache__", "node_modules",
                     ".venv", "venv", ".tox", "dist", "build", ".mypy_cache"}

        files: list[Path] = []
        try:
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=self.base_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for rel in result.stdout.splitlines():
                    p = self.base_path / rel
                    if p.suffix in SOURCE_EXTS and p.is_file():
                        files.append(p)
                return files
        except Exception:
            pass

        # Fallback: honour .gitignore via pathspec if available
        spec = None
        try:
            import pathspec
            gitignore_path = self.base_path / ".gitignore"
            if gitignore_path.exists():
                with open(gitignore_path) as f:
                    spec = pathspec.PathSpec.from_lines("gitwildmatch", f)
        except ImportError:
            pass

        for p in self.base_path.rglob("*"):
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            if not p.is_file() or p.suffix not in SOURCE_EXTS:
                continue
            if spec is not None:
                rel = str(p.relative_to(self.base_path))
                if spec.match_file(rel):
                    continue
            files.append(p)
        return files

    def _count_tokens(self, text: str) -> int:
        """Return the exact token count using the model's tokenizer."""
        return self._tokenizer.count(text)

    def _build_stable_context(self, budget_tokens: int) -> str:
        """Build codebase context that is byte-for-byte identical every session.

        Uses the model's exact tokenizer for all token budget decisions.
        """
        SKIP_SUFFIXES = {".lock", ".sum", ".mod"}
        SKIP_NAMES = {"package-lock.json", "yarn.lock", "poetry.lock"}

        files = self._collect_source_files()
        parts: list[str] = []
        total_tokens = 0
        for f in files:
            if f.suffix in SKIP_SUFFIXES or f.name in SKIP_NAMES:
                continue
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = str(f.relative_to(self.base_path))
            block = f"\n--- {rel} ---\n{content}\n"
            block_tokens = self._count_tokens(block)
            if total_tokens + block_tokens > budget_tokens:
                break
            parts.append(block)
            total_tokens += block_tokens

        if not parts:
            return ""
        return "Codebase:\n" + "".join(parts)

    def _get_template(self) -> ChatTemplate:
        """Detect and cache the active model's native instruction-template family.

        Sniffs the GGUF's own embedded chat_template (via the loaded engine's
        model metadata) when available, otherwise falls back to the model
        name; see cacheflow.templates.detect_template.
        """
        if self._template is None:
            metadata = None
            if self.server is not None:
                metadata = getattr(self.server, "model", None)
                metadata = getattr(metadata, "metadata", None)
            self._template = detect_template(self.config.model_name, metadata)
        return self._template

    def _build_stable_prefix(self, system_prompt: str, knowledge_summary: Optional[str] = None) -> str:
        """Build the stable KV prefix: system prompt + codebase, WITHOUT the task.

        If the agent has a distilled `knowledge_summary` (produced by background
        consolidation), it is folded into the prefix so the learned knowledge
        persists across sessions as part of the cached KV. Including it changes the
        prefix hash, which triggers exactly one re-prime; thereafter it's stable.
        """
        budget_tokens = int(self.config.ctx_size * 0.6)
        context = self._build_stable_context(budget_tokens=budget_tokens)
        template = self._get_template()

        summary = (knowledge_summary or "").strip()
        summary_block = (
            f"Consolidated knowledge from previous sessions:\n{summary}\n" if summary else ""
        )
        user_body = "".join(p for p in (context, summary_block) if p)

        if template.supports_system:
            blocks = template.wrap_system(system_prompt)
            if user_body:
                blocks += template.wrap_user(user_body)
            return blocks
        # Family has no dedicated system role (e.g. Mistral, Gemma) -- fold the
        # system prompt into the first user turn instead of dropping it.
        combined = f"{system_prompt}\n\n{user_body}" if user_body else system_prompt
        return template.wrap_user(combined)

    def _build_task_suffix(self, task: str) -> str:
        """Build the task-specific suffix that appends to the stable prefix."""
        template = self._get_template()
        return f"{template.wrap_user(f'Task: {task}')}{template.assistant_open}{self._think_prefill()}"

    def run(
        self,
        task: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tokens: int = 1024,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> SessionResult:
        """Run a single agent session.

        If `on_token` is given, the final completion streams each generated text
        piece to the callback as it is produced (the codebase is already cached, so
        only the task suffix is evaluated before tokens start flowing).
        """
        start_time = time.time()

        try:
            # Step a: Load or create agent
            agent = self.store.get_agent(self.agent_name)
            if not agent:
                agent = self.store.create_agent(
                    self.agent_name,
                    self.config.model_name,
                    self.config.model_hash,
                    self.config.ctx_size,
                )

            # Step b: Acquire KV cache slot
            self._acquire_lock()

            # Step c: Get the in-process model engine (loads model once per process).
            # In-process — not an HTTP subprocess — so token decode runs at full GPU
            # speed (the HTTP path throttled decode ~10x on macOS).
            self.server = get_global_engine(
                model_path=self.config.model_path,
                slot_save_path=str(self.config.slot_save_path),
                ctx_size=self.config.ctx_size,
                n_gpu_layers=self.config.n_gpu_layers,
            )

            # Step d: Build stable prefix and detect codebase changes
            restore_time_ms = 0
            prime_time_ms = 0
            restore_cpu_ms = 0.0
            prime_cpu_ms = 0.0
            is_first_session = agent.current_snapshot_path is None

            stable_prefix = self._build_stable_prefix(system_prompt, agent.knowledge_summary)
            task_suffix = self._build_task_suffix(task)
            full_prompt = stable_prefix + task_suffix

            # Compare hashes, not full text — avoids loading multi-MB strings from DB
            current_hash = _hash_context(stable_prefix)
            context_changed = (agent.stable_context_hash != current_hash)

            # The agent's stored snapshot was written by whatever model was active
            # when it was last primed. Snapshots are raw KV-cache bytes tied to that
            # model's tokenizer/vocab/hidden dims — restoring them into a different
            # model (e.g. after `cf model use` swaps the active model) would corrupt
            # state or crash. Treat a model identity mismatch exactly like a
            # stable_context_hash mismatch: force a re-prime instead of restoring.
            model_changed = (agent.model_name != self.config.model_name)

            if is_first_session or context_changed or model_changed:
                prime_start = time.time()
                prime_cpu_start = _cpu_time_ms()
                self.server.prime_slot(stable_prefix, slot_id=self.slot_id)
                prime_time_ms = int((time.time() - prime_start) * 1000)
                prime_cpu_ms = _cpu_time_ms() - prime_cpu_start
            else:
                if agent.current_snapshot_path:
                    restore_start = time.time()
                    restore_cpu_start = _cpu_time_ms()
                    snapshot_filename = Path(agent.current_snapshot_path).name
                    self.server.restore_slot(snapshot_filename, slot_id=self.slot_id)
                    restore_time_ms = int((time.time() - restore_start) * 1000)
                    restore_cpu_ms = _cpu_time_ms() - restore_cpu_start

            # Step e: Save snapshot (stable prefix only, before task evaluation).
            # Only save when we actually re-primed: on the restore path the HEAD
            # snapshot already exists on disk and is byte-identical to what save_slot
            # would write, so saving again is pure redundant I/O (~503 MB write).
            primed = is_first_session or context_changed or model_changed
            save_result = None
            if primed:
                save_result = self.server.save_slot(slot_id=self.slot_id)

            # Step f: Run completion. Always send the full prompt so
            # llama-cpp-python's prefix matching can find the stable prefix in
            # the KV cache (whether from prime or restore) and only evaluate
            # the task suffix tokens.
            response_data = self.server.completion(
                prompt=full_prompt,
                slot_id=self.slot_id,
                max_tokens=max_tokens,
                on_token=on_token,
            )

            response_text = response_data.get("content", "")
            tokens_in = response_data.get("tokens_evaluated", 0)
            tokens_out = response_data.get("tokens_predicted", 0)
            total_prompt_tokens = response_data.get("usage", {}).get("prompt_tokens", 0)

            if tokens_out == 0:
                raise RuntimeError("Server returned zero tokens - likely a server error or no response")

            tokens_this_session = tokens_in + tokens_out

            # Persist stable_context_hash whenever it changes (64-byte hash, not multi-MB text)
            if context_changed or is_first_session or model_changed:
                self.store.update_agent_stable_context(agent, stable_prefix)
                agent.stable_context_hash = current_hash

            if model_changed:
                # The agent's KV cache (and any baseline computed against the old
                # model) is now stale — the re-prime above just rebuilt it under
                # the new model, so re-point the agent's stored identity at it.
                # Without this, every future session would keep detecting this
                # same "mismatch" and re-priming forever.
                self.store.update_agent_model(agent, self.config.model_name, self.config.model_hash)
                agent.model_name = self.config.model_name
                agent.model_hash = self.config.model_hash

            if is_first_session or model_changed or agent.baseline_tokens_evaluated is None:
                tokens_saved = 0
                baseline = total_prompt_tokens if total_prompt_tokens > 0 else tokens_in
                self.store.update_agent_baseline(agent, baseline)
            else:
                tokens_saved = max(0, agent.baseline_tokens_evaluated - tokens_in)

            if primed:
                # Re-measure the cold-prime cost on every prime (codebase growth
                # changes it over time), so time_saved_ms/cpu_time_saved_ms always
                # compare against the current baseline rather than a stale
                # first-session number.
                self.store.update_agent_time_baseline(agent, prime_time_ms)
                agent.baseline_prime_time_ms = prime_time_ms
                self.store.update_agent_cpu_time_baseline(agent, int(prime_cpu_ms))
                agent.baseline_prime_cpu_ms = int(prime_cpu_ms)
                time_saved_ms = 0
                cpu_time_saved_ms = 0.0
            else:
                baseline_prime_ms = agent.baseline_prime_time_ms or 0
                time_saved_ms = max(0, baseline_prime_ms - restore_time_ms)
                baseline_prime_cpu_ms = agent.baseline_prime_cpu_ms or 0
                cpu_time_saved_ms = max(0.0, baseline_prime_cpu_ms - restore_cpu_ms)

            # Exact parameter count and architecture dims read off the loaded model.
            param_count = self.server.get_param_count()
            arch_info = self.server.get_arch_info()
            flops_avoided = _compute_flops_avoided(param_count, tokens_saved, arch_info)

            if primed:
                # Validate save result and promote the new snapshot to HEAD
                saved_filename = save_result.get("filename", "")
                if not saved_filename:
                    raise RuntimeError(f"Server failed to save snapshot: {save_result}")

                saved_path = self.config.slot_save_path / saved_filename
                if not saved_path.exists():
                    raise RuntimeError(f"Snapshot file not created by server: {saved_path}")

                snapshot_size = saved_path.stat().st_size
                if snapshot_size == 0:
                    saved_path.unlink()
                    raise RuntimeError("Server created empty snapshot file")

                final_snapshot_name = f"{agent.name}_{uuid4()}.bin"
                final_snapshot_path = self.config.slot_save_path / final_snapshot_name
                saved_path.rename(final_snapshot_path)
            else:
                # Restore path: reuse the existing HEAD snapshot (no new file written)
                final_snapshot_path = Path(agent.current_snapshot_path)

            self.store.update_agent_snapshot(
                agent=agent,
                snapshot_path=str(final_snapshot_path),
                snapshot_size_bytes=final_snapshot_path.stat().st_size,
                tokens_saved=tokens_saved,
                time_saved_ms=time_saved_ms,
                cpu_time_saved_ms=int(cpu_time_saved_ms),
            )

            total_duration_ms = int((time.time() - start_time) * 1000)
            snapshot_size_bytes = final_snapshot_path.stat().st_size if final_snapshot_path.exists() else 0

            # Accumulate this session's token volume, then let the background
            # compressor decide whether to consolidate (≥70% of context).
            self.store.add_accumulated_tokens(agent, tokens_this_session)
            compressor = Compressor(self.store, self.config)
            compressor.maybe_compact_async(agent)

            SnapshotGC(self.store, self.config.slot_save_path).collect()

            return SessionResult(
                agent_name=self.agent_name,
                task=task,
                response=response_text,
                tokens_this_session=tokens_this_session,
                tokens_saved=tokens_saved,
                snapshot_size_bytes=snapshot_size_bytes,
                duration_ms=total_duration_ms,
                is_first_session=is_first_session,
                tokens_per_sec=response_data.get("tokens_per_sec", 0.0),
                prime_time_ms=prime_time_ms,
                restore_time_ms=restore_time_ms,
                time_saved_ms=time_saved_ms,
                prime_cpu_ms=prime_cpu_ms,
                restore_cpu_ms=restore_cpu_ms,
                cpu_time_saved_ms=cpu_time_saved_ms,
                flops_avoided=flops_avoided,
            )

        finally:
            self._release_lock()

    def _restore_or_prime(self, agent: Agent, system_prompt: str) -> str:
        """Ensure the agent's codebase KV is loaded in the active slot.

        Restores the HEAD snapshot if it still matches; otherwise primes from
        scratch and promotes the new snapshot to HEAD. Returns the stable prefix.
        """
        stable_prefix = self._build_stable_prefix(system_prompt, agent.knowledge_summary)
        current_hash = _hash_context(stable_prefix)
        is_first = agent.current_snapshot_path is None
        context_changed = agent.stable_context_hash != current_hash
        # See run(): a stored HEAD snapshot was written by whichever model was
        # active at the time. If the active model has since changed (e.g. via
        # `cf model use`), restoring that snapshot into the new model would
        # corrupt state — force a re-prime instead, exactly like a context change.
        model_changed = agent.model_name != self.config.model_name

        if is_first or context_changed or model_changed:
            prime_start = time.time()
            prime_cpu_start = _cpu_time_ms()
            self.server.prime_slot(stable_prefix, slot_id=self.slot_id)
            prime_time_ms = int((time.time() - prime_start) * 1000)
            prime_cpu_ms = _cpu_time_ms() - prime_cpu_start
            self.store.update_agent_time_baseline(agent, prime_time_ms)
            agent.baseline_prime_time_ms = prime_time_ms
            self.store.update_agent_cpu_time_baseline(agent, int(prime_cpu_ms))
            agent.baseline_prime_cpu_ms = int(prime_cpu_ms)
            save_result = self.server.save_slot(slot_id=self.slot_id)
            saved_filename = save_result.get("filename", "")
            saved_path = self.config.slot_save_path / saved_filename
            if saved_filename and saved_path.exists() and saved_path.stat().st_size > 0:
                final_name = f"{agent.name}_{uuid4()}.bin"
                final_path = self.config.slot_save_path / final_name
                saved_path.rename(final_path)
                self.store.update_agent_snapshot(
                    agent=agent,
                    snapshot_path=str(final_path),
                    snapshot_size_bytes=final_path.stat().st_size,
                    tokens_saved=0,
                    time_saved_ms=0,
                )
                self.store.update_agent_stable_context(agent, stable_prefix)
                agent.stable_context_hash = current_hash
            if model_changed:
                self.store.update_agent_model(agent, self.config.model_name, self.config.model_hash)
                agent.model_name = self.config.model_name
                agent.model_hash = self.config.model_hash
        else:
            self.server.restore_slot(
                Path(agent.current_snapshot_path).name, slot_id=self.slot_id
            )
        return stable_prefix

    def _think_prefill(self) -> str:
        """Force-close Qwen3's thinking block immediately on the assistant turn.

        Qwen3 (unlike 2.5) defaults to emitting a `<think>...</think>` reasoning
        block before any real content. In this loop each step has a small,
        fixed `max_tokens_per_step` budget for the whole THOUGHT/ACTION/ARGS
        turn — if thinking runs unchecked it can consume that entire budget
        and the turn gets cut off before ACTION/ARGS ever appears, so
        parse_action raises on every step. Pre-filling an empty think block is
        the documented hard way to suppress it (more reliable than the
        soft "/no_think" text hint).
        """
        if "qwen3" in self.config.model_name.lower():
            return "<think>\n\n</think>\n\n"
        return ""

    def _build_consolidation_suffix(self) -> str:
        """The prompt suffix that asks the model to distill what it has learned."""
        question = (
            "Based on this codebase and your prior analysis, write a dense, factual "
            "summary (max ~500 tokens) of the most important things to know: key "
            "modules and their responsibilities, important functions/classes, "
            "architectural patterns, and known risks. Be specific and terse. "
            "Do not include preamble."
        )
        template = self._get_template()
        return f"{template.wrap_user(question)}{template.assistant_open}{self._think_prefill()}"

    def consolidate(self, summary_max_tokens: int = 500) -> Optional[str]:
        """Distill the agent's learned knowledge into a persistent summary.

        Runs against the agent's hot codebase KV (restored from HEAD, or primed if
        needed), asks the model for a dense summary, and stores it. The summary is
        folded into the stable prefix on the next session, so learned knowledge
        survives even though each session otherwise restores only the codebase KV.
        Resets the token accumulator. Best-effort: never raises into the caller.

        Returns the summary text, or None if consolidation was skipped/failed.
        """
        try:
            agent = self.store.get_agent(self.agent_name)
            if agent is None or agent.current_snapshot_path is None:
                # Nothing primed yet — nothing to consolidate.
                return None

            self._acquire_lock()
            try:
                self.server = get_global_engine(
                    model_path=self.config.model_path,
                    slot_save_path=str(self.config.slot_save_path),
                    ctx_size=self.config.ctx_size,
                    n_gpu_layers=self.config.n_gpu_layers,
                )

                stable_prefix = self._build_stable_prefix(
                    DEFAULT_SYSTEM_PROMPT, agent.knowledge_summary
                )
                current_hash = _hash_context(stable_prefix)
                # See run()/_restore_or_prime(): never restore a snapshot written
                # by a different model than the one currently active.
                model_changed = agent.model_name != self.config.model_name

                # Restore the codebase KV if it still matches; otherwise prime fresh.
                if (
                    agent.stable_context_hash == current_hash
                    and agent.current_snapshot_path
                    and not model_changed
                ):
                    snapshot_filename = Path(agent.current_snapshot_path).name
                    self.server.restore_slot(snapshot_filename, slot_id=self.slot_id)
                else:
                    self.server.prime_slot(stable_prefix, slot_id=self.slot_id)
                    if model_changed:
                        self.store.update_agent_model(
                            agent, self.config.model_name, self.config.model_hash
                        )
                        agent.model_name = self.config.model_name
                        agent.model_hash = self.config.model_hash

                response = self.server.completion(
                    prompt=stable_prefix + self._build_consolidation_suffix(),
                    slot_id=self.slot_id,
                    max_tokens=summary_max_tokens,
                )
                summary = (response.get("content") or "").strip()
                if not summary:
                    return None

                self.store.update_agent_knowledge_summary(agent, summary)
                logger.info(
                    "consolidated agent '%s': %d-char knowledge summary, accumulator reset",
                    self.agent_name, len(summary),
                )
                return summary
            finally:
                self._release_lock()
        except Exception:
            logger.exception("consolidation failed for agent '%s'", self.agent_name)
            return None


def fork_agent(
    parent_name: str, child_name: str, base_path: Path, scope: str = ""
) -> Agent:
    """Fork a new agent from an existing agent's snapshot."""
    base_path = Path(base_path)
    db_path = base_path / ".cacheflow" / "agents.db"
    store = CacheFlowStore(db_path)

    parent_agent = store.get_agent(parent_name)
    if not parent_agent:
        raise ValueError(f"Parent agent '{parent_name}' not found")

    if not parent_agent.current_snapshot_path:
        raise ValueError(f"Parent agent '{parent_name}' has no snapshot to fork from")

    parent_snapshot_path = Path(parent_agent.current_snapshot_path)
    if not parent_snapshot_path.is_absolute():
        # Stored paths are relative to base_path already (e.g. ".cacheflow/snapshots/x.bin"),
        # matching how the restore path in run() uses current_snapshot_path directly.
        parent_snapshot_path = base_path / parent_snapshot_path

    if not parent_snapshot_path.exists():
        raise ValueError(
            f"Parent snapshot not found at {parent_snapshot_path}. "
            f"Cannot fork without a valid snapshot to copy."
        )

    child_agent = store.create_agent(
        name=child_name,
        model_name=parent_agent.model_name,
        model_hash=parent_agent.model_hash,
        ctx_size=parent_agent.ctx_size,
    )

    # Update child to have parent_agent_id
    session = store._get_session()
    try:
        child_agent.parent_agent_id = parent_agent.id
        session.merge(child_agent)
        session.commit()
    finally:
        session.close()

    snapshots_dir = base_path / ".cacheflow" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    fork_snapshot_name = f"{child_name}_{uuid4()}.bin"
    fork_snapshot_path = snapshots_dir / fork_snapshot_name
    shutil.copy2(parent_snapshot_path, fork_snapshot_path)

    store.update_agent_snapshot(
        agent=child_agent,
        snapshot_path=str(fork_snapshot_path),
        snapshot_size_bytes=fork_snapshot_path.stat().st_size,
        tokens_saved=parent_agent.last_tokens_saved,
        time_saved_ms=parent_agent.last_time_saved_ms,
        cpu_time_saved_ms=parent_agent.last_cpu_time_saved_ms,
    )
    if parent_agent.baseline_prime_time_ms is not None:
        store.update_agent_time_baseline(child_agent, parent_agent.baseline_prime_time_ms)
    if parent_agent.baseline_prime_cpu_ms is not None:
        store.update_agent_cpu_time_baseline(child_agent, parent_agent.baseline_prime_cpu_ms)

    return child_agent
