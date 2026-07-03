"""Cloud (frontier-model) workloads for CacheFlow — reasoning-token reuse.

The local W1..W8 experiments (run.py) measure KV-cache prefill time saved on a
self-hosted model. This module measures the *cloud* analogue: extended-thinking
(reasoning) tokens NOT regenerated because a prior agent's reasoning was reused
from the ThinkingStore / KnowledgeStore instead of re-derived from scratch.

Path C (chosen): the Anthropic SDK with extended thinking enabled. This is the
only capture route that hands us BOTH the raw reasoning text (`ThinkingBlock.
thinking`) and the *exact, isolated* reasoning-token count
(`usage.output_tokens_details.thinking_tokens`) in a single response — no
transcript scraping, no PTY, and no local tokenizer approximation. It costs real
API tokens for the cold calls we actually make; every warm reuse avoids one.

The two SQLite pools do the accounting, exactly as on the feature branch:
  - ThinkingStore: submit(thinking_text, token_count=thinking_tokens) on a cold
    reasoning turn; query() before a warm one. An exact-hash hit logs the reused
    block's real token_count into thinking_reuse_log — that IS the reasoning
    tokens saved, never estimated from text length.
  - KnowledgeStore: region summaries for the consolidation workloads (W7).

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    python experiments/cloud_runner.py --workload W2
    python experiments/cloud_runner.py --workload all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Reasoning is offloaded to the Anthropic API; the only thing that runs locally
# is ThinkingStore's semantic-reuse embedding model, now all-MiniLM-L6-v2
# (~90 MB) -- small enough to be safe. (It replaced a 7B, ~13 GB model that
# swap-stormed and kernel-panicked memory-constrained Macs.) Semantic reuse is
# left ENABLED so warm hits on *similar*, not just identical, tasks are
# exercised. Set CACHEFLOW_DISABLE_EMBEDDINGS=1 in the env to force exact-hash
# only.

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from cacheflow.thinking_store import ThinkingStore
from cacheflow.knowledge_store import KnowledgeStore

CORPORA_DIR = REPO_ROOT / "experiments" / "corpora"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

# Same real-codebase ladder as the local experiments, so cloud and local rollups
# line up workload-for-workload.
CORPUS: dict[str, str] = {
    "W1": "itsdangerous",   # ~1.5K LOC — single reasoning task (baseline)
    "W2": "click",          # ~8K LOC — cold + one warm reuse
    "W3": "requests",       # ~12K LOC — repeated warm reuse (compounding)
    "W4": "httpx",          # ~18K LOC — concurrent agents share one pool
    "W5": "flask",          # ~15K LOC — multi-step reasoning, each step cached
    "W6": "pytest",         # ~40K LOC — fork: agent B reuses agent A's reasoning
    "W7": "sqlalchemy",     # ~80K LOC — consolidation into the KnowledgeStore
    "W8": "django",         # ~280K LOC — largest prefix, hardest task
}

DEFAULT_MODEL = "claude-opus-4-8"
# Medium-confidence ('validate'-band) reuse candidates are confirmed by a cheap,
# fast model rather than reused blind or re-derived on Opus. Haiku 4.5's yes/no
# costs a tiny fraction of a cold Opus reasoning call.
VALIDATOR_MODEL = "claude-haiku-4-5"
# claude-opus-4-8 (and other newer models) reject the legacy
# thinking={"type": "enabled", "budget_tokens": N} shape ("thinking.type.enabled
# is not supported for this model") — they use adaptive thinking, controlled by
# output_config.effort instead of an explicit token budget. --budget still
# exists as a CLI knob but now selects an effort tier, kept modest to bound cost.
DEFAULT_THINKING_EFFORT = "low"
_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
ANSWER_BUDGET = 1024
MAX_TOKENS_TOTAL = 4096  # ceiling on thinking + answer tokens for one cold call
# Codebase context is bounded (and prompt-cached) so cost stays on the reasoning,
# not on re-ingesting the tree every call. ~40K chars ≈ ~10K tokens.
CONTEXT_CHAR_BUDGET = 40_000

SYSTEM_PROMPT = (
    "You are a senior software engineer working inside this codebase as a coding "
    "agent. The user gives you real tasks -- debug something, add a feature, "
    "refactor, plan a migration, write code. Reason through the problem, then act: "
    "give the concrete change, diff, command, or step-by-step plan you'd apply, "
    "citing the specific modules or functions involved. Be direct and buildable, "
    "not a high-level essay."
)


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

@dataclass
class CloudMetric:
    workload: str
    corpus: str
    step: str
    agent: str
    is_first_session: bool          # True on a cold (real API) reasoning call
    duration_ms: int
    # Reasoning-token accounting — exact, from the API's own usage object.
    reasoning_tokens: Optional[int] = None      # cold: this turn's thinking_tokens
    reasoning_tokens_saved: int = 0             # warm: from thinking_reuse_log
    # What a reuse ACTUALLY avoided: the origin derivation's full cost (codebase
    # input context + full output), not just the isolated thinking sliver. This
    # is the number that scales with codebase size and reflects real savings.
    total_tokens_avoided: int = 0
    thinking_store_hit: bool = False
    had_thinking_block: bool = False            # did the model actually reason?
    confidence: Optional[float] = None          # store's reuse confidence (1.0 exact)
    # Knowledge pool (W7 consolidation).
    knowledge_tokens_saved: int = 0
    knowledge_store_hit: bool = False
    # Real dollars spent on the cold calls we did make (input+output priced).
    # input_tokens is the UNCACHED input only; the cached codebase prefix is
    # counted separately (billed at 1.25x on create, 0.1x on read) so total
    # input processed = input + cache_creation + cache_read.
    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class CloudWorkloadRunner:
    def __init__(
        self,
        repo_root: Path = REPO_ROOT,
        model: str = DEFAULT_MODEL,
        thinking_effort: str = DEFAULT_THINKING_EFFORT,
    ):
        self.repo_root = Path(repo_root)
        self.model = model
        if thinking_effort not in _EFFORT_LEVELS:
            raise ValueError(f"thinking_effort must be one of {_EFFORT_LEVELS}")
        self.thinking_effort = thinking_effort
        self._client = None
        # Dedicated, cloud-only store files so a run never clobbers the local
        # KV-engine's databases. Reset per-workload (see _reset_stores) so a
        # "cold" reasoning call is genuinely cold.
        self._thinking_db = str(self.repo_root / ".cacheflow" / "cloud_thinking.db")
        self._knowledge_db = str(self.repo_root / ".cacheflow" / "cloud_knowledge.db")
        self.thinking_store = ThinkingStore(self._thinking_db)
        self.knowledge_store = KnowledgeStore(self._knowledge_db)

    # -- Anthropic client (lazy) --------------------------------------------

    @property
    def client(self):
        if self._client is None:
            import anthropic  # imported lazily so --help etc. work without the dep
            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        return self._client

    def _reset_stores(self) -> None:
        """Delete and recreate both pools so cold calls in a workload are cold."""
        for p in (self._thinking_db, self._knowledge_db):
            fp = Path(p)
            if fp.exists():
                fp.unlink()
        self.thinking_store = ThinkingStore(self._thinking_db)
        self.knowledge_store = KnowledgeStore(self._knowledge_db)

    # -- Corpus / hashing ----------------------------------------------------

    def _corpus_path(self, workload: str) -> Path:
        p = CORPORA_DIR / CORPUS[workload]
        if not p.is_dir():
            raise RuntimeError(f"Corpus for {workload} not found at {p}")
        return p

    def _codebase_hash(self, corpus_path: Path) -> str:
        """Content hash of the corpus (git HEAD tree if available, else file list).

        Mirrors the stores' staleness convention: a hash computed at write time,
        compared at read time, so a changed codebase silently invalidates reuse.
        """
        try:
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=corpus_path, capture_output=True, text=True, timeout=10,
            )
            if head.returncode == 0 and head.stdout.strip():
                return head.stdout.strip()
        except Exception:
            pass
        # Fallback: hash the sorted tracked-file list.
        try:
            files = subprocess.run(
                ["git", "ls-files"],
                cwd=corpus_path, capture_output=True, text=True, timeout=30,
            ).stdout
        except Exception:
            files = ""
        return hashlib.sha256(files.encode()).hexdigest()

    def _load_context(self, corpus_path: Path) -> str:
        """Bounded codebase context: README + a slice of tracked source files.

        Capped at CONTEXT_CHAR_BUDGET and prompt-cached on the API side, so the
        experiment's cost tracks the reasoning we're measuring, not repeated
        re-ingestion of the whole tree.
        """
        try:
            listed = subprocess.run(
                ["git", "ls-files"],
                cwd=corpus_path, capture_output=True, text=True, timeout=30,
            ).stdout.splitlines()
        except Exception:
            listed = []

        # Prefer readmes and Python sources; skip tests/vendored noise.
        def _rank(f: str) -> tuple:
            low = f.lower()
            is_readme = "readme" in low
            is_py = low.endswith(".py")
            is_test = "test" in low or low.startswith("tests/")
            return (0 if is_readme else 1, 0 if is_py else 1, 1 if is_test else 0, len(f))

        listed.sort(key=_rank)

        parts: list[str] = []
        total = 0
        for rel in listed:
            fp = corpus_path / rel
            if not fp.is_file():
                continue
            try:
                text = fp.read_text(errors="ignore")
            except Exception:
                continue
            chunk = f"\n### FILE: {rel}\n{text}\n"
            if total + len(chunk) > CONTEXT_CHAR_BUDGET:
                # take a final partial chunk to fill the budget, then stop
                remaining = CONTEXT_CHAR_BUDGET - total
                if remaining > 500:
                    parts.append(chunk[:remaining])
                break
            parts.append(chunk)
            total += len(chunk)
        return "".join(parts)

    # -- The one API primitive ----------------------------------------------

    def _call_claude(self, context: str, task: str) -> dict:
        """One extended-thinking call. Returns exact thinking text + token counts.

        `thinking_tokens` and `output_tokens` come straight from the API's usage
        object — never a local tokenizer, never len(text).
        """
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS_TOTAL,
            thinking={"type": "adaptive"},
            output_config={"effort": self.thinking_effort},
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT + "\n\n# CODEBASE CONTEXT\n" + context,
                "cache_control": {"type": "ephemeral"},  # cache the big prefix
            }],
            messages=[{"role": "user", "content": task}],
        )

        thinking_parts, answer_parts = [], []
        redacted = False
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "thinking":
                thinking_parts.append(getattr(block, "thinking", "") or "")
            elif btype == "redacted_thinking":
                redacted = True  # safety-redacted; text unavailable, count still real
            elif btype == "text":
                answer_parts.append(getattr(block, "text", "") or "")

        usage = resp.usage
        details = getattr(usage, "output_tokens_details", None)
        thinking_tokens = getattr(details, "thinking_tokens", None) if details else None

        return {
            "thinking_text": "".join(thinking_parts),
            "answer_text": "".join(answer_parts),
            "thinking_tokens": thinking_tokens,   # exact isolated reasoning tokens
            # NB: with prompt caching, usage.input_tokens is ONLY the uncached
            # portion -- the big cached codebase prefix lands in
            # cache_creation_input_tokens (first call) or cache_read_input_tokens
            # (subsequent). Capture all three so total input processed isn't
            # undercounted by ~thousands of tokens per call.
            "input_tokens": getattr(usage, "input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
            "output_tokens": getattr(usage, "output_tokens", 0) or 0,
            "had_thinking_block": bool(thinking_parts) or redacted,
            "redacted": redacted,
        }

    # -- Cold / warm primitives ---------------------------------------------

    def _cold(self, workload: str, corpus_path: Path, task: str,
              role: str = "analyzer", step: str = "cold") -> CloudMetric:
        """Real reasoning call; store the thinking text + exact thinking_tokens."""
        context = self._load_context(corpus_path)
        t0 = time.time()
        r = self._call_claude(context, task)
        dur = int((time.time() - t0) * 1000)

        phash = self.thinking_store._hash_problem(task)
        chash = self._codebase_hash(corpus_path)
        # Store the reasoning text keyed by problem+codebase, with the EXACT
        # isolated reasoning-token count (None if the model didn't break it out,
        # e.g. fully redacted — stored as NULL rather than guessed).
        full_cost = (r["input_tokens"] + r["cache_creation_input_tokens"]
                     + r["cache_read_input_tokens"] + r["output_tokens"])
        self.thinking_store.submit(
            thinking_block=r["thinking_text"] or r["answer_text"],
            problem_hash=phash,
            codebase_hash=chash,
            token_count=r["thinking_tokens"],
            full_cost_tokens=full_cost,
            problem_type="codebase_analysis",
            task_description=task,
            role=role,
            source_agent=f"cloud-{workload}",
        )
        return CloudMetric(
            workload=workload, corpus=corpus_path.name, step=step,
            agent=self.model, is_first_session=True, duration_ms=dur,
            reasoning_tokens=r["thinking_tokens"],
            had_thinking_block=r["had_thinking_block"],
            input_tokens=r["input_tokens"],
            cache_creation_input_tokens=r["cache_creation_input_tokens"],
            cache_read_input_tokens=r["cache_read_input_tokens"],
            output_tokens=r["output_tokens"],
            extra={"redacted": r["redacted"], "answer_chars": len(r["answer_text"])},
        )

    def _validate_reuse(self, task: str, cached_reasoning: str) -> dict:
        """Ask Haiku 4.5 whether cached reasoning actually answers the new task.

        The embedding gate can only say two requests *look* similar; on the
        medium-confidence 'validate' band that isn't enough to trust blindly.
        Routing the yes/no confirmation to a cheap, fast model (Haiku 4.5)
        instead of re-running the full Opus reasoning is the whole point: a
        ~1-token decision costs a tiny fraction of a cold call, so a confirmed
        reuse still avoids almost all the reasoning tokens. Returns the verdict
        plus Haiku's exact token usage so the validation cost is accounted, not
        hidden.
        """
        prompt = (
            "A new engineering task has arrived, and we have cached reasoning "
            "from an earlier, similar task. Decide if the cached reasoning "
            "substantively answers the NEW task well enough to reuse as-is.\n\n"
            f"# NEW TASK\n{task}\n\n"
            f"# CACHED REASONING\n{cached_reasoning[:6000]}\n\n"
            "Answer with exactly one word: YES (safe to reuse) or NO (re-derive)."
        )
        resp = self.client.messages.create(
            model=VALIDATOR_MODEL,
            max_tokens=5,
            system="You are the precision gate for reasoning reuse. The cheap "
                   "embedding filter that got us here only measures surface "
                   "similarity and can't be trusted for the real decision — that "
                   "is your job. Judge on substance: if the cached reasoning "
                   "genuinely answers the new task (even when worded very "
                   "differently), say YES; only say NO when it addresses a "
                   "materially different question.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(getattr(b, "text", "") for b in resp.content
                       if getattr(b, "type", None) == "text").strip().upper()
        return {
            "confirmed": text.startswith("Y"),
            "verdict": text[:10],
            "input_tokens": getattr(resp.usage, "input_tokens", 0) or 0,
            "output_tokens": getattr(resp.usage, "output_tokens", 0) or 0,
        }

    def _warm(self, workload: str, corpus_path: Path, task: str,
              role: str = "analyzer", step: str = "warm") -> CloudMetric:
        """Query the pool; on a high-confidence hit reuse directly (no API call),
        on a medium 'validate' hit confirm with Haiku before reusing, else cold."""
        block, confidence, action = self.thinking_store.query(task, role=role)

        if action == "use_directly" and block is not None:
            saved = self.thinking_store.last_tokens_saved or 0
            avoided = self.thinking_store.last_full_cost_saved or 0
            return CloudMetric(
                workload=workload, corpus=corpus_path.name, step=step,
                agent=self.model, is_first_session=False, duration_ms=0,
                reasoning_tokens_saved=saved, total_tokens_avoided=avoided,
                thinking_store_hit=True,
                confidence=confidence, extra={"action": "use_directly"},
            )

        if action == "validate" and block is not None:
            t0 = time.time()
            v = self._validate_reuse(task, block)
            dur = int((time.time() - t0) * 1000)
            if v["confirmed"]:
                # Confirmed: now (and only now) log the reuse in the store.
                saved = self.thinking_store.commit_validated_reuse() or 0
                avoided = self.thinking_store.last_full_cost_saved or 0
                return CloudMetric(
                    workload=workload, corpus=corpus_path.name, step=step,
                    agent=self.model, is_first_session=False, duration_ms=dur,
                    reasoning_tokens_saved=saved, total_tokens_avoided=avoided,
                    thinking_store_hit=True,
                    confidence=confidence,
                    # Haiku's confirmation is the only cost paid on this reuse.
                    input_tokens=v["input_tokens"], output_tokens=v["output_tokens"],
                    extra={"action": "validate", "validated": True,
                           "validator": VALIDATOR_MODEL, "verdict": v["verdict"]},
                )
            # Rejected: re-derive cold. Fold the (small) validation cost in too.
            m = self._cold(workload, corpus_path, task, role=role,
                           step=f"{step}_validate_rejected_cold")
            m.input_tokens += v["input_tokens"]
            m.output_tokens += v["output_tokens"]
            m.extra.update({"validated": False, "validator": VALIDATOR_MODEL,
                            "verdict": v["verdict"], "validator_ms": dur})
            return m

        m = self._cold(workload, corpus_path, task, role=role, step=f"{step}_miss_cold")
        return m

    # -- Workloads (mirror the local ladder; harder tasks) ------------------

    # Workloads model how a human actually drives a coding agent: imperative,
    # action-oriented tasks (debug / add / refactor / plan / implement), and
    # reuse that happens when a *later* teammate or session asks for the same
    # underlying work in DIFFERENT words -- never a byte-identical repeat. Warm
    # turns pass a paraphrase to _warm(); the semantic gate decides whether the
    # cached reasoning is close enough to reuse (use_directly / validate) or too
    # different to trust (re_think -> a real cold call). Some turns are genuine
    # new work and are expected to miss -- an honest, sub-100% hit rate.

    def w1(self, corpus_path):
        """W1 — one real task; baseline reasoning cost (nothing to reuse yet)."""
        return [self._cold("W1", corpus_path,
            "I'm about to depend on this library to sign our session cookies. "
            "Walk me through how signing and verification actually work here, and "
            "where I could footgun myself around key rotation.")]

    def w2(self, corpus_path):
        """W2 — a task, then a teammate asks for the same thing in other words."""
        cold = ("Add a global --verbose flag to our CLI that turns on debug "
                "logging across every subcommand. How do I wire it in cleanly "
                "with this framework?")
        warm = ("I need every subcommand to share one debug/logging switch — "
                "what's the clean way to add that here?")
        return [self._cold("W2", corpus_path, cold, step="cold"),
                self._warm("W2", corpus_path, warm, step="warm_rephrased")]

    def w3(self, corpus_path):
        """W3 — one diagnosis, re-asked by several people in different words
        (compounding reuse), with an unrelated ask that correctly stays cold."""
        cold = ("Our client built on this library hangs forever when the server "
                "stops responding mid-body. Diagnose the timeout handling and "
                "tell me how to make it fail fast.")
        rephrasings = [
            "Why does a dead connection make our calls block indefinitely, and "
            "how do we bound that?",
            "Requests sometimes just never return on flaky servers — where do I "
            "set timeouts so it can't hang?",
            "Make the client give up instead of blocking when the peer goes "
            "silent on us.",
        ]
        out = [self._cold("W3", corpus_path, cold, step="cold")]
        for i, r in enumerate(rephrasings):
            out.append(self._warm("W3", corpus_path, r, step=f"reask_{i+1}"))
        # Genuinely different follow-up work: should NOT reuse the timeout
        # reasoning, so the gate re-thinks (a real cold call).
        out.append(self._warm("W3", corpus_path,
            "Separately, add automatic retry with exponential backoff on 503s.",
            step="unrelated_expected_miss"))
        return out

    def w4(self, corpus_path):
        """W4 — one agent does the audit; N concurrent teammates ask for it
        differently and reuse the same reasoning."""
        cold = ("Audit how this codebase handles errors around the network layer "
                "and point out where it swallows exceptions inconsistently.")
        rephrasings = [
            "Where does our error handling around requests hide failures we "
            "should be surfacing?",
            "Find the spots where exceptions get silently eaten in the transport "
            "code.",
            "Is exception handling in the HTTP path consistent? Where isn't it?",
        ]
        out = [self._cold("W4", corpus_path, cold, step="cold_prime")]

        def _hit(i):
            return self._warm("W4", corpus_path, rephrasings[i - 1],
                              step=f"concurrent_agent_{i}")

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=3) as ex:
            results = list(ex.map(_hit, range(1, 4)))
        parallel_ms = int((time.time() - t0) * 1000)
        for m in results:
            m.extra["parallel_wallclock_ms"] = parallel_ms
        out.extend(results)
        return out

    def w5(self, corpus_path):
        """W5 — multi-step: plan, then later recall of that plan (reworded, reuse),
        then a genuinely new sub-task (cold)."""
        step1 = ("I want to add full type annotations to this app. Give me a "
                 "sequenced plan: what to annotate first, what's risky, and how "
                 "to validate each stage.")
        step2 = ("Remind me the sequenced plan we made for adding type hints "
                 "here — the order to annotate in and the risks at each stage.")
        out = [self._cold("W5", corpus_path, step1, step="step1_plan_cold")]
        out.append(self._warm("W5", corpus_path, step2, step="step2_recall_plan"))
        # A concretely different sub-task still costs reasoning (cold).
        out.append(self._cold("W5", corpus_path,
            "Now actually write the annotated signature for the request-dispatch "
            "function and justify the types you chose.", step="step3_new_cold"))
        return out

    def w6(self, corpus_path):
        """W6 — 'fork': agent A reasons about the plugin system; agent B, building
        an extension, asks about it differently and reuses A's reasoning."""
        parent = ("Explain how third-party plugins hook into this framework so I "
                  "can write one that adds a new report section.")
        child = ("I'm writing a plugin — how does a third-party plugin hook into "
                 "this framework and register itself?")
        return [self._cold("W6", corpus_path, parent, role="parent", step="parent_cold"),
                # child shares the role so the (role-scoped) semantic lookup can hit
                self._warm("W6", corpus_path, child, role="parent", step="child_reuse")]

    def w7(self, corpus_path):
        """W7 — accumulate reasoning, consolidate into the KnowledgeStore, reuse it."""
        # Four genuinely distinct onboarding investigations — they should NOT
        # reuse each other (different questions); their reasoning is what gets
        # consolidated into one summary a later agent can reuse instead.
        tasks = [
            "I'm onboarding onto this ORM. Trace how a query object actually "
            "becomes SQL under the hood so I know where to debug.",
            "Walk me through how sessions and the unit-of-work track dirty state "
            "and decide what to flush.",
            "How do connection pooling and transaction scoping fit together here? "
            "I need to reason about where connections come from.",
            "Where do I hook in custom column types or dialect-specific behavior?",
        ]
        out = [self._cold("W7", corpus_path, t, step=f"accumulate_{i+1}")
               for i, t in enumerate(tasks)]

        # Consolidation: one call distills everything into a dense summary, stored
        # in the KnowledgeStore with its EXACT output_tokens as the cost.
        context = self._load_context(corpus_path)
        t0 = time.time()
        r = self._call_claude(context,
            "Produce a dense ~300-token knowledge summary of this codebase's "
            "architecture that a new agent could use INSTEAD of re-reading the "
            "source. Cover execution model, config, extension points, lifecycle.")
        dur = int((time.time() - t0) * 1000)
        region = CORPUS["W7"]
        rhash = self._codebase_hash(corpus_path)
        summary_tokens = r["output_tokens"]  # exact, from usage
        # Full re-derivation cost this summary lets a later agent skip: the whole
        # consolidation call (codebase input context + output), not the ~500-token
        # summary size. That is what a knowledge hit actually avoids.
        consolidation_cost = (r["input_tokens"] + r["cache_creation_input_tokens"]
                              + r["cache_read_input_tokens"] + r["output_tokens"])
        self.knowledge_store.submit(
            region=region, summary=r["answer_text"], region_hash=rhash,
            source_agent="cloud-W7", token_count=summary_tokens,
            full_cost_tokens=consolidation_cost, role="analyzer",
        )
        out.append(CloudMetric(
            workload="W7", corpus=corpus_path.name, step="consolidate",
            agent=self.model, is_first_session=True, duration_ms=dur,
            reasoning_tokens=r["thinking_tokens"], had_thinking_block=r["had_thinking_block"],
            input_tokens=r["input_tokens"],
            cache_creation_input_tokens=r["cache_creation_input_tokens"],
            cache_read_input_tokens=r["cache_read_input_tokens"],
            output_tokens=r["output_tokens"],
            extra={"summary_token_count": summary_tokens},
        ))

        # A later agent queries the knowledge pool instead of re-analyzing.
        hit = self.knowledge_store.query(region, rhash, role="analyzer")
        out.append(CloudMetric(
            workload="W7", corpus=corpus_path.name, step="knowledge_reuse",
            agent=self.model, is_first_session=False, duration_ms=0,
            knowledge_store_hit=hit is not None,
            knowledge_tokens_saved=(self.knowledge_store.last_summary_tokens or 0) if hit else 0,
            total_tokens_avoided=(self.knowledge_store.last_full_cost_saved or 0) if hit else 0,
        ))
        return out

    def w8(self, corpus_path):
        """W8 — largest corpus, hardest task; cold cost + one reworded reuse."""
        cold = ("We need to modernize this codebase: type hints, tighter error "
                "handling, and a cleaner public API. Give me a phased plan with "
                "sequencing, blast radius, and how to validate each phase.")
        warm = ("Lay out a staged roadmap for bringing this project up to date — "
                "typing, error handling, API cleanup — and how we de-risk each "
                "stage.")
        return [self._cold("W8", corpus_path, cold, step="cold_large"),
                self._warm("W8", corpus_path, warm, step="warm_reworded_large")]

    # -- Dispatch ------------------------------------------------------------

    _DISPATCH = {"W1": "w1", "W2": "w2", "W3": "w3", "W4": "w4",
                 "W5": "w5", "W6": "w6", "W7": "w7", "W8": "w8"}

    def run_workload(self, name: str) -> list[CloudMetric]:
        if name not in self._DISPATCH:
            raise ValueError(f"Unknown workload {name}")
        self._reset_stores()  # cold calls genuinely cold
        corpus = self._corpus_path(name)
        return getattr(self, self._DISPATCH[name])(corpus)


# ---------------------------------------------------------------------------
# Results / rollup
# ---------------------------------------------------------------------------

def summarize_cloud(metrics: list[CloudMetric]) -> dict:
    thinking_hits = [m for m in metrics if m.thinking_store_hit]
    cold_calls = [m for m in metrics if m.is_first_session]
    return {
        "n_sessions": len(metrics),
        "n_cold_calls": len(cold_calls),
        "n_thinking_hits": len(thinking_hits),
        "thinking_hit_rate": (len(thinking_hits) / len(metrics)) if metrics else 0.0,
        "total_reasoning_tokens_saved": sum(m.reasoning_tokens_saved for m in metrics),
        "total_knowledge_tokens_saved": sum(m.knowledge_tokens_saved for m in metrics),
        # The headline: full derivation cost avoided by reuse (codebase input +
        # output), which scales with codebase size -- not the thinking sliver.
        "total_tokens_avoided": sum(m.total_tokens_avoided for m in metrics),
        "baseline_reasoning_tokens": sum(m.reasoning_tokens or 0 for m in cold_calls),
        "total_input_tokens": sum(m.input_tokens for m in metrics),
        "total_cache_creation_input_tokens": sum(m.cache_creation_input_tokens for m in metrics),
        "total_cache_read_input_tokens": sum(m.cache_read_input_tokens for m in metrics),
        "total_output_tokens": sum(m.output_tokens for m in metrics),
    }


def write_results(workload: str, metrics: list[CloudMetric]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    path = RESULTS_DIR / f"{workload}_cloud_{ts}.json"
    path.write_text(json.dumps({
        "workload": workload,
        "corpus": CORPUS.get(workload, "n/a"),
        "backend": "cloud",
        "model": DEFAULT_MODEL,
        "timestamp": ts,
        "metrics": [asdict(m) for m in metrics],
        "summary": summarize_cloud(metrics),
    }, indent=2))
    return path


def write_rollup(all_metrics: dict[str, list[CloudMetric]], model: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    path = RESULTS_DIR / f"rollup_cloud_{ts}.md"
    lines = [
        f"# CacheFlow Cloud Experiments — {model}",
        "",
        f"Timestamp: {ts}",
        "",
        "Reasoning-token reuse: extended-thinking tokens NOT regenerated because a "
        "prior agent's reasoning was reused from the ThinkingStore/KnowledgeStore. "
        "All token counts are exact (from the Anthropic `usage` object), never estimated.",
        "",
        "| Workload | Corpus | Sessions | Cold Calls | Thinking Hits | Baseline Reasoning | Reasoning Saved | Knowledge Saved | Tokens Avoided |",
        "|----------|--------|----------|------------|---------------|--------------------|-----------------|-----------------|----------------|",
    ]
    for w in sorted(all_metrics):
        s = summarize_cloud(all_metrics[w])
        lines.append(
            f"| {w} | {CORPUS.get(w, 'n/a')} | {s['n_sessions']} | {s['n_cold_calls']} | "
            f"{s['n_thinking_hits']} ({s['thinking_hit_rate']*100:.0f}%) | "
            f"{s['baseline_reasoning_tokens']} | {s['total_reasoning_tokens_saved']} | "
            f"{s['total_knowledge_tokens_saved']} | {s['total_tokens_avoided']} |"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def _print_summary(workload: str, metrics: list[CloudMetric]):
    s = summarize_cloud(metrics)
    print(f"  [{workload}/cloud corpus={CORPUS.get(workload)}] "
          f"{s['n_sessions']} sessions, {s['n_cold_calls']} cold, "
          f"{s['n_thinking_hits']} thinking hits, "
          f"reasoning_saved={s['total_reasoning_tokens_saved']}, "
          f"knowledge_saved={s['total_knowledge_tokens_saved']}, "
          f"TOKENS_AVOIDED={s['total_tokens_avoided']}, "
          f"in={s['total_input_tokens']}(+{s['total_cache_creation_input_tokens']}c "
          f"+{s['total_cache_read_input_tokens']}r) out={s['total_output_tokens']}")


def _kill_lingering_local_model():
    """Kill any lingering run.py process still holding Qwen3 8B in memory.

    get_global_engine() keeps the model loaded for the process lifetime, so a
    background run.py that finished its workloads but didn't exit will hold
    6-10 GB of RAM indefinitely. On a 16 GB Mac this leaves no room for
    anything else and causes a full system freeze.
    """
    try:
        result = subprocess.run(
            ["pgrep", "-f", "python.*experiments/run.py"],
            capture_output=True, text=True, timeout=5,
        )
        pids = [p.strip() for p in result.stdout.splitlines() if p.strip()]
        if not pids:
            return
        own_pid = str(os.getpid())
        pids = [p for p in pids if p != own_pid]
        if not pids:
            return
        print(f"  [safety] killing lingering local-model processes: {pids}")
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        time.sleep(2)
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    except Exception:
        pass


def _check_memory_headroom(min_free_gb: float = 2.0):
    """Warn if available memory is dangerously low before starting API calls."""
    try:
        vm = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=5,
        )
        free_pages = 0
        page_size = 16384
        for line in vm.stdout.splitlines():
            if "Pages free" in line:
                free_pages += int(line.split(":")[1].strip().rstrip("."))
            if "Pages speculative" in line:
                free_pages += int(line.split(":")[1].strip().rstrip("."))
        free_gb = (free_pages * page_size) / (1024 ** 3)
        if free_gb < min_free_gb:
            print(f"  [warning] only {free_gb:.1f} GB free memory — "
                  f"consider closing apps to avoid a system freeze")
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", nargs="+", default=["all"],
                   help="One or more of W1..W8, or 'all'.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--effort", default=DEFAULT_THINKING_EFFORT, choices=_EFFORT_LEVELS,
                   help="Adaptive-thinking effort tier per cold call.")
    args = p.parse_args()

    _kill_lingering_local_model()
    _check_memory_headroom()

    runner = CloudWorkloadRunner(model=args.model, thinking_effort=args.effort)
    workloads = list(CORPUS.keys()) if args.workload == ["all"] else args.workload

    all_metrics: dict[str, list[CloudMetric]] = {}
    for w in workloads:
        print(f"\n== {w} (cloud) — corpus: {CORPUS.get(w)} ==")
        try:
            metrics = runner.run_workload(w)
            all_metrics[w] = metrics
            path = write_results(w, metrics)
            _print_summary(w, metrics)
            print(f"  results: {path}")
        except Exception as e:
            print(f"  [{w}] FAILED: {e.__class__.__name__}: {e}")
            import traceback; traceback.print_exc()
            all_metrics[w] = []

    path = write_rollup(all_metrics, args.model)
    print(f"\nRollup: {path}")


if __name__ == "__main__":
    main()
