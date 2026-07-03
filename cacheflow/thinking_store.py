"""ThinkingStore: Cache and reuse extended thinking blocks across sessions."""

import os
import sqlite3
import pickle
import json
import hashlib
import math
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any


_embedding_model = None  # Global cache for embedding model

# Reuse gates for all-MiniLM-L6-v2, calibrated to THIS domain (see query()).
#
# Key empirical finding (scratchpad gate experiments): absolute cosine values do
# NOT transfer across domains. Short Quora-style paraphrases score ~0.86, but our
# long, multi-clause engineering-task rewordings score only ~0.45-0.84 while
# genuinely-unrelated engineering tasks sit at <=0.26. There is a clean gap, but
# it lives ~0.25 lower than Quora-scale intuition. The old 0.70 USE gate recalled
# only ~30% of true reuse; 0.55 recalled ~80%. Recalibrated to the measured
# in-domain band (best-F1 threshold ~0.45): reuse the mid-band directly and
# reserve validation for the genuinely ambiguous low edge.
#
# For domains where this absolute scale is wrong, query() prefers a
# BACKGROUND-NORMALIZED z-score (how far the candidate stands out from the
# query's similarity to the rest of the store) which is scale-invariant; the
# absolute thresholds below are the small-pool fallback.
USE_THRESHOLD = 0.62       # close paraphrase -> reuse reasoning directly
VALIDATE_THRESHOLD = 0.40  # medium match -> reuse but flag for validation
# Background z-score bands (used when the store has enough blocks for stable
# statistics). A true in-domain reuse measured z >= 3.6; unrelated tasks z <= 2.7.
Z_USE_THRESHOLD = 4.0
Z_VALIDATE_THRESHOLD = 2.8
Z_MIN_POOL = 8             # need this many background blocks for a stable mean/std


class ThinkingStore:
    """Store and retrieve cached thinking blocks with semantic search."""

    def __init__(self, db_path: str = ".cacheflow/thinking.db"):
        self.db_path = db_path
        self.qdrant_client = None
        # Exact token_count of the most recent exact-hash hit (None if the last
        # query() call missed, or hasn't run yet). Set by _exact_lookup so CLI
        # callers can report real savings for that specific call without
        # changing query()'s existing 3-tuple return contract.
        # last_tokens_saved and the pending 'validate' candidate are per-query
        # scratch state read back by the caller right after query(). W4 runs
        # several agents concurrently against ONE shared store, so these must be
        # thread-local -- otherwise one thread's query() would clobber another's
        # between its query() and its read/commit, mis-logging reuse. Exposed as
        # properties so existing single-threaded callers are unaffected.
        self._tls = threading.local()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @property
    def last_tokens_saved(self) -> Optional[int]:
        return getattr(self._tls, "last_tokens_saved", None)

    @last_tokens_saved.setter
    def last_tokens_saved(self, value: Optional[int]) -> None:
        self._tls.last_tokens_saved = value

    @property
    def last_full_cost_saved(self) -> Optional[int]:
        """Full origin-call cost (codebase input + full output) of the most
        recent reuse -- i.e. what skipping that derivation actually avoided, not
        just the isolated thinking tokens (last_tokens_saved). Thread-local for
        the same concurrency reason as last_tokens_saved."""
        return getattr(self._tls, "last_full_cost_saved", None)

    @last_full_cost_saved.setter
    def last_full_cost_saved(self, value: Optional[int]) -> None:
        self._tls.last_full_cost_saved = value

    @property
    def _pending_validate(self):
        return getattr(self._tls, "pending_validate", None)

    @_pending_validate.setter
    def _pending_validate(self, value) -> None:
        self._tls.pending_validate = value

    def init_db(self):
        """Initialize SQLite schema for thinking blocks."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS thinking_blocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thinking_block TEXT NOT NULL,
                embedding BLOB,
                problem_hash TEXT NOT NULL,
                codebase_hash TEXT NOT NULL,
                delta JSON,
                problem_type TEXT,
                task_description TEXT,
                role TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                accessed_at TIMESTAMP,
                validation_passed INTEGER DEFAULT 0,
                validation_reason TEXT,
                source_agent TEXT,
                session_id TEXT,
                token_count INTEGER,
                full_cost_tokens INTEGER,
                embedding_model_version TEXT DEFAULT 'all-MiniLM-L6-v2'
            )
        """
        )
        # Migration for DBs created before full_cost_tokens existed.
        cols = {r[1] for r in conn.execute("PRAGMA table_info(thinking_blocks)")}
        if "full_cost_tokens" not in cols:
            conn.execute("ALTER TABLE thinking_blocks ADD COLUMN full_cost_tokens INTEGER")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_problem_hash
            ON thinking_blocks(problem_hash)
        """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_codebase_hash
            ON thinking_blocks(codebase_hash)
        """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_accessed_at
            ON thinking_blocks(accessed_at)
        """
        )
        # Every successful reuse (exact-hash hit) logs the exact token_count of
        # the block it reused -- that IS the number of thinking tokens NOT
        # regenerated, in real Anthropic-reported tokens, not an estimate.
        # cf thinking stats sums this table for the headline savings number.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS thinking_reuse_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thinking_block_id INTEGER NOT NULL,
                tokens_saved INTEGER,
                reused_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (thinking_block_id) REFERENCES thinking_blocks(id)
            )
        """
        )
        conn.commit()
        conn.close()

    def _get_embedding_model(self):
        """Lazy-load and cache the embedding model globally.

        Uses all-MiniLM-L6-v2 (~90 MB, 384-dim) -- small enough to load safely
        on any machine. It replaced intfloat/e5-mistral-7b-instruct, a 7B,
        ~13 GB local model whose load swap-stormed and hard-froze/kernel-panicked
        memory-constrained hosts. Byte-identical task reuse hits the exact-hash
        fast path and never needs this; the embedding model exists for the
        realistic case where a later agent asks a *similar* (not identical)
        question, which the semantic/validation gate handles.

        CACHEFLOW_DISABLE_EMBEDDINGS=1 still skips loading entirely (semantic
        reuse then degrades to exact-hash only).
        """
        global _embedding_model
        if os.environ.get("CACHEFLOW_DISABLE_EMBEDDINGS"):
            return None
        if _embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                _embedding_model = False  # Mark as unavailable
        return _embedding_model if _embedding_model is not False else None

    def submit(
        self,
        thinking_block: str,
        problem_hash: str,
        codebase_hash: str,
        **metadata,
    ) -> None:
        """Store a thinking block with metadata and embedding.

        `token_count` (in metadata) must be the exact, real cost of this
        thinking block -- e.g. the `output_tokens` field from the Anthropic
        API's own usage accounting for the turn it came from (see
        hooks.extract_thinking_blocks_from_transcript). It is never derived
        from `len(thinking_block)` here: character count is not a token
        count, and guessing one from the other would make every downstream
        savings metric (cf thinking stats) fictional. If the caller doesn't
        have an exact count, leave it unset -- it's stored as NULL rather
        than a fabricated number.
        """
        embedding_blob = None
        model = self._get_embedding_model()
        if model:
            try:
                # Embed the PROBLEM (task description), not the reasoning text:
                # query() searches by the incoming task, so the stored vector
                # must be in the same space (task-vs-task) for similarity to
                # mean "these two requests are asking the same thing". Falls
                # back to the reasoning text only if no task_description was
                # supplied.
                embed_text = metadata.get("task_description") or thinking_block
                embedding = model.encode(embed_text)
                embedding_blob = pickle.dumps(embedding)
            except Exception:
                # Fallback: store without embedding if encoding fails
                embedding_blob = None

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT INTO thinking_blocks
            (thinking_block, embedding, problem_hash, codebase_hash,
             problem_type, task_description, role, delta, source_agent, session_id,
             token_count, full_cost_tokens)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                thinking_block,
                embedding_blob,
                problem_hash,
                codebase_hash,
                metadata.get("problem_type"),
                metadata.get("task_description"),
                metadata.get("role"),
                json.dumps(metadata.get("delta", {})),
                metadata.get("source_agent"),
                metadata.get("session_id"),
                metadata.get("token_count"),
                metadata.get("full_cost_tokens"),
            ),
        )
        conn.commit()
        conn.close()

        # Also index in Qdrant (if available)
        if embedding_blob:
            self._index_in_qdrant(embedding, problem_hash, metadata)

    def query(
        self,
        problem_description: str,
        role: Optional[str] = None,
        confidence_threshold: float = 0.85,
    ) -> Tuple[Optional[str], float, str]:
        """
        Query for cached thinking blocks.
        Returns: (thinking_block, confidence, action)
        Actions: "use_directly", "validate", "re_think"
        """
        problem_hash = self._hash_problem(problem_description)
        exact_hit = self._exact_lookup(problem_hash, role)
        if exact_hit:
            return exact_hit, 1.0, "use_directly"

        model = self._get_embedding_model()
        if not model:
            return None, 0, "re_think"

        try:
            embedding = model.encode(problem_description)
        except Exception:
            return None, 0, "re_think"

        # Pre-filter floor is the low (validate) gate; final banding is below.
        semantic_hits = self._semantic_search(
            embedding, limit=5, threshold=VALIDATE_THRESHOLD, role=role
        )

        if not semantic_hits:
            return None, 0, "re_think"

        best_hit = semantic_hits[0]
        confidence = best_hit["confidence"]

        # Thresholds re-tuned for all-MiniLM-L6-v2. On realistically-varied
        # (non-identical) human task phrasings this small embedder does NOT
        # cleanly separate "same task, reworded" from "different but related
        # task" -- measured cosine for genuinely-unrelated tasks tops out ~0.46,
        # while heavily-reworded true paraphrases can dip into that range too.
        # So the gate is deliberately conservative: only a close paraphrase
        # (> USE_THRESHOLD) is reused outright, a medium one (> VALIDATE_
        # THRESHOLD) is flagged for validation, and everything else re-thinks
        # (a cold call). This trades recall for near-zero false reuse, which is
        # the honest posture -- reusing the wrong reasoning is worse than
        # paying to re-derive it. For higher recall, confirm "validate"
        # candidates with a cheap model call rather than lowering the floor.
        # Band the best hit. Prefer the scale-invariant background z-score when
        # the store holds enough blocks for a stable mean/std; otherwise fall
        # back to the domain-calibrated absolute cosine thresholds.
        pool_size = best_hit.get("pool_size", 0)
        zscore = best_hit.get("zscore", 0.0)
        if pool_size >= Z_MIN_POOL and best_hit.get("zscore") is not None:
            if zscore < Z_VALIDATE_THRESHOLD:
                return None, 0, "re_think"
            action = "use_directly" if zscore >= Z_USE_THRESHOLD else "validate"
        else:
            if confidence < VALIDATE_THRESHOLD:
                return None, 0, "re_think"
            action = "use_directly" if confidence >= USE_THRESHOLD else "validate"
        if action == "use_directly":
            # High-confidence: a definite reuse. Log the matched block's exact
            # token_count now so reasoning_tokens_saved is real, not estimated.
            if best_hit.get("id") is not None:
                conn = sqlite3.connect(self.db_path)
                self._log_reuse(conn, best_hit["id"], best_hit.get("token_count"))
                conn.commit()
                conn.close()
            self.last_full_cost_saved = best_hit.get("full_cost_tokens")
            self._pending_validate = None
        else:
            # Medium-confidence: NOT a reuse yet. Stash the candidate so the
            # caller can confirm it (e.g. a cheap validator model) and then call
            # commit_validated_reuse(); logging here would record savings for a
            # reuse that may be rejected.
            self._pending_validate = (best_hit.get("id"), best_hit.get("token_count"),
                                      best_hit.get("full_cost_tokens"))
            self.last_tokens_saved = None
            self.last_full_cost_saved = None
        return best_hit["thinking_block"], confidence, action

    def commit_validated_reuse(self) -> Optional[int]:
        """Confirm the most recent 'validate'-band candidate as a real reuse.

        Call this only after an external check (see the cloud runner's Haiku
        validator) has confirmed the cached reasoning actually answers the new
        task. Logs the candidate's exact token_count into thinking_reuse_log and
        returns it (also on self.last_tokens_saved). No-op if there's no pending
        candidate. Keeps reuse accounting honest: a 'validate' hit only counts
        once it's been confirmed, never on retrieval alone.
        """
        pending = getattr(self, "_pending_validate", None)
        if not pending or pending[0] is None:
            return None
        block_id, token_count, full_cost = pending
        conn = sqlite3.connect(self.db_path)
        self._log_reuse(conn, block_id, token_count)
        conn.commit()
        conn.close()
        self.last_full_cost_saved = full_cost
        self._pending_validate = None
        return token_count

    def _exact_lookup(self, problem_hash: str, role: Optional[str] = None) -> Optional[str]:
        """Fast exact hash lookup. On a hit, logs the exact tokens this reuse
        saved (the matched block's real token_count) into thinking_reuse_log,
        and records it on self.last_tokens_saved for the caller to report.
        """
        self.last_tokens_saved = None
        self.last_full_cost_saved = None
        conn = sqlite3.connect(self.db_path)
        query = ("SELECT id, thinking_block, token_count, full_cost_tokens "
                 "FROM thinking_blocks WHERE problem_hash = ?")
        params = [problem_hash]

        if role:
            query += " AND role = ?"
            params.append(role)

        query += " ORDER BY created_at DESC LIMIT 1"

        cursor = conn.execute(query, params)
        result = cursor.fetchone()
        if result:
            block_id, thinking_block, token_count, full_cost = result
            self._log_reuse(conn, block_id, token_count)
            conn.commit()
            self.last_full_cost_saved = full_cost
        conn.close()
        return result[1] if result else None

    def _log_reuse(self, conn, block_id: int, token_count: Optional[int]) -> None:
        """Record a reuse of `block_id`: append its real token_count to
        thinking_reuse_log, bump accessed_at, and set self.last_tokens_saved so
        the caller can report exactly what this reuse saved. Shared by the
        exact-hash and semantic paths so both account savings identically. Does
        not commit -- the caller owns the transaction.
        """
        conn.execute(
            "INSERT INTO thinking_reuse_log (thinking_block_id, tokens_saved) VALUES (?, ?)",
            (block_id, token_count),
        )
        conn.execute(
            "UPDATE thinking_blocks SET accessed_at = ? WHERE id = ?",
            (datetime.now().isoformat(), block_id),
        )
        self.last_tokens_saved = token_count

    def get_reuse_stats(self) -> Dict[str, Any]:
        """Exact cumulative reuse savings: real count of reuses and the sum of
        their real token_count values -- never estimated, since every row in
        thinking_reuse_log was logged from a real stored token_count at the
        moment of reuse (NULL token_count entries, where the source block had
        no exact count, are excluded from the sum rather than counted as 0).
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT COUNT(*), SUM(tokens_saved) FROM thinking_reuse_log"
        )
        reuse_count, total_tokens_saved = cursor.fetchone()
        conn.close()
        return {
            "reuse_count": reuse_count or 0,
            "total_tokens_saved": total_tokens_saved or 0,
        }

    def _semantic_search(
        self, embedding, limit: int = 5, threshold: float = VALIDATE_THRESHOLD,
        role: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find similar thinking blocks by embedding cosine similarity.

        Prefers Qdrant when it's reachable; otherwise falls back to an
        in-SQLite cosine scan over the embedding BLOBs already stored on every
        row. The fallback is what makes semantic reuse actually work on a plain
        laptop: without it, a stopped Qdrant server silently returns nothing and
        every non-identical task degrades to a cold call. Both paths apply the
        same age-decay and return the matched row's `id` and `token_count` so a
        reuse can be logged with its real saved-token count.
        """
        if self._ensure_qdrant():
            try:
                return self._qdrant_search(embedding, limit, threshold, role)
            except Exception:
                pass  # fall through to the local scan
        return self._sqlite_semantic_search(embedding, limit, threshold, role)

    def _qdrant_search(self, embedding, limit, threshold, role):
        qfilter = None
        if role:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            qfilter = Filter(must=[FieldCondition(key="role", match=MatchValue(value=role))])
        results = self.qdrant_client.search(
            collection_name="thinking_blocks",
            query_vector=embedding.tolist(),
            limit=limit,
            score_threshold=threshold,
            query_filter=qfilter,
        )
        hits = []
        for result in results:
            created_at_str = result.payload.get("created_at", datetime.now().isoformat())
            try:
                created_at = datetime.fromisoformat(created_at_str)
            except (ValueError, TypeError):
                created_at = datetime.now()
            age_days = (datetime.now() - created_at).days
            confidence = result.score * math.exp(-0.1 * age_days)
            hits.append({
                "thinking_block": result.payload.get("thinking_block", ""),
                "confidence": confidence,
                "id": result.payload.get("sqlite_id", result.id),
                "token_count": result.payload.get("token_count"),
                "age_days": age_days,
            })
        return sorted(hits, key=lambda x: x["confidence"], reverse=True)

    def _sqlite_semantic_search(self, embedding, limit, threshold, role):
        """Cosine-similarity scan over stored embedding BLOBs (no external DB)."""
        import numpy as np

        q = np.asarray(embedding, dtype=np.float32).ravel()
        qn = float(np.linalg.norm(q))
        if qn == 0.0:
            return []
        q = q / qn

        conn = sqlite3.connect(self.db_path)
        sql = ("SELECT id, thinking_block, embedding, token_count, created_at, full_cost_tokens "
               "FROM thinking_blocks WHERE embedding IS NOT NULL")
        params: list = []
        if role:
            sql += " AND role = ?"
            params.append(role)
        rows = conn.execute(sql, params).fetchall()
        conn.close()

        scored = []
        now = datetime.now()
        for block_id, block, emb_blob, token_count, created_at_str, full_cost in rows:
            try:
                v = np.asarray(pickle.loads(emb_blob), dtype=np.float32).ravel()
            except Exception:
                continue
            vn = float(np.linalg.norm(v))
            if vn == 0.0 or v.shape != q.shape:
                continue
            score = float(np.dot(q, v / vn))
            try:
                created_at = datetime.fromisoformat(created_at_str) if created_at_str else now
            except (ValueError, TypeError):
                created_at = now
            age_days = (now - created_at).days
            scored.append((score, block_id, block, token_count, age_days, full_cost))

        # Background-normalized z-score: how far each candidate's raw cosine
        # stands out from the query's similarity to the WHOLE pool. This is
        # scale-invariant, so it works whether "similar" means 0.86 (Quora) or
        # 0.50 (engineering tasks) -- the absolute threshold can't do both.
        # Leave-one-out is essential: a candidate must NOT be in its own
        # background, or a lone true match inflates the mean/std and crushes its
        # own z-score (worse the smaller the pool). Each z uses the mean/std of
        # every OTHER stored block's cosine to this query.
        all_scores = np.array([s[0] for s in scored], dtype=np.float64)
        pool_size = len(all_scores)
        total = all_scores.sum()
        sq_total = np.square(all_scores).sum()

        def loo_z(score):
            n = pool_size - 1
            if n < 2:
                return 0.0
            mean = (total - score) / n
            var = max((sq_total - score * score) / n - mean * mean, 0.0)
            std = math.sqrt(var)
            return (score - mean) / std if std > 1e-9 else 0.0

        hits = []
        for score, block_id, block, token_count, age_days, full_cost in scored:
            confidence = score * math.exp(-0.1 * age_days)
            if confidence >= threshold:
                zscore = loo_z(score)
                hits.append({
                    "thinking_block": block,
                    "confidence": confidence,
                    "zscore": zscore,
                    "pool_size": pool_size,
                    "id": block_id,
                    "token_count": token_count,
                    "full_cost_tokens": full_cost,
                    "age_days": age_days,
                })
        return sorted(hits, key=lambda x: x["confidence"], reverse=True)[:limit]

    def _ensure_qdrant(self) -> bool:
        """Lazy-init Qdrant connection."""
        if self.qdrant_client is None:
            try:
                from qdrant_client import QdrantClient

                self.qdrant_client = QdrantClient(url="http://localhost:6333")
                return True
            except Exception:
                return False
        return True

    def _hash_problem(self, problem_description: str) -> str:
        return hashlib.sha256(problem_description.encode()).hexdigest()

    def _index_in_qdrant(self, embedding, problem_hash: str, metadata: Dict[str, Any]) -> None:
        """Index embedding in Qdrant."""
        if not self._ensure_qdrant():
            return

        try:
            from qdrant_client.models import PointStruct

            point_id = int(hashlib.md5(problem_hash.encode()).hexdigest()[:8], 16)
            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "thinking_block": metadata.get("thinking_block", ""),
                    "problem_hash": problem_hash,
                    "created_at": datetime.now().isoformat(),
                    **metadata,
                },
            )
            self.qdrant_client.upsert(collection_name="thinking_blocks", points=[point])
        except Exception:
            pass  # Graceful fallback to SQLite-only

    def list_blocks(
        self, older_than_days: Optional[int] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List cached thinking blocks."""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT id, problem_hash, role, problem_type, created_at, token_count FROM thinking_blocks"
        params = []

        if older_than_days:
            cutoff = datetime.now() - timedelta(days=older_than_days)
            query += " WHERE created_at < ?"
            params.append(cutoff.isoformat())

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "id": row[0],
                "problem_hash": row[1],
                "role": row[2],
                "problem_type": row[3],
                "created_at": row[4],
                "token_count": row[5],
            }
            for row in rows
        ]

    def garbage_collect(self, older_than_days: int = 60) -> int:
        """Garbage collect old/unused thinking blocks."""
        conn = sqlite3.connect(self.db_path)
        cutoff = datetime.now() - timedelta(days=older_than_days)
        cursor = conn.execute(
            "DELETE FROM thinking_blocks WHERE created_at < ?", (cutoff.isoformat(),)
        )
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted_count
