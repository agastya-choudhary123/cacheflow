"""ThinkingStore: Cache and reuse extended thinking blocks across sessions."""

import sqlite3
import pickle
import json
import hashlib
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any


_embedding_model = None  # Global cache for embedding model


class ThinkingStore:
    """Store and retrieve cached thinking blocks with semantic search."""

    def __init__(self, db_path: str = ".cacheflow/thinking.db"):
        self.db_path = db_path
        self.qdrant_client = None
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

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
                embedding_model_version TEXT DEFAULT 'e5-mistral-7b-instruct'
            )
        """
        )
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
        conn.commit()
        conn.close()

    def _get_embedding_model(self):
        """Lazy-load and cache the embedding model globally."""
        global _embedding_model
        if _embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                _embedding_model = SentenceTransformer("intfloat/e5-mistral-7b-instruct")
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
        """Store a thinking block with metadata and embedding."""
        embedding_blob = None
        model = self._get_embedding_model()
        if model:
            try:
                embedding = model.encode(thinking_block)
                embedding_blob = pickle.dumps(embedding)
            except Exception:
                # Fallback: store without embedding if encoding fails
                embedding_blob = None

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT INTO thinking_blocks
            (thinking_block, embedding, problem_hash, codebase_hash,
             problem_type, task_description, role, delta, source_agent, session_id, token_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                len(thinking_block),
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

        semantic_hits = self._semantic_search(embedding, limit=5, threshold=confidence_threshold)

        if not semantic_hits:
            return None, 0, "re_think"

        best_hit = semantic_hits[0]
        confidence = best_hit["confidence"]

        if confidence > 0.90:
            return best_hit["thinking_block"], confidence, "use_directly"
        elif confidence > 0.85:
            return best_hit["thinking_block"], confidence, "validate"
        else:
            return None, 0, "re_think"

    def _exact_lookup(self, problem_hash: str, role: Optional[str] = None) -> Optional[str]:
        """Fast exact hash lookup."""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT thinking_block FROM thinking_blocks WHERE problem_hash = ?"
        params = [problem_hash]

        if role:
            query += " AND role = ?"
            params.append(role)

        query += " ORDER BY created_at DESC LIMIT 1"

        cursor = conn.execute(query, params)
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None

    def _semantic_search(
        self, embedding, limit: int = 5, threshold: float = 0.85
    ) -> List[Dict[str, Any]]:
        """Search Qdrant for similar thinking blocks."""
        if not self._ensure_qdrant():
            return []

        try:
            results = self.qdrant_client.search(
                collection_name="thinking_blocks",
                query_vector=embedding.tolist(),
                limit=limit,
                score_threshold=threshold,
            )

            hits = []
            for result in results:
                created_at_str = result.payload.get("created_at", datetime.now().isoformat())
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                except (ValueError, TypeError):
                    created_at = datetime.now()

                age_days = (datetime.now() - created_at).days
                decay = math.exp(-0.1 * age_days)
                confidence = result.score * decay

                hits.append(
                    {
                        "thinking_block": result.payload.get("thinking_block", ""),
                        "confidence": confidence,
                        "id": result.id,
                        "age_days": age_days,
                    }
                )

            return sorted(hits, key=lambda x: x["confidence"], reverse=True)
        except Exception:
            return []

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
