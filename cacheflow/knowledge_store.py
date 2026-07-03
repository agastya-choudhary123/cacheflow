"""KnowledgeStore: Share code understanding summaries between agents."""

import sqlite3
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any


class KnowledgeStore:
    """Store and retrieve regional knowledge summaries for agents."""

    def __init__(self, db_path: str = ".cacheflow/knowledge.db"):
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def init_db(self):
        """Initialize SQLite schema for knowledge entries."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                region TEXT NOT NULL,
                region_hash TEXT NOT NULL,
                role TEXT,
                summary TEXT NOT NULL,
                source_agent TEXT NOT NULL,
                token_count INTEGER,
                full_cost_tokens INTEGER,
                supersedes_id INTEGER,
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (supersedes_id) REFERENCES knowledge_entries(id)
            )
        """
        )
        cols = {r[1] for r in conn.execute("PRAGMA table_info(knowledge_entries)")}
        if "full_cost_tokens" not in cols:
            conn.execute("ALTER TABLE knowledge_entries ADD COLUMN full_cost_tokens INTEGER")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_region_hash
            ON knowledge_entries(region, region_hash, role)
        """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_created_at
            ON knowledge_entries(created_at)
        """
        )
        conn.commit()
        conn.close()

    def submit(
        self, region: str, summary: str, source_agent: str, region_hash: str, role: Optional[str] = None,
        token_count: Optional[int] = None, full_cost_tokens: Optional[int] = None,
    ) -> int:
        """Store a knowledge summary for a region.

        `token_count`, if given, must be the real, exact token cost of
        `summary` (e.g. from the API usage that produced it) -- never derived
        from `len(summary)` here, since character count isn't a token count.
        If the caller doesn't have an exact figure, leave it None; it's
        stored as NULL rather than a fabricated number.

        `full_cost_tokens` is the total cost of the derivation that PRODUCED this
        summary (codebase input context + full output of the consolidation call).
        This -- not the ~500-token summary size -- is what a reuse actually saves:
        an agent injects the small summary instead of re-reading and re-analyzing
        the whole region from scratch. That is the real, exact figure to credit
        on a knowledge hit.
        """
        conn = sqlite3.connect(self.db_path)

        # Find and mark previous entries for this region+role as superseded
        cursor = conn.execute(
            "SELECT id FROM knowledge_entries WHERE region = ? AND role IS ?",
            (region, role),
        )
        previous_id = cursor.fetchone()

        entry_id = None
        try:
            cursor = conn.execute(
                """
                INSERT INTO knowledge_entries
                (region, region_hash, role, summary, source_agent, token_count,
                 full_cost_tokens, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (region, region_hash, role, summary, source_agent, token_count,
                 full_cost_tokens, datetime.now()),
            )
            entry_id = cursor.lastrowid

            # Mark previous entry as superseded
            if previous_id:
                conn.execute("UPDATE knowledge_entries SET supersedes_id = ? WHERE id = ?", (entry_id, previous_id[0]))

            conn.commit()
        finally:
            conn.close()

        return entry_id

    def query(
        self, region: str, current_region_hash: str, role: Optional[str] = None, max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """
        Query for valid knowledge summaries for a region.
        Returns: summary text if valid (region_hash matches), else None
        """
        # Exposed on a hit so the caller can credit the real saving: the full
        # re-derivation cost avoided (last_full_cost_saved), plus the summary's
        # own token size (last_summary_tokens) as the narrow secondary metric.
        self.last_full_cost_saved = None
        self.last_summary_tokens = None
        conn = sqlite3.connect(self.db_path)

        # Try exact role match first
        query_role = role
        cursor = conn.execute(
            """
            SELECT summary, token_count, full_cost_tokens FROM knowledge_entries
            WHERE region = ? AND region_hash = ? AND role IS ?
            ORDER BY created_at DESC LIMIT 1
        """,
            (region, current_region_hash, query_role),
        )
        result = cursor.fetchone()

        # Fall back to generic (role=NULL) if no exact match
        if not result and role is not None:
            cursor = conn.execute(
                """
                SELECT summary, token_count, full_cost_tokens FROM knowledge_entries
                WHERE region = ? AND region_hash = ? AND role IS NULL
                ORDER BY created_at DESC LIMIT 1
            """,
                (region, current_region_hash),
            )
            result = cursor.fetchone()

        conn.close()

        if result:
            summary, self.last_summary_tokens, self.last_full_cost_saved = result
            if max_tokens and len(summary) > max_tokens:
                summary = summary[:max_tokens]
            return summary

        return None

    def garbage_collect(self, older_than_days: int = 60) -> int:
        """Garbage collect old/unused knowledge entries."""
        conn = sqlite3.connect(self.db_path)
        cutoff = datetime.now() - timedelta(days=older_than_days)

        # Delete entries that are superseded or old
        cursor = conn.execute(
            """
            DELETE FROM knowledge_entries
            WHERE supersedes_id IS NOT NULL OR created_at < ?
        """,
            (cutoff,),
        )
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted_count

    def list_entries(self, region: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """List knowledge entries."""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT id, region, role, source_agent, created_at, token_count FROM knowledge_entries"
        params = []

        if region:
            query += " WHERE region = ?"
            params.append(region)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "id": row[0],
                "region": row[1],
                "role": row[2],
                "source_agent": row[3],
                "created_at": row[4],
                "token_count": row[5],
            }
            for row in rows
        ]
