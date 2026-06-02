"""Semantic search and live querying across snapshot knowledge."""

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional
from uuid import UUID

from sentence_transformers import SentenceTransformer

from cacheflow.store import CacheFlowStore, Agent, Commit, SnapshotEmbedding
from cacheflow.knowledge_prober import KnowledgeProber, KnowledgeFacets


@dataclass
class SnapshotMatch:
    """Result of a semantic search across snapshots."""

    commit_id: str
    agent_name: str
    task: str
    short_summary: str
    score: float
    created_at: datetime


class SnapshotQueryEngine:
    """Query snapshots semantically and restore them for live interaction."""

    def __init__(self, store: CacheFlowStore):
        """Initialize query engine with a store reference."""
        self.store = store
        self._embedding_model = None

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy-load embedding model."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        return self._embedding_model

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two normalized vectors."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        return max(0.0, min(1.0, dot_product))  # Clamp to [0, 1]

    def query(
        self,
        text: str,
        agent_name: Optional[str] = None,
        top_k: int = 5,
        global_search: bool = False,
    ) -> list[SnapshotMatch]:
        """
        Semantic search across snapshot embeddings.

        Args:
            text: Natural language query
            agent_name: Optional agent filter
            top_k: Number of top results to return
            global_search: Search across all registered CacheFlow projects

        Returns:
            List of SnapshotMatch results ranked by score (descending)
        """
        # Embed query
        query_embedding = self.embedding_model.encode(text, normalize_embeddings=True).tolist()

        matches = []

        if global_search:
            # Search across all registered projects
            from cacheflow.config import get_global_registry
            from pathlib import Path

            registry = get_global_registry()
            for project_path, project_info in registry.items():
                try:
                    db_path = Path(project_info["db"])
                    if db_path.exists():
                        project_store = CacheFlowStore(db_path)
                        project_embeddings = project_store.get_all_embeddings(
                            agent_name=agent_name
                        )
                        matches.extend(
                            self._compute_matches(
                                query_embedding,
                                project_embeddings,
                                project_store,
                            )
                        )
                except Exception:
                    continue  # Skip projects with errors
        else:
            # Search current project only
            embeddings = self.store.get_all_embeddings(agent_name=agent_name)
            matches = self._compute_matches(query_embedding, embeddings, self.store)

        # Sort by score descending and return top-k
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:top_k]

    def _compute_matches(
        self,
        query_embedding: list[float],
        embeddings,
        store: CacheFlowStore,
    ) -> list[SnapshotMatch]:
        """Compute similarity scores for embeddings against query."""
        matches = []
        for emb in embeddings:
            stored_embedding = json.loads(emb.embedding)
            score = self._cosine_similarity(query_embedding, stored_embedding)

            # Fetch commit and agent metadata
            commit = store.get_commit_by_id_prefix(str(emb.commit_id))
            agent = store.get_agent_by_id(emb.agent_id) if hasattr(
                store, "get_agent_by_id"
            ) else None

            if commit and agent:
                matches.append(
                    SnapshotMatch(
                        commit_id=str(commit.id)[:8],
                        agent_name=agent.name,
                        task=commit.task,
                        short_summary=emb.short_summary,
                        score=score,
                        created_at=commit.created_at,
                    )
                )

        return matches

    def query_live(
        self,
        text: str,
        agent_name: Optional[str] = None,
        server=None,
    ) -> Iterator[str]:
        """
        Restore best matching snapshot and ask the model directly.

        Args:
            text: Natural language question
            agent_name: Optional agent filter
            server: LlamaServer instance (required)

        Yields:
            Response chunks as the model streams
        """
        if not server:
            raise ValueError("server parameter required for live querying")

        # Find best match
        matches = self.query(text, agent_name=agent_name, top_k=1)
        if not matches:
            yield "No relevant snapshots found."
            return

        best_match = matches[0]

        # Restore snapshot
        try:
            commit = self.store.get_commit_by_id_prefix(best_match.commit_id)
            if not commit or not Path(commit.snapshot_path).exists():
                yield f"Snapshot for {best_match.commit_id} not found."
                return

            # Restore KV cache to a slot (use slot 1 to avoid interfering with other operations)
            restore_response = server.restore_slot(
                path=commit.snapshot_path,
                slot_id=1,
            )
            if not restore_response.get("success"):
                yield "Failed to restore snapshot."
                return

            # Query the restored model
            response = server.completion(
                prompt=text,
                slot_id=1,
                max_tokens=512,
                stream=True,
            )

            # Yield chunks as they come
            if isinstance(response, dict) and "content" in response:
                # Non-streaming response
                yield response["content"]
            elif isinstance(response, Iterator):
                # Streaming response
                for chunk in response:
                    yield chunk
            else:
                yield str(response)

        except Exception as e:
            yield f"Error during live query: {str(e)}"

    def diff(
        self,
        commit_id_a: str,
        commit_id_b: str,
        server=None,
    ) -> dict:
        """
        Show what changed between two snapshots via knowledge diffing.

        Args:
            commit_id_a: First commit ID (or prefix)
            commit_id_b: Second commit ID (or prefix)
            server: LlamaServer instance (required for live diffing)

        Returns:
            Dict with structured diff of knowledge facets
        """
        if not server:
            raise ValueError("server parameter required for diffing")

        # Fetch commits and embeddings
        commit_a = self.store.get_commit_by_id_prefix(commit_id_a)
        commit_b = self.store.get_commit_by_id_prefix(commit_id_b)
        if not commit_a or not commit_b:
            return {"error": "One or both commits not found"}

        emb_a = self.store.get_snapshot_embedding(commit_a.id)
        emb_b = self.store.get_snapshot_embedding(commit_b.id)
        if not emb_a or not emb_b:
            return {"error": "One or both snapshots not indexed"}

        # Parse facets
        facets_a = KnowledgeFacets(**json.loads(emb_a.facets))
        facets_b = KnowledgeFacets(**json.loads(emb_b.facets))

        # Compute differences
        diff_result = {
            "commit_a": str(commit_a.id)[:8],
            "commit_b": str(commit_b.id)[:8],
            "task_a": commit_a.task,
            "task_b": commit_b.task,
            "new_functions": list(set(facets_b.functions) - set(facets_a.functions)),
            "removed_functions": list(set(facets_a.functions) - set(facets_b.functions)),
            "new_bugs": list(set(facets_b.bugs) - set(facets_a.bugs)),
            "fixed_bugs": list(set(facets_a.bugs) - set(facets_b.bugs)),
            "new_patterns": list(set(facets_b.patterns) - set(facets_a.patterns)),
            "new_facts": list(set(facets_b.facts) - set(facets_a.facts)),
        }

        return diff_result
