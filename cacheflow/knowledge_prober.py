"""Knowledge probing protocol: extract structured facets from KV cache snapshots."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer

from cacheflow.store import CacheFlowStore, Commit, SessionLog


@dataclass
class KnowledgeFacets:
    """Structured knowledge extracted from a snapshot via probing."""

    functions: list[str]  # key functions/classes analyzed
    bugs: list[str]  # issues/risks identified
    patterns: list[str]  # architectural patterns observed
    facts: list[str]  # most important learned facts


class KnowledgeProber:
    """Probes live KV cache to extract knowledge facets at snapshot save time."""

    PROBES = [
        ("functions", "List the key functions and classes you analyzed in this session. Be specific."),
        ("bugs", "What bugs, issues, or risks did you identify? List them."),
        ("patterns", "What architectural patterns or design decisions did you observe?"),
        ("facts", "What are the 3 most important facts you learned in this session?"),
    ]

    def __init__(self, store: CacheFlowStore):
        """Initialize prober with a store reference."""
        self.store = store
        self._embedding_model = None

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy-load embedding model."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        return self._embedding_model

    def probe(
        self,
        server,
        slot_id: int,
        commit: Commit,
        session_log: SessionLog,
    ) -> Optional[KnowledgeFacets]:
        """
        Probe the live KV cache with targeted questions and extract knowledge facets.

        Args:
            server: LlamaServer instance with hot KV slot
            slot_id: slot ID containing the current KV state
            commit: Commit object with metadata
            session_log: SessionLog with prompt/response content

        Returns:
            KnowledgeFacets struct, or None if probing fails

        Side effects:
            Saves embedding to store via save_snapshot_embedding()
        """
        try:
            # Run each probe question against the live KV slot
            facets_dict = {}
            for facet_name, probe_question in self.PROBES:
                try:
                    response = server.completion(
                        prompt=probe_question,
                        slot_id=slot_id,
                        max_tokens=150,
                    )
                    facet_text = response.get("content", "").strip()
                    # Parse multi-line responses into a list
                    facets_dict[facet_name] = [
                        line.strip() for line in facet_text.split("\n") if line.strip()
                    ]
                except Exception:
                    # If a single probe fails, continue with empty list
                    facets_dict[facet_name] = []

            facets = KnowledgeFacets(**facets_dict)

            # Derive short summary from facets
            short_summary = self.derive_summary(facets)

            # Embed facets and summary
            facet_embeddings = self.embed_facets(facets, short_summary)

            # Serialize and store
            self.store.save_snapshot_embedding(
                commit_id=commit.id,
                agent_id=commit.agent_id,
                short_summary=short_summary,
                facets=json.dumps(asdict(facets)),
                embedding=json.dumps(facet_embeddings["summary"]),
                facet_embeddings=json.dumps(
                    {k: v for k, v in facet_embeddings.items() if k != "summary"}
                ),
            )

            return facets
        except Exception as e:
            # Non-blocking: probing failure never crashes the agent
            print(f"Warning: KnowledgeProber failed: {e}")
            return None

    def derive_summary(self, facets: KnowledgeFacets) -> str:
        """
        Derive a 2-3 sentence summary from knowledge facets (no model call).

        Args:
            facets: KnowledgeFacets with parsed probe responses

        Returns:
            Short natural-language summary
        """
        parts = []

        if facets.functions:
            funcs = ", ".join(facets.functions[:3])  # Top 3 functions
            parts.append(f"Analyzed: {funcs}.")

        if facets.patterns:
            patterns = ", ".join(facets.patterns[:2])  # Top 2 patterns
            parts.append(f"Observed patterns: {patterns}.")

        if facets.facts:
            key_fact = facets.facts[0]  # Most important fact
            parts.append(f"Key learning: {key_fact}")

        if facets.bugs:
            bugs = ", ".join(facets.bugs[:2])  # Top 2 issues
            parts.append(f"Identified issues: {bugs}.")

        summary = " ".join(parts)
        # Truncate to ~300 chars for storage efficiency
        if len(summary) > 300:
            summary = summary[:297] + "..."

        return summary if summary else "Snapshot analyzed."

    def embed_facets(
        self, facets: KnowledgeFacets, short_summary: str
    ) -> dict[str, list[float]]:
        """
        Embed knowledge facets and summary for semantic search.

        Args:
            facets: KnowledgeFacets struct
            short_summary: Derived summary string

        Returns:
            Dict mapping facet names (and "summary") to 384-dim embeddings
        """
        embeddings = {}

        # Embed summary
        summary_embedding = self.embedding_model.encode(
            short_summary, normalize_embeddings=True
        )
        embeddings["summary"] = summary_embedding.tolist()

        # Embed each facet as a single string
        for facet_name in ["functions", "bugs", "patterns", "facts"]:
            facet_list = getattr(facets, facet_name)
            if facet_list:
                facet_text = " ".join(facet_list)
                facet_embedding = self.embedding_model.encode(
                    facet_text, normalize_embeddings=True
                )
                embeddings[facet_name] = facet_embedding.tolist()

        return embeddings
