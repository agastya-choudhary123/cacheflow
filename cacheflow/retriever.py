"""RAG retrieval: find relevant code for a task."""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CodeItem:
    """Matches indexer.CodeItem structure."""
    type: str
    name: str
    signature: str
    docstring: str
    location: str
    body: str = ""
    calls: list[str] = None
    embedding: list[float] = None

    def __post_init__(self):
        if self.calls is None:
            self.calls = []
        if self.embedding is None:
            self.embedding = []


class CodeRetriever:
    """Retrieve relevant code items based on task semantic similarity."""

    def __init__(self, index_path: Path):
        """
        Load index from disk.

        Args:
            index_path: Path to index.json
        """
        self.index_path = index_path
        self.items: list[CodeItem] = []
        self.index_data: dict = {}
        self.embedding_model = None

        self._load_index()
        self._init_embedding_model()

    def _load_index(self) -> None:
        """Load index from JSON file."""
        if not self.index_path.exists():
            logger.warning(f"Index not found at {self.index_path}")
            return

        try:
            with open(self.index_path, "r") as f:
                index = json.load(f)
            self.index_data = index
            self.items = [CodeItem(**item) for item in index.get("items", [])]
            self._build_graph()
            logger.info(f"Loaded index with {len(self.items)} items")
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            self.items = []
            self.index_data = {}
            self._name_index: dict[str, list[CodeItem]] = {}
            self._called_by: dict[str, list[CodeItem]] = {}

    def _init_embedding_model(self) -> None:
        """Initialize embedding model for task encoding."""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Retrieval disabled. "
                "Run: pip install sentence-transformers"
            )
            self.embedding_model = None

    def _build_graph(self) -> None:
        """Build name→items and called_by indexes from the loaded items."""
        self._name_index: dict[str, list[CodeItem]] = {}
        self._called_by: dict[str, list[CodeItem]] = {}
        for item in self.items:
            self._name_index.setdefault(item.name, []).append(item)
            for called in item.calls:
                self._called_by.setdefault(called, []).append(item)

    def graph_expand(self, seeds: list[CodeItem], max_neighbors: int = 8) -> list[CodeItem]:
        """
        Expand a seed set via call graph edges (one hop).
        Returns neighbors not already in seeds, up to max_neighbors.
        Callees (what seeds call) + callers (what calls seeds) are both included.
        """
        seed_keys = {(i.location, i.name) for i in seeds}
        seen = set(seed_keys)
        neighbors: list[CodeItem] = []

        for item in seeds:
            # Callees: functions/classes this item calls
            for called_name in item.calls:
                for neighbor in self._name_index.get(called_name, []):
                    key = (neighbor.location, neighbor.name)
                    if key not in seen:
                        neighbors.append(neighbor)
                        seen.add(key)
            # Callers: functions that call this item
            for caller in self._called_by.get(item.name, []):
                key = (caller.location, caller.name)
                if key not in seen:
                    neighbors.append(caller)
                    seen.add(key)

        return neighbors[:max_neighbors]

    def retrieve(self, task: str, top_k: int = 5) -> list[CodeItem]:
        """
        Find top-K code items most relevant to the task.

        Args:
            task: Task string from user
            top_k: Number of items to return

        Returns:
            Top-K CodeItems sorted by relevance (highest first)
        """
        if not self.items or not self.embedding_model:
            return []

        try:
            # Embed task
            task_embedding = self.embedding_model.encode(
                task, normalize_embeddings=True
            ).tolist()

            # Compute similarity with all items
            scores = []
            for item in self.items:
                if not item.embedding:
                    similarity = 0.0
                else:
                    similarity = self._cosine_similarity(
                        task_embedding, item.embedding
                    )
                scores.append((item, similarity))

            # Sort by similarity, descending
            scores.sort(key=lambda x: x[1], reverse=True)

            # Return top-K
            return [item for item, _ in scores[:top_k]]

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

    def is_system_question(self, task: str) -> bool:
        """
        Detect if task is system-level (architecture, design, constraints)
        vs. code-specific (how do I, what function).

        Args:
            task: Task string

        Returns:
            True if system-level question
        """
        system_keywords = [
            "architecture", "design", "pattern", "how does", "why",
            "limitation", "constraint", "issue", "problem", "flow",
            "workflow", "process", "structure", "organization",
            "can i", "is it possible", "supported", "work together",
        ]
        task_lower = task.lower()
        return any(kw in task_lower for kw in system_keywords)

    def get_knowledge_context(self) -> str:
        """
        Extract stored knowledge summary from index.

        Returns:
            Formatted knowledge context string
        """
        if not hasattr(self, "index_data") or "knowledge" not in self.index_data:
            return ""

        knowledge = self.index_data.get("knowledge", {})
        if not knowledge:
            return ""

        parts = ["System knowledge:"]

        if knowledge.get("architecture"):
            parts.append(f"\nArchitecture: {knowledge['architecture']}")

        if knowledge.get("key_components"):
            parts.append("\nKey components:")
            for comp in knowledge["key_components"][:3]:
                parts.append(f"  - {comp.get('name')}: {comp.get('purpose')}")

        if knowledge.get("patterns"):
            parts.append("\nPatterns:")
            for pattern in knowledge["patterns"][:2]:
                parts.append(f"  - {pattern}")

        if knowledge.get("constraints"):
            parts.append("\nConstraints:")
            for constraint in knowledge["constraints"][:3]:
                parts.append(f"  - {constraint}")

        if knowledge.get("known_issues"):
            parts.append("\nKnown issues:")
            for issue in knowledge["known_issues"][:2]:
                parts.append(f"  - {issue}")

        return "\n".join(parts)

    def _cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec_a: First vector
            vec_b: Second vector

        Returns:
            Cosine similarity (0 to 1)
        """
        if len(vec_a) != len(vec_b):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        mag_a = math.sqrt(sum(a * a for a in vec_a))
        mag_b = math.sqrt(sum(b * b for b in vec_b))

        if mag_a == 0 or mag_b == 0:
            return 0.0

        return dot_product / (mag_a * mag_b)

    def generate_schema(self, items: list[CodeItem]) -> dict[str, str]:
        """
        Build a JSON answer schema keyed on the retrieved code items.
        Each key is snake_cased from the item name, value is an empty string
        the model must fill in. Forces the model to answer about specific
        retrieved code rather than generating generic text.
        """
        schema: dict[str, str] = {}
        seen: set[str] = set()
        for item in items[:5]:
            # snake_case the name and suffix with role hint
            key = item.name.lower()
            if item.type == "function":
                key = f"{key}_behavior"
            elif item.type == "class":
                key = f"{key}_role"
            # deduplicate
            base = key
            i = 2
            while key in seen:
                key = f"{base}_{i}"
                i += 1
            seen.add(key)
            schema[key] = ""
        schema["summary"] = ""
        return schema

    def format_context(
        self,
        items: list[CodeItem],
        neighbors: list[CodeItem] = None,
        budget_chars: int = 6000,
        task: str = "",
    ) -> str:
        """
        Format retrieved items + graph neighbors as context.
        Seeds (items) get full bodies; neighbors get signatures only.
        """
        if not items:
            return ""

        MAX_BODY = 1200  # chars per seed body to avoid blowing budget
        parts = ["Relevant code (retrieved by semantic search):"]
        used = len(parts[0])

        for item in items:
            header = f"\n\n### {item.type} `{item.name}` ({item.location})"
            body = item.body[:MAX_BODY] + ("..." if len(item.body) > MAX_BODY else "") if item.body else ""
            if body:
                chunk = header + f"\n```python\n{body}\n```"
            else:
                doc = f"  # {item.docstring[:120]}" if item.docstring else ""
                chunk = header + f"\n```python\n{item.signature}{doc}\n```"

            if used + len(chunk) > budget_chars:
                break
            parts.append(chunk)
            used += len(chunk)

        if neighbors:
            neighbor_header = "\n\nRelated code (via call graph):"
            if used + len(neighbor_header) < budget_chars:
                parts.append(neighbor_header)
                used += len(neighbor_header)
                for nb in neighbors:
                    doc = f"  # {nb.docstring[:80]}" if nb.docstring else ""
                    chunk = f"\n- `{nb.name}` ({nb.location}): `{nb.signature}`{doc}"
                    if used + len(chunk) > budget_chars:
                        break
                    parts.append(chunk)
                    used += len(chunk)

        return "\n".join(parts)
