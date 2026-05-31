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
    embedding: list[float]


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
            self.index_data = index  # Store full index for knowledge access
            self.items = [CodeItem(**item) for item in index.get("items", [])]
            logger.info(f"Loaded index with {len(self.items)} items")
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            self.items = []
            self.index_data = {}

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

    def format_context(self, items: list[CodeItem], budget_chars: int = 2000, task: str = "") -> str:
        """
        Format retrieved items as context string.

        For system-level questions, includes knowledge summary.
        For code-specific questions, includes code snippets.

        Args:
            items: Code items to format
            budget_chars: Maximum total characters
            task: Original task string (to detect system vs. code questions)

        Returns:
            Formatted context string
        """
        if not items and not task:
            return ""

        parts = []

        # For system-level questions, prioritize knowledge summary
        if task and self.is_system_question(task):
            knowledge = self.get_knowledge_context()
            if knowledge:
                parts.append(knowledge)
                used = len(knowledge)

                # Add relevant code snippets if space allows
                if used < budget_chars * 0.7:
                    code_budget = budget_chars - used
                    code_parts = ["Relevant code:"]
                    for item in items[:3]:  # Limit to top 3 for system questions
                        snippet = f"\n- {item.name} ({item.location})"
                        if len(code_parts[-1]) + len(snippet) < code_budget:
                            code_parts.append(snippet)
                    if len(code_parts) > 1:
                        parts.append("\n".join(code_parts))

                return "\n".join(parts)

        # For code-specific questions, use existing logic
        if not items:
            return ""

        parts = ["Relevant code snippets:"]
        used = 0

        for item in items:
            header = f"\n- {item.type} {item.name} ({item.location})"
            sig = f"\n  Signature: {item.signature}"
            doc = f"\n  {item.docstring[:200]}" if item.docstring else ""

            chunk = header + sig + doc
            if used + len(chunk) > budget_chars:
                break

            parts.append(chunk)
            used += len(chunk)

        return "\n".join(parts)
