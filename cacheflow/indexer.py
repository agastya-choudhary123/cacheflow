"""Semantic indexing: extract code structure and compute embeddings."""

import ast
import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional
import re
import subprocess

logger = logging.getLogger(__name__)


@dataclass
class CodeItem:
    """A code unit: function, class, module, or pattern."""
    type: str  # "function", "class", "module", "pattern"
    name: str
    signature: str  # e.g., "def foo(x: int) -> str:"
    docstring: str
    location: str  # "path.py:line"
    body: str = ""  # full source of the function/class
    calls: list[str] = field(default_factory=list)  # names called/referenced by this item
    embedding: list[float] = field(default_factory=list)


class CodeIndexer:
    """Extract code structure and compute embeddings."""

    def __init__(self):
        """Initialize indexer with local embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
            self.embedding_model = None

    def extract_from_codebase(self, base_path: Path) -> list[CodeItem]:
        """
        Walk codebase and extract function/class/pattern metadata.

        Args:
            base_path: Project root

        Returns:
            List of CodeItems without embeddings
        """
        items = []

        # Get all tracked files via git ls-files
        try:
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=base_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                files = [
                    base_path / line
                    for line in result.stdout.splitlines()
                    if line and (base_path / line).is_file()
                ]
            else:
                files = list(base_path.rglob("*.py"))
        except Exception:
            files = list(base_path.rglob("*.py"))

        for fpath in files:
            if fpath.suffix == ".py":
                items.extend(self._extract_from_python(fpath, base_path))

        return items

    def _extract_from_python(self, fpath: Path, base_path: Path) -> list[CodeItem]:
        """Extract functions and classes from a Python file."""
        items = []

        try:
            content = fpath.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(content)
        except SyntaxError:
            logger.debug(f"Syntax error in {fpath}, skipping")
            return items

        try:
            rel_path = fpath.relative_to(base_path)
        except ValueError:
            rel_path = fpath

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                sig = self._function_signature(node)
                docstring = ast.get_docstring(node) or ""
                body = ast.get_source_segment(content, node) or ""
                calls = self._extract_calls(node)
                items.append(
                    CodeItem(
                        type="function",
                        name=node.name,
                        signature=sig,
                        docstring=docstring,
                        location=f"{rel_path}:{node.lineno}",
                        body=body,
                        calls=calls,
                    )
                )
            elif isinstance(node, ast.ClassDef):
                docstring = ast.get_docstring(node) or ""
                body = ast.get_source_segment(content, node) or ""
                calls = self._extract_calls(node)
                items.append(
                    CodeItem(
                        type="class",
                        name=node.name,
                        signature=f"class {node.name}:",
                        docstring=docstring,
                        location=f"{rel_path}:{node.lineno}",
                        body=body,
                        calls=calls,
                    )
                )

        return items

    def _extract_calls(self, node) -> list[str]:
        """Extract names of functions/classes called or inherited by this AST node."""
        calls: set[str] = set()
        # Capture base classes for ClassDef
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name):
                    calls.add(base.id)
                elif isinstance(base, ast.Attribute):
                    calls.add(base.attr)
        # Walk body for all Call expressions
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.add(child.func.attr)
        calls.discard(getattr(node, "name", None))  # remove self-reference
        return sorted(calls)

    def _function_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature from AST node."""
        args = []
        for arg in node.args.args:
            args.append(arg.arg)

        sig = f"def {node.name}({', '.join(args)}):"
        return sig

    def embed_items(self, items: list[CodeItem]) -> list[CodeItem]:
        """
        Compute embeddings locally for each item.

        Args:
            items: Code items without embeddings

        Returns:
            Same items with .embedding populated
        """
        if not self.embedding_model:
            logger.warning("Embedding model not available, skipping embeddings")
            return items

        texts = [f"{item.name} {item.signature} {item.docstring}" for item in items]
        try:
            embeddings = self.embedding_model.encode(
                texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False
            )
            for item, emb in zip(items, embeddings):
                item.embedding = emb.tolist()
        except Exception as e:
            logger.warning(f"Batch embedding failed: {e}")
            for item in items:
                item.embedding = []

        return items

    def save_index(
        self, items: list[CodeItem], output_path: Path
    ) -> None:
        """
        Save index to JSON file.

        Args:
            items: Code items with embeddings
            output_path: Path to save index.json
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        index = {
            "version": 1,
            "indexed_at": str(Path(__file__).stat().st_mtime),
            "items": [asdict(item) for item in items],
        }

        with open(output_path, "w") as f:
            json.dump(index, f, indent=2)

        logger.info(f"Saved index with {len(items)} items to {output_path}")

    def consolidate_knowledge(self, consolidation_text: str) -> dict:
        """
        Extract structured knowledge from consolidation text.

        Simple extraction: look for bulleted patterns in the model output.

        Args:
            consolidation_text: Model's dense knowledge snapshot

        Returns:
            {
                "architecture": "...",
                "key_apis": [{"name": "...", "purpose": "..."}],
                "patterns": [...],
                "constraints": [...],
            }
        """
        knowledge = {
            "architecture": "",
            "key_apis": [],
            "patterns": [],
            "constraints": [],
        }

        lines = consolidation_text.split("\n")

        current_section = None
        for line in lines:
            line_lower = line.lower().strip()

            # Detect section headers
            if "architecture" in line_lower and ("#" in line or ":" in line):
                current_section = "architecture"
                continue
            elif ("api" in line_lower or "function" in line_lower) and ("#" in line or ":" in line):
                current_section = "key_apis"
                continue
            elif "pattern" in line_lower and ("#" in line or ":" in line):
                current_section = "patterns"
                continue
            elif ("constraint" in line_lower or "requirement" in line_lower) and ("#" in line or ":" in line):
                current_section = "constraints"
                continue

            # Process bullet points
            if line.strip() and (line.lstrip().startswith("-") or line.lstrip().startswith("*")):
                content = line.lstrip().lstrip("-* ").strip()
                if current_section == "architecture":
                    knowledge["architecture"] += content + " "
                elif current_section == "key_apis":
                    # Try to parse "name: description" format
                    if ":" in content:
                        name, desc = content.split(":", 1)
                        knowledge["key_apis"].append(
                            {"name": name.strip(), "purpose": desc.strip()}
                        )
                    else:
                        knowledge["key_apis"].append(
                            {"name": content, "purpose": ""}
                        )
                elif current_section == "patterns":
                    knowledge["patterns"].append(content)
                elif current_section == "constraints":
                    knowledge["constraints"].append(content)

        knowledge["architecture"] = knowledge["architecture"].strip()
        return knowledge

    def load_index(self, index_path: Path) -> Optional[dict]:
        """
        Load index from JSON file.

        Args:
            index_path: Path to index.json

        Returns:
            Index dict or None if not found
        """
        if not index_path.exists():
            return None

        try:
            with open(index_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load index: {e}")
            return None
