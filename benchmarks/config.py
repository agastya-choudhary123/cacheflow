"""Benchmark configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_CORPUS = {
    "itsdangerous": {"url": "https://github.com/pallets/itsdangerous", "tier": "xsmall"},   # ~1.5K LOC
    "click":        {"url": "https://github.com/pallets/click",         "tier": "small"},    # ~8K LOC
    "requests":     {"url": "https://github.com/psf/requests",          "tier": "small-med"},# ~12K LOC
    "httpx":        {"url": "https://github.com/encode/httpx",          "tier": "medium"},   # ~18K LOC
    "pytest":       {"url": "https://github.com/pytest-dev/pytest",     "tier": "med-large"},# ~40K LOC
    "sqlalchemy":   {"url": "https://github.com/sqlalchemy/sqlalchemy", "tier": "large"},    # ~80K LOC
    "django":       {"url": "https://github.com/django/django",         "tier": "xlarge"},   # ~280K LOC
    "sympy":        {"url": "https://github.com/sympy/sympy",           "tier": "xxlarge"},  # ~400K LOC
}

ALL_WORKLOADS = ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8"]

# Primary benchmarks for Tier 2/3 runs
BENCHMARKS_PRIMARY = ["swebench-lite", "swebench-verified", "prodcodebench", "gaia", "agentbench", "featbench"]

# Secondary benchmarks for Tier 4 (optional)
BENCHMARKS_SECONDARY = ["repomod-bench", "octobench", "ml-bench", "repodebug"]

# Legacy single-turn / QA benchmarks
BENCHMARKS_LEGACY = ["humaneval", "bigcodebench", "repobench", "custom-qa", "custom-agentic-chain"]

# All benchmarks (default for harness)
ALL_BENCHMARKS = BENCHMARKS_PRIMARY + BENCHMARKS_SECONDARY + BENCHMARKS_LEGACY

LOCAL_MODEL = "qwen3:8b"
CLOUD_MODEL = "claude-opus-4-8"


@dataclass
class BenchConfig:
    repos: list[str]                          # subset of REPO_CORPUS keys
    workloads: list[str]                      # e.g. ["W1", "W3"]
    benchmarks: list[str]                     # e.g. ["humaneval", "swebench-lite"]
    backend: str                              # "local" | "cloud"
    output_dir: Path
    repos_dir: Path                           # where to clone repos
    concurrency: int = 1
    max_tasks: Optional[int] = None           # limit tasks per benchmark for quick runs
    ctx_size: int = 8192
    n_gpu_layers: int = -1
    max_steps: int = 20                       # agentic loop max steps
    max_tokens_per_step: int = 2048
    max_tokens: int = 1024                    # single-turn generation limit
    thinking_budget_tokens: int = 10000       # cloud only
    dry_run: bool = False                     # print plan without running
    log_file: Optional[Path] = None
