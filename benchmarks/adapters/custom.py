"""Custom task adapters for cross-cutting workload types.

custom-qa:           General Q&A about the repo (W1/W2/W3)
custom-same-task:    Repeat same task across N agents (W3 cloud cross-agent reuse)
custom-agentic-chain: Sequential agentic task chains (W6)
"""

from pathlib import Path
from typing import Iterator, Optional

from benchmarks.adapters.base import BenchmarkAdapter, BenchTask


QA_TASKS_TEMPLATE = [
    ("Where is the main entry point of this project defined?", "qa_entrypoint"),
    ("What database does this project use and where is it initialised?", "qa_database"),
    ("List the top-level modules in this package and their responsibilities.", "qa_modules"),
    ("What is the primary data structure used to track agent state?", "qa_data_structure"),
    ("How does this project handle errors or exceptions at the boundary?", "qa_error_handling"),
    ("Describe the concurrency model used by this project.", "qa_concurrency"),
    ("What external dependencies does this project have?", "qa_deps"),
    ("How are configuration options loaded?", "qa_config"),
    ("Describe the test strategy used in this project.", "qa_testing"),
    ("What is the most complex function in this codebase and what does it do?", "qa_complex"),
]

MULTITURN_CHAINS = [
    [
        "What is the main entry point of this project?",
        "What does that entry point call next?",
        "Drill into the most important function it calls — explain its logic.",
        "How would you refactor that function to be more testable?",
        "Summarise the changes you'd make in 3 bullet points.",
    ],
    [
        "Which module handles the most important business logic?",
        "What are the key data flows through that module?",
        "Are there any obvious error-handling gaps?",
        "Propose a fix for the most serious gap.",
        "Write a test that would catch that gap.",
    ],
    [
        "What external APIs or services does this project call?",
        "How are those calls authenticated?",
        "What happens if an external call fails?",
        "Propose an improvement to the error handling.",
        "How would you test that improvement?",
    ],
]

AGENTIC_CHAINS = [
    [
        ("Find the function with the most parameters and add a docstring to it.", "chain_docstring"),
        ("Add a type hint to every parameter of that same function.", "chain_typehints"),
        ("Write a unit test for that function.", "chain_test"),
        ("Run the test and fix any failures.", "chain_fix"),
        ("Add that test to the project's test suite.", "chain_add_test"),
    ],
    [
        ("Find the largest class and extract any method longer than 30 lines into a helper.", "chain_refactor"),
        ("Add a docstring to the extracted helper.", "chain_helper_doc"),
        ("Write a test for the extracted helper.", "chain_helper_test"),
        ("Commit the changes with a meaningful message.", "chain_commit"),
        ("Write a CHANGELOG entry describing the refactor.", "chain_changelog"),
    ],
]


class CustomQAAdapter(BenchmarkAdapter):
    def tasks(self, repo_path: Path, max_tasks: Optional[int] = None) -> Iterator[BenchTask]:
        for i, (text, tid) in enumerate(QA_TASKS_TEMPLATE):
            if max_tasks and i >= max_tasks:
                break
            yield BenchTask(task_id=f"{repo_path.name}_{tid}", task_text=text)


class CustomSameTaskAdapter(BenchmarkAdapter):
    """Yields the same task N times for cross-agent reuse testing (W3 cloud)."""

    def __init__(self, n_repeats: int = 4):
        self.n_repeats = n_repeats

    def tasks(self, repo_path: Path, max_tasks: Optional[int] = None) -> Iterator[BenchTask]:
        task = "Explain the core architecture of this codebase and identify the three most critical functions."
        n = max_tasks or self.n_repeats
        for i in range(n):
            yield BenchTask(
                task_id=f"{repo_path.name}_same_task_agent{i}",
                task_text=task,
            )


class CustomAgenticChainAdapter(BenchmarkAdapter):
    """Yields sequential agentic tasks for multi-turn chain testing (W6)."""

    def __init__(self, chain_index: int = 0):
        self.chain_index = chain_index % len(AGENTIC_CHAINS)

    def tasks(self, repo_path: Path, max_tasks: Optional[int] = None) -> Iterator[BenchTask]:
        chain = AGENTIC_CHAINS[self.chain_index]
        for i, (text, tid) in enumerate(chain):
            if max_tasks and i >= max_tasks:
                break
            yield BenchTask(
                task_id=f"{repo_path.name}_{tid}",
                task_text=text,
            )


def get_multiturn_chains(repo_key: str) -> list[list[str]]:
    """Return conversation chains for W2 workload."""
    return MULTITURN_CHAINS
