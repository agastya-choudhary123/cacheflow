# CacheFlow Benchmark Suite

Evaluate how well CacheFlow performs on standardized code benchmarks.

## What This Does

Runs benchmark datasets (like HumanEval, SWE-bench, GAIA) on **two backends**:

1. **Local Backend** (`qwen3:8b` via CacheFlow)
   - Runs model locally with KV-cache optimization
   - Measures wall-clock time, CPU time, tokens saved
   
2. **Cloud Backend** (`claude-opus-4-8` via Anthropic API)
   - Runs state-of-the-art model in the cloud
   - Baseline for comparison

Then reports **pass rates (%)** for each benchmark/backend combination.

## Usage

```bash
# Run HumanEval on both backends (local + cloud)
python benchmarks/harness.py --bench humaneval

# Run multiple benchmarks, limit to 5 tasks each for speed
python benchmarks/harness.py --bench humaneval swebench-lite gaia --max-tasks 5

# Run all benchmarks (slow!)
python benchmarks/harness.py --bench all

# Run only local backend
python benchmarks/harness.py --bench humaneval --backends local

# Run only cloud backend  
python benchmarks/harness.py --bench humaneval --backends cloud
```

## Benchmarks Available

- **humaneval** - Code generation (Python functions)
- **swebench-lite** - Real GitHub issues from open-source projects
- **swebench-verified** - Verified subset of SWE-bench
- **gaia** - General AI Assistant tasks (requires HF access)
- **agentbench** - Diverse agent environments
- **devbench** - Development tasks
- **featbench** - Feature implementation
- **prodcodebench** - Production code tasks
- ... and others

## Output

```
================================================================================
CacheFlow Benchmark Suite
================================================================================
Benchmarks: humaneval, swebench-lite
Max tasks: 5
Backends: both
================================================================================

[1/2] humaneval
  [local] Loading tasks... 5 tasks
  [local] Running... 1/5 2/5 3/5 4/5 5/5 ✓ 2/5 (40.0%)
  [cloud] Loading tasks... 5 tasks
  [cloud] Running... 1/5 2/5 3/5 4/5 5/5 ✓ 4/5 (80.0%)

[2/2] swebench-lite
  [local] Loading tasks... 3 tasks
  [local] Running... 1/3 2/3 3/3 ✓ 1/3 (33.3%)
  [cloud] Loading tasks... 3 tasks
  [cloud] Running... 1/3 2/3 3/3 ✓ 2/3 (66.7%)

================================================================================
RESULTS
================================================================================

| Benchmark         | Backend | Model              | Pass Rate |
|-------------------|---------|--------------------|-----------:|
| humaneval         | local   | qwen3:8b           |     40.0% |
| humaneval         | cloud   | claude-opus-4-8    |     80.0% |
| swebench-lite     | local   | qwen3:8b           |     33.3% |
| swebench-lite     | cloud   | claude-opus-4-8    |     66.7% |

Total time: 45.2s
```

## Key Metrics

Each run captures:
- **Pass Rate** - % of tasks solved correctly
- **Local Cache Performance** - Time saved from KV-cache restoration
- **Tokens Saved** - How many tokens CacheFlow avoided re-processing
- **FLOPS Avoided** - Compute saved vs re-priming from scratch

## Architecture

```
harness.py
├── Load benchmark adapter (HumanEval, SWE-bench, etc.)
├── Run on LOCAL backend
│   └── benchmarks/runners/local.py → CacheFlow agent → qwen3:8b
├── Run on CLOUD backend
│   └── benchmarks/runners/cloud.py → Claude API → claude-opus-4-8
├── Evaluate responses with adapter.evaluate()
└── Report pass rates
```

## Installation

```bash
# Required
pip install -e ".[dev]"
pip install datasets huggingface_hub

# Optional (for specific benchmarks)
pip install agentbench
```

## Notes

- GAIA is a gated dataset; request access at https://huggingface.co/datasets/gaia-benchmark/GAIA
- AgentBench requires `pip install agentbench` or manual setup
- Cloud runs require valid `ANTHROPIC_API_KEY`
- Local runs use the model specified in CacheFlow config
