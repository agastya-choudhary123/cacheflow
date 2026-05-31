# CacheFlow — MVP Implementation Guide for Claude Code

A step-by-step build guide structured as Claude Code sessions.
Each session has a clear goal, exact prompts to paste, and a done condition.
Do not start a new session until the previous one passes its done condition.

---

## What You're Building

A CLI runtime called `cacheflow` that wraps llama.cpp's slot save/restore API
and adds git-style versioning on top. Agents accumulate persistent KV cache
state across sessions instead of starting from zero each time.

**The demo in one sentence:** Run a coding agent 5 times against your repo.
Watch the token cost drop from ~80K to ~3K by session 3. Show the commit graph.

---

## Prerequisites (Do These Before Claude Code)

### Part 1: Install Dependencies

```bash
# 1. Install Ollama
brew install ollama
ollama pull llama3.1:8b

# 2. Install llama.cpp (needed for slot save/restore — Ollama doesn't expose this)
brew install llama.cpp

# 3. Install uv (fast Python package manager)
brew install uv

# 4. Create project
mkdir cacheflow && cd cacheflow
git init
uv init
```

### Part 2: CRITICAL — Pre-Flight API Validation (REQUIRED)

**Do NOT start Claude Code until this passes.** This script validates that your llama.cpp installation is compatible with cacheflow.

Create `scripts/validate_llama_api.py` and run it:

```bash
mkdir -p scripts
curl -o scripts/validate_llama_api.py https://raw.githubusercontent.com/ggerganov/llama.cpp/master/examples/server_validation.py
# OR manually create the script below:
```

**What the script does:**
- Verifies llama-server binary exists and is executable
- Starts llama-server with your actual model and `--slots` enabled
- Tests `/completion`, `/slots/0/save`, `/slots/0/restore`, `/slots` endpoints
- **CRITICAL:** Validates ctx_size is immutable (tests that changing ctx_size breaks restore)
- Prints the exact response JSON schema from each endpoint
- Saves response schema to `.cacheflow/api_schema.json` for reference

**Run it:**
```bash
python3 scripts/validate_llama_api.py
```

**Expected output:**
```
✓ llama-server started
✓ Completion endpoint works
  - Response includes usage: {'prompt_tokens': X, 'completion_tokens': Y}
✓ Slot save endpoint works
✓ Slot list endpoint works
✓ Slot restore endpoint exists
✓ ctx_size is locked and immutable

All prerequisite checks passed!
```

**If it fails:** Do not proceed. Debug with llama.cpp docs at https://github.com/ggerganov/llama.cpp

**If it passes:** You are clear to start Session 1. The `.cacheflow/api_schema.json` file contains the real response format — Session 3 tests will use this to build accurate mocks.

---

## Project Structure (What You're Building Toward)

```
cacheflow/
├── cacheflow/
│   ├── __init__.py
│   ├── cli.py              # Click CLI: run, log, fork, diff, merge
│   ├── server.py           # llama-server process manager
│   ├── store.py            # SQLite DAG + KV cache file management
│   ├── agent.py            # Agent loop: prompt → tool calls → response
│   ├── compressor.py       # Background idle consolidation
│   └── config.py           # Model config, paths, defaults
├── tests/
│   ├── test_store.py
│   └── test_agent.py
├── pyproject.toml
├── README.md
└── .cacheflow/              # Created at runtime per project
    ├── config.json         # Model hash, quantization, context length
    ├── snapshots/          # KV cache .bin files, named by hash
    └── dag.db              # SQLite: commits, branches, metadata
```

---

## Session 1 — Project Scaffold + llama-server Wrapper

**Goal:** llama-server starts, serves a model, and you can make a basic
completion request from Python. Nothing else.

**Time estimate:** 1-2 hours with Claude Code.

### Claude Code Prompt 1.1 — Scaffold

```
Create a Python project called cacheflow using uv.

Project structure:
cacheflow/
├── cacheflow/
│   ├── __init__.py
│   ├── cli.py
│   ├── server.py
│   ├── store.py
│   ├── agent.py
│   ├── compressor.py
│   └── config.py
├── tests/
│   ├── __init__.py
│   ├── test_store.py
│   └── test_agent.py
└── pyproject.toml

Dependencies to add to pyproject.toml:
- click >= 8.0
- httpx >= 0.27
- sqlalchemy >= 2.0
- rich >= 13.0
- pydantic >= 2.0
- pytest >= 8.0
- pytest-asyncio

Create empty files with docstring stubs only. Do not implement anything yet.
```

### Claude Code Prompt 1.2 — llama-server Manager

```
Implement cacheflow/server.py.

This module manages a llama-server subprocess. llama-server is a binary
installed via `brew install llama.cpp`.

CRITICAL: ctx_size is immutable. Once set, you cannot change it between
sessions without breaking all existing snapshots. Add validation on startup.

Requirements:
1. Class LlamaServer with methods:
   - start(model_path: str, slot_save_path: str, ctx_size: int = 8192,
           n_gpu_layers: int = 99) -> None
     Starts llama-server as a subprocess with these exact flags:
     --model {model_path}
     --ctx-size {ctx_size}
     --n-gpu-layers {n_gpu_layers}
     --slots
     --slot-save-path {slot_save_path}
     --port 8080
     --host 127.0.0.1
     Waits until the server is ready by polling GET /health every 500ms.
     Raises TimeoutError after 30 seconds.

   - stop() -> None
     Terminates the subprocess cleanly. Handles already-dead process.

   - is_running() -> bool
     Returns True if subprocess is alive and /health returns 200.

   - completion(prompt: str, slot_id: int = 0,
                max_tokens: int = 512) -> dict
     POST to /completion with:
     {"prompt": prompt, "slot_id": slot_id, "n_predict": max_tokens,
      "cache_prompt": true}
     Returns the full response dict. This response includes token counts
     in response["usage"]["prompt_tokens"] and ["completion_tokens"].

   - save_slot(slot_id: int = 0) -> dict
     POST to /slots/{slot_id}/save
     Waits for 200 response.
     Returns dict with "save_time_ms" (time taken to save).

   - restore_slot(filename: str, slot_id: int = 0) -> dict
     POST to /slots/{slot_id}/restore with {"filename": filename}
     Waits for 200 response.
     Returns dict with "restore_time_ms" (time taken to restore).

   - erase_slot(slot_id: int = 0) -> None
     POST to /slots/{slot_id}/erase

2. Use httpx for all HTTP calls (not requests).
3. Log all subprocess stdout/stderr to a file: .cacheflow/server.log
4. Handle port-in-use error gracefully — if port 8080 is taken,
   try 8081, 8082, up to 8090.
5. Add timing: measure save_slot() and restore_slot() durations.
   Return timing data so the caller can log it.

Find the llama-server binary path using `shutil.which("llama-server")`.
Raise FileNotFoundError with a helpful message if not found.
```

### Claude Code Prompt 1.3 — Smoke Test

```
Write a standalone test script tests/test_server_smoke.py that:

1. Finds the smallest available GGUF model on disk by searching:
   - ~/.ollama/models/blobs/ for any .gguf files
   - ~/Library/Caches/llama.cpp/ for any .gguf files
   - The current directory for any .gguf files
   If no GGUF found, print a helpful message and exit.

2. Starts LlamaServer with that model and ctx_size=2048.

3. Sends the completion: "Say hello in exactly 3 words."

4. Prints the response text and token counts from response["usage"].

5. Tests save_slot() and restore_slot(), printing save_time_ms and restore_time_ms.

6. Stops the server.

7. Prints "PASS" if response is non-empty and timings are recorded.

8. Note: If restore_time_ms > 10 seconds, this is a signal that snapshot
   I/O may become a bottleneck. Consider reducing ctx_size before Session 2.

Run with: uv run python tests/test_server_smoke.py
```

**Done condition:** `uv run python tests/test_server_smoke.py` prints a
3-word response, token counts, save/restore timings, and PASS. If it fails, 
fix server.py before proceeding. Review the timings — if restore is >10s, 
note this for later optimization.

---

## Session 2 — SQLite DAG Store

**Goal:** A database layer that stores commits, tracks agent lineage,
and manages KV cache file references.

**Time estimate:** 2-3 hours.

### Claude Code Prompt 2.1 — Schema + Store

```
Implement cacheflow/store.py using SQLAlchemy with SQLite.

Database location: .cacheflow/dag.db (relative to where `cf init` is run).

Schema — define these SQLAlchemy models:

1. Agent
   - id: UUID primary key
   - name: str, unique, not null         # e.g. "main", "test-agent"
   - created_at: datetime
   - model_hash: str                     # sha256 of model file
   - model_name: str                     # e.g. "llama3.1:8b"
   - ctx_size: int
   - head_commit_id: UUID nullable FK → Commit.id  # current HEAD

2. Commit
   - id: UUID primary key (content-addressed: sha256 of snapshot file)
   - agent_id: UUID FK → Agent.id
   - parent_id: UUID nullable FK → Commit.id       # null for first commit
   - forked_from_id: UUID nullable FK → Commit.id  # set when forked
   - snapshot_path: str                 # relative path to .bin file
   - snapshot_size_bytes: int
   - task: str                          # what the agent did this session
   - tokens_this_session: int           # tokens consumed this session
   - tokens_saved: int                  # vs naive re-ingestion baseline
   - llama_cpp_version: str             # e.g. "1.2.3" (for migration checks)
   - snapshot_save_time_ms: int         # how long save_slot took
   - snapshot_restore_time_ms: int      # how long restore_slot took
   - created_at: datetime

3. Session
   - id: UUID primary key
   - agent_id: UUID FK → Agent.id
   - commit_id: UUID nullable FK → Commit.id  # null until committed
   - prompt: str
   - response: str
   - tokens_in: int
   - tokens_out: int
   - duration_ms: int
   - created_at: datetime

Class AgentGitStore with methods:

init_db(base_path: Path) -> None
  Creates .cacheflow/ directory and initializes all tables.

create_agent(name: str, model_name: str, model_hash: str,
             ctx_size: int) -> Agent

get_agent(name: str) -> Agent | None

list_agents() -> list[Agent]

create_commit(agent: Agent, snapshot_path: str, task: str,
              tokens_this_session: int, tokens_saved: int,
              parent_id: UUID | None = None,
              forked_from_id: UUID | None = None) -> Commit
  Computes commit ID as sha256(snapshot file contents).
  Updates agent.head_commit_id to the new commit.

get_commit(commit_id: UUID) -> Commit | None

get_commit_history(agent: Agent) -> list[Commit]
  Returns commits from HEAD back to root, oldest last.

log_session(agent: Agent, commit: Commit, prompt: str, response: str,
            tokens_in: int, tokens_out: int, duration_ms: int) -> Session
```

### Claude Code Prompt 2.2 — Store Tests

```
Write pytest tests in tests/test_store.py covering:

1. test_init_db: creates .cacheflow/dag.db in a temp directory
2. test_create_agent: creates an agent, retrieves by name
3. test_create_commit: creates agent, creates commit with fake snapshot file,
   verifies commit ID is sha256 of file contents
4. test_commit_history: creates 3 sequential commits, verifies
   get_commit_history returns them in correct order
5. test_fork_tracking: creates a commit, creates a fork commit with
   forked_from_id set, verifies the relationship is stored

Use pytest tmp_path fixture for temp directories.
Use a small fake binary file (os.urandom(1024)) as the fake snapshot.

Run with: uv run pytest tests/test_store.py -v
```

**Done condition:** All 5 store tests pass.

---

## Session 3 — The Agent Loop

**Goal:** An agent that loads its previous KV cache if it exists, runs a
task, saves the updated cache, and commits to the DAG. This is the core loop.

**Time estimate:** 3-4 hours.

### Claude Code Prompt 3.1 — Config

```
Implement cacheflow/config.py.

Class AgentGitConfig (Pydantic BaseModel):
  base_path: Path                 # project root (.cacheflow lives here)
  model_path: str                 # path to GGUF file
  model_name: str                 # human name e.g. "llama3.1:8b"
  model_hash: str                 # sha256 of model file (computed on init)
  ctx_size: int = 8192
  n_gpu_layers: int = 99          # -1 = CPU only, 99 = all GPU layers
  slot_save_path: Path            # .cacheflow/snapshots/

Functions:
  compute_model_hash(model_path: str) -> str
    sha256 of first 10MB of model file (fast approximation)

  load_config(base_path: Path) -> AgentGitConfig
    Reads .cacheflow/config.json. Raises FileNotFoundError if not initialized.

  save_config(config: AgentGitConfig) -> None
    Writes to .cacheflow/config.json.

  find_gguf_for_model(model_name: str) -> str | None
    Searches common paths for a GGUF matching the model name:
    - ~/.ollama/models/blobs/
    - ~/Library/Caches/llama.cpp/
    - ~/.cache/lm-studio/models/
    Returns the path string or None if not found.
```

### Claude Code Prompt 3.2 — Agent Loop

```
Implement cacheflow/agent.py.

This is the core loop. It:
1. Starts llama-server (or connects to running instance)
2. Checks if agent has a previous snapshot and restores it
3. Runs the task (sends prompt, gets response)
4. Saves the updated slot
5. Creates a commit in the DAG
6. Returns results

Class AgentSession:
  def __init__(self, agent_name: str, base_path: Path):
    Loads config and store. Does not start server yet.

  def run(self, task: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT,
          max_tokens: int = 1024) -> SessionResult:

    Full flow:
    a. Load agent from store (create if first run)
    b. Acquire file lock: .cacheflow/.cacheflow.lock (prevents concurrent runs)
    c. Start LlamaServer
    d. If agent has a head commit:
         - Restore slot from head_commit.snapshot_path (record restore_time_ms)
         - baseline_tokens = ctx_size (what naive re-ingestion would cost)
       Else:
         - baseline_tokens = 0 (first session, no savings yet)
    e. Build prompt:
         If first session: system_prompt + "\n\nTask: " + task
         If has history: "Task: " + task (system prompt is in the cache)
    f. Send completion, record tokens_in and tokens_out from response["usage"]
       (or response["tokens_evaluated"] depending on llama.cpp version).
       See .cacheflow/api_schema.json for the exact response format.
    g. Save slot to .cacheflow/snapshots/.tmp_{uuid}.bin (record save_time_ms)
    h. In a single DB transaction:
         - Create commit in store with snapshot_path = .tmp_{uuid}.bin
         - Commit ID = sha256(file contents)
         - Record: tokens_this_session, tokens_saved, save_time_ms, restore_time_ms
    i. Only AFTER transaction succeeds: rename .tmp_{uuid}.bin to {commit_id}.bin
    j. Update commit record with final snapshot_path
    k. Log session
    l. Release file lock
    m. Stop server
    n. Return SessionResult

Dataclass SessionResult:
  agent_name: str
  commit_id: str
  task: str
  response: str
  tokens_this_session: int
  tokens_saved: int
  snapshot_size_bytes: int
  duration_ms: int
  is_first_session: bool

DEFAULT_SYSTEM_PROMPT = """You are an expert software engineer with deep
knowledge of the codebase you've been given access to. You help with
coding tasks efficiently and precisely. When you complete a task, briefly
summarize what you did and what you learned about the codebase."""
```

### Claude Code Prompt 3.3 — Fork Operation

```
Add to cacheflow/agent.py:

Function fork_agent(parent_name: str, child_name: str,
                    scope: str, base_path: Path) -> Agent:

  Fork creates a new agent that starts from the parent's current HEAD snapshot.

  Steps:
  1. Load parent agent from store. Raise ValueError if no HEAD commit.
  2. Create child agent in store with same model config as parent.
  3. Copy parent's HEAD snapshot file to a new file:
     .cacheflow/snapshots/fork_{child_name}_{parent_commit_id[:8]}.bin
  4. Create an initial commit for child agent:
     - snapshot_path = copied file path
     - task = f"Forked from {parent_name} at {parent_commit_id[:8]}"
     - tokens_this_session = 0
     - tokens_saved = 0
     - forked_from_id = parent's HEAD commit id
     - parent_id = None (child's lineage starts fresh)
  5. Return child agent.

  Key: the child gets a COPY of the snapshot, not a reference.
  Copy-on-write optimization (reference + lazy copy) is a future optimization.
  For MVP, physical copy is correct and simple.
```

### Claude Code Prompt 3.4 — Integration Tests (Mock-Based)

```
Write pytest tests in tests/test_agent_integration.py.

CRITICAL: The mock server must match the REAL response schema from
.cacheflow/api_schema.json (generated by the pre-flight validation script).

Do not guess the response format. Load the actual schema and mock to match it.

Tests to include:

1. test_token_savings_across_sessions()
   - Create temp project with .cacheflow/
   - Run agent.run() with task "describe this file" twice
   - Assert tokens_this_session[1] < tokens_this_session[0]
   - Assert tokens_saved[1] > 0

2. test_snapshot_save_restore_flow()
   - Mock the server to return correct response format
   - Run agent once, verify snapshot file exists
   - Load agent again, verify restore succeeds
   - Run agent again with different task, verify still uses cache

3. test_atomic_snapshot_commit()
   - Verify that rename happens AFTER DB transaction
   - Simulate DB failure, verify tmpfile is cleaned up
   - Simulate rename failure, verify recovery is possible

Mock the server using the actual response schema from api_schema.json.
Example:

    import json
    with open(".cacheflow/api_schema.json") as f:
        api_schema = json.load(f)
    
    class MockLlamaServer:
        def completion(self, prompt, slot_id=0, max_tokens=512):
            return {
                "content": "mock response text",
                **api_schema["completion"]  # Use real schema fields
            }
```

**Done condition:** Write a manual test script `tests/test_agent_manual.py`
that runs AgentSession.run() twice with the same agent on a simple task.
Print tokens_this_session for both runs. Session 2 should use significantly
fewer tokens than session 1.

Run: `uv run pytest tests/test_agent_integration.py -v`
All tests pass. The manual test shows token reduction.

---

## Session 4 — The CLI

**Goal:** `cacheflow` works as a command-line tool with `init`, `run`,
`log`, `fork`, `diff`, and `status` commands. Dashboard is optional.

**Merge is v2**, not MVP. Don't implement it yet — it depends on 
consolidation being proven in Session 5.

**Time estimate:** 2-3 hours (3-4 if you add dashboard).

### Claude Code Prompt 4.1 — CLI Commands

```
Implement cacheflow/cli.py using Click.

Install entry point in pyproject.toml:
[project.scripts]
cacheflow = "cacheflow.cli:cli"

Commands:

1. cf init [--model MODEL_NAME] [--model-path PATH] [--ctx-size CTX_SIZE]
   Initialize cacheflow in the current directory.
   - Creates .cacheflow/ directory
   - If --model-path provided: use that directly (user-supplied path)
   - Else if --model provided: search for it (default: "llama3.1:8b")
     Handle Ollama symlinks, validate file exists and is >100MB
   - If model not found: error with searched paths and suggestion to run
     "ollama pull {model_name}"
   - Computes model hash (sha256 of first 10MB)
   - Saves config.json with: model_path, model_name, model_hash, ctx_size, llama_cpp_version
   - Initializes database
   - Prints: "Initialized cacheflow with model {model_name}"
   - Prints: "Model hash: {hash[:12]}..."
   - Prints: "Context size: {ctx_size} tokens"
   - CRITICAL: ctx_size is now locked and immutable. To change it, delete
     .cacheflow/ and reinit.

2. cf run [--agent AGENT_NAME] [--max-tokens MAX] TASK
   Run a task with an agent (default agent name: "main").
   Uses AgentSession.run().
   Output format (use Rich):

   ┌─ cf run ──────────────────────────────────────┐
   │ Agent: main                                          │
   │ Task: <first 60 chars of task>                      │
   └──────────────────────────────────────────────────────┘

   [spinner] Running...

   ╭─ Response ───────────────────────────────────────────╮
   │ <response text>                                      │
   ╰──────────────────────────────────────────────────────╯

   ╭─ Commit ─────────────────────────────────────────────╮
   │ Hash:          a3f9b2c1                              │
   │ Tokens used:   4,200                                │
   │ Tokens saved:  47,800  (91.9% reduction)            │
   │ Snapshot size: 847 MB                               │
   ╰──────────────────────────────────────────────────────╯

3. cf log [--agent AGENT_NAME] [--limit N]
   Show commit history for an agent.
   Output format (use Rich Table):

   COMMIT    TASK                          TOKENS   SAVED    DATE
   ───────────────────────────────────────────────────────────────
   a3f9b2c   refactor auth module          4,200    47,800   2h ago
   def456a   add rate limiting             3,800    48,200   1d ago
   abc123f   initial codebase analysis    52,000        0   2d ago

   Show savings as a percentage bar on the right:
   [████████████░░░] 91.9%

4. cf fork PARENT_AGENT CHILD_AGENT [--scope SCOPE_DESCRIPTION]
   Fork an agent.
   Output:
   Forked 'main' → 'test-agent'
   Child agent starts from commit a3f9b2c1
   Snapshot copied: 847 MB

5. cf diff COMMIT_A COMMIT_B [--agent AGENT_NAME]
   Show semantic diff between two commits.
   Asks the model: "Compare what the agent knew at commit A vs commit B.
   What was added, what changed, what was invalidated?"
   Uses a SEPARATE small completion (does not restore either snapshot —
   uses the stored task descriptions and session logs as context).
   Output:
   ╭─ Diff: a3f9b2c → def456a ───────────────────────────╮
   │ + Learned: auth uses JWT with 24h expiry             │
   │ + Learned: rate limiter is in middleware/rate.py     │
   │ ~ Updated: understanding of session flow             │
   ╰──────────────────────────────────────────────────────╯

6. cf status
   Show current state:
   - Active agent and HEAD commit
   - Total snapshots stored + total disk usage
   - Model loaded
   - Savings summary across all sessions
```

### Claude Code Prompt 4.2 — Rich Dashboard (OPTIONAL)

```
OPTIONAL: If you have time after Session 4.1, add this command.
If not, skip it. The demo script in Session 6 doesn't require it.

cf dashboard (optional)

Shows a live Rich Layout with:
- Left panel: commit DAG as ASCII art tree
  main
  ├── a3f9b2c  "refactor auth"    [91% saved]
  ├── def456a  "add rate limit"   [92% saved]
  │   └── (fork) test-agent
  │       └── ghi789b  "fix tests"  [89% saved]
  └── abc123f  "initial analysis" [first run]

- Right panel: Token savings chart (use Rich's built-in bar rendering)
  Session 1: ████████████████████████████████ 52,000 tokens
  Session 2: ████                              4,200 tokens
  Session 3: ███                               3,800 tokens
  Session 4: ███                               3,600 tokens

- Bottom panel: Stats summary
  Total sessions: 4  |  Total saved: ~144,000 tokens  |
  Agents: 2

Update every 2 seconds using Rich Live.

Implementation: Write tests for DAG rendering with fork scenarios.
```

**Done condition:**
```bash
uv run cf init
uv run cf run "Analyze this Python project and tell me its structure"
uv run cf run "What are the main dependencies?"
uv run cf log
uv run cf dashboard
```
All commands work. The log shows token reduction from session 1 to session 2.

---

## Session 5 — Background Consolidation (Idle Compaction)

**Goal:** When an agent finishes a session, a background process runs
consolidation to prevent context window overflow.

**Time estimate:** 2-3 hours.

### Claude Code Prompt 5.1 — Compressor

```
Implement cacheflow/compressor.py.

This module runs consolidation on agent snapshots when they exceed a
threshold. It's inspired by how the OS pages out unused memory and how
humans consolidate memory during sleep.

The mechanism: if an agent's current context (tokens_accumulated) exceeds
70% of ctx_size, run a consolidation session that asks the model to produce
a dense summary of everything it knows, then restores the agent to a fresh
context with only that summary as its knowledge base.

Class Compressor:
  def __init__(self, store: AgentGitStore, config: AgentGitConfig)

  def needs_compaction(self, agent: Agent) -> bool:
    Returns True if sum of tokens_this_session across all commits for this
    agent exceeds 0.7 * config.ctx_size.

  def compact(self, agent: Agent) -> Commit | None:
    If not needs_compaction, return None.

    Steps:
    a. Start LlamaServer
    b. Restore agent's HEAD snapshot
    c. Send this consolidation prompt:
       "You have accumulated knowledge about this codebase across multiple
        sessions. Produce a DENSE KNOWLEDGE SNAPSHOT that will become your
        entire memory.
        
        Format: Structured bullet points. For each point, be specific:
        - File paths (relative to repo root)
        - Function/class names and what they do
        - Key data structures and their layouts
        - Critical algorithms or patterns
        - Important gotchas or limitations
        - Dependencies and how they're used
        
        Example:
        * auth/middleware.py
          - AuthMiddleware: checks JWT in Authorization header
          - Token format: "Bearer {jwt}". Tokens expire after 24h.
          - Gotcha: refresh_token endpoint is broken (issue #42)
        
        Only include information critical for coding tasks. Be exhaustive:
        this must cover everything needed to solve future tasks."
    d. Save the response as a string: consolidation_text
    e. Erase the current slot
    f. Send consolidation_text as a new "base" prompt to seed a fresh slot
    g. Save the new slot as a snapshot
    h. Create a new commit:
         task = "consolidation (compacted N sessions into dense snapshot)"
         tokens_this_session = len(consolidation_text.split())  # rough estimate
         tokens_saved = 0  # consolidation doesn't save, it prevents waste
    i. Stop server
    j. Return new commit
    k. Log consolidation to .cacheflow/consolidation.log

  def maybe_compact_async(self, agent: Agent) -> None:
    Runs compact() in a background thread using ThreadPoolExecutor.
    Does not block the caller.
    Logs result to .cacheflow/compaction.log.

Call maybe_compact_async() at the end of AgentSession.run() after
the commit is created.
```

### Claude Code Prompt 5.2 — Consolidation Tests

```
Write pytest tests in tests/test_compressor.py.

IMPORTANT: Test the mechanics, not the semantics.
Don't test that "knowledge from session 1 appears in session 4" — LLM outputs
are non-deterministic. Instead test that consolidation works mechanically:

1. test_consolidation_save_restore()
   - Run agent 3 times to accumulate knowledge
   - Trigger consolidation
   - Assert consolidation produces a non-empty snapshot file
   - Load the agent again
   - Assert it can run a new task without error
   - (Don't assert specific knowledge is preserved.)

2. test_consolidation_triggers_at_threshold()
   - Create mock commits that sum to >70% of ctx_size
   - Assert needs_compaction() returns True
   - Mock the server, trigger consolidation
   - Assert compact() produces a commit

3. test_consolidation_logs_result()
   - Trigger consolidation
   - Assert result is logged to .cacheflow/consolidation.log
```

**Done condition:** Run an agent 10 times. The context size should never
exceed 70% of ctx_size. Check .cacheflow/consolidation.log to confirm
consolidation ran. `uv run pytest tests/test_compressor.py -v` passes.

---

## Session 6 — The Demo Script

**Goal:** A single script that proves the value proposition end-to-end.
This is what you run for YC.

**Time estimate:** 1 hour.

### Claude Code Prompt 6.1 — Demo Script

```
Create demo.py in the project root.

This script demonstrates AgentGit's value proposition by running a coding
agent against the cacheflow codebase itself.

Steps:
1. Print banner:
   ╔══════════════════════════════════════════════════╗
   ║            AgentGit — Live Demo                  ║
   ║   Agents that remember. Costs that drop.         ║
   ╚══════════════════════════════════════════════════╝

2. Initialize cacheflow in a temp directory that contains a copy of
   the cacheflow source code.

3. Run the agent 5 times with these tasks, recording tokens each time:
   Session 1: "Analyze the codebase structure. What are the main modules
               and what does each do?"
   Session 2: "What database schema does this project use? List all tables
               and their key fields."
   Session 3: "How does the KV cache save/restore flow work? Trace the
               code path from CLI to llama-server."
   Session 4: "What error handling exists in server.py? What cases are
               not handled?"
   Session 5: "If I wanted to add support for multiple concurrent agents,
               what would need to change?"

4. After each session, print:
   Session N complete.
   Tokens this session: X,XXX
   Tokens saved vs baseline: X,XXX (XX% reduction)
   Save time: XXms  Restore time: XXms
   Cumulative savings: XX,XXX tokens
   
   Note: If save/restore times are very high (>30s), that's a signal to
   consider reducing ctx_size in future runs.

5. After all 5 sessions, print the savings curve:
   TOKEN USAGE PER SESSION
   ─────────────────────────────────────────────────
   Session 1  ████████████████████████████████ 52,000
   Session 2  ████                              4,200
   Session 3  ███                               3,800
   Session 4  ███                               3,600
   Session 5  ███                               3,400
   ─────────────────────────────────────────────────
   Total saved: ~87,000 tokens (84% reduction by session 5)

6. Fork a sub-agent:
   Print: "Forking sub-agent for test coverage task..."
   Fork main → test-agent
   Run test-agent with: "What tests are missing? List the top 5
   highest-priority test cases that should be written."
   Print the response and its token count.
   Print: "Sub-agent used X tokens. It inherited full codebase knowledge
   from main agent at zero re-ingestion cost."

7. Show commit graph:
   Run `cf dashboard` as a subprocess for 5 seconds then exit.

Run with: uv run python demo.py
```

**Done condition:** `uv run python demo.py` runs end-to-end without errors.
Session 2+ shows dramatically lower token counts than session 1.
The fork runs and shows the inherited knowledge working.

---

## Session 7 — Polish + README

**Goal:** Make it presentable for GitHub and YC.

### Claude Code Prompt 7.1 — README

```
Write README.md for the cacheflow project.

Structure:
1. One-line description
2. The problem (3 sentences max)
3. How it works (with the token savings example table)
4. Quick start (5 commands to get running)
5. CLI reference (all commands)
6. How it works technically (KV cache persistence explanation, 2 paragraphs)
7. Roadmap (the OS-inspired solutions: tiered paging, CoW forking,
   idle consolidation — label these as "coming soon")
8. Architecture diagram in ASCII art

Keep it under 300 lines. Dense. Technical. No fluff.
```

### Claude Code Prompt 7.2 — Error Handling Pass

```
Do a full error handling pass on all files in cacheflow/.

For each error, add clear, actionable error messages and implement recovery:

Error Recovery Matrix:

1. llama-server binary not found
   → Error: "llama-server not found. Run: brew install llama.cpp"
   → Recovery: User installs llama.cpp

2. GGUF model not found
   → Error: "Model not found in: [list paths]. Run: ollama pull {model_name}"
   → Recovery: User pulls the model or specifies --model-path

3. Port conflict (8080 already in use)
   → Recovery: Automatic (already in server.py — try 8081, 8082, etc.)

4. Snapshot file corrupted or missing
   → Error: "Snapshot corrupted. Run: cacheflow recover --agent NAME"
   → Recovery: Move bad snapshot to .backup, start fresh from parent commit

5. Model hash mismatch
   → Error: "Snapshot created with model {old_hash}, but loaded {new_hash}.
             Delete .cacheflow/ and reinit, or restore from backups."
   → Recovery: Delete .cacheflow/ or restore snapshots

6. llama.cpp version mismatch (detected on init/run)
   → Error: "Snapshots created with llama.cpp {old_version}, but you have
             {new_version}. Snapshots may be incompatible."
   → Recovery: Suggest cf run --new-agent to start fresh

7. Context overflow (tokens_accumulated > ctx_size)
   → Recovery: Auto-trigger consolidation. If consolidation fails 3x,
             warn: "Consolidation failed. Start new branch: cf fork"

8. llama-server crash during session
   → Error: Catch subprocess exit, read server.log tail, raise AgentSessionError
   → Recovery: Log the crash details. User can retry with smaller --ctx-size

Add a custom exception hierarchy in cacheflow/__init__.py:
  AgentGitError (base)
  ├── ServerError
  ├── SnapshotError
  ├── ModelMismatchError
  ├── VersionMismatchError
  └── AgentSessionError

Implementation details:
- Store llama_cpp_version in config.json on init
- On run, check if llama_cpp_version differs from current version
- Add cacheflow recover command to move bad snapshots
- Wrap llama-server subprocess exit with try/except, log to server.log
```

**Done condition:** Run `uv run python demo.py` and intentionally break
things (wrong model name, missing snapshot dir) to confirm error messages
are clear and helpful.

---

## What You Have After Session 7

A working CLI tool that:
- Persists agent KV cache state across sessions
- Shows measurable token reduction (session 2 costs ~5-10% of session 1)
- Supports fork/merge with inherited context
- Has a commit DAG with full session history
- Runs background compaction to prevent context overflow
- Has a demo script that proves the value prop in 5 minutes

---

## Claude Code Tips for This Project

**Start each session by saying this:**
"We're building cacheflow, a CLI tool that gives AI agents persistent KV
cache memory across sessions using llama.cpp's slot save/restore API.
Read the existing code in cacheflow/ before writing anything new."

**When Claude Code gets stuck on llama-server API:**
The llama-server REST API docs are at:
https://github.com/ggerganov/llama.cpp/blob/master/tools/server/README.md
Tell Claude Code to fetch this URL before implementing server.py.

**When tests fail:**
Ask Claude Code to run the failing test with -v and -s flags to see
full output, then fix the specific failure before moving on.

**The most likely failure point:**
The slot save/restore API in llama-server. The exact endpoint paths and
request body format change between llama.cpp versions. If `/slots/0/save`
returns 404, ask Claude Code to check the llama-server --help output and
find the correct endpoint for your installed version.

**Measuring token savings:**
Token counts come from response["usage"]["prompt_tokens"] and ["completion_tokens"]
(or response["tokens_evaluated"] depending on llama.cpp version — check 
.cacheflow/api_schema.json to know which field to read).
Session 1 will be high (full context ingestion).
Session 2+ will be low (only new task tokens, cache already loaded).
This is your proof.

**CRITICAL: ctx_size is immutable:**
Once set during cf init, ctx_size cannot change between sessions.
If you change ctx_size, old snapshots become incompatible and restore will 
fail or produce garbage. There is no automatic migration. To change ctx_size,
delete .cacheflow/ and reinit. This is enforced by server.py on startup.

---

## The YC Demo Flow

After Session 6 is done, this is your demo:

```bash
cd my-real-project
cf init --model llama3.1:8b
cf run "Understand the full architecture of this codebase"
# Shows: 52,000 tokens, 0 saved

cf run "What are the three highest-priority bugs to fix?"
# Shows: 3,200 tokens, 48,800 saved (93.8% reduction)

cf run "Write a fix for the first bug"
# Shows: 2,900 tokens, 49,100 saved

cf fork main bug-fix-agent --scope "Focus only on the auth module"
cf run --agent bug-fix-agent "Fix all auth-related issues"
# Shows: 1,800 tokens — inherited full codebase knowledge, scoped to auth

cf dashboard
# Shows the commit graph + token savings curve
```

Point at the curve. "Session 1: 52,000 tokens. Session 2: 3,200 tokens.
Same model. Same quality. The agent just stopped being an amnesiac."

---

## Critical Checklist Before Starting

**You must complete these before any Claude Code sessions:**

- [ ] Run `python3 scripts/validate_llama_api.py` and it passes
- [ ] `.cacheflow/api_schema.json` exists with the real response schema
- [ ] You understand that ctx_size is immutable once set
- [ ] You understand that merge is v2, not MVP
- [ ] You've reviewed what "mock based on real schema" means for Session 3

If any of these is unclear, ask before starting.

---

## High-Risk Checkpoints During Implementation

Watch these closely:

| Session | Risk | Mitigation |
|---------|------|-----------|
| 1 | llama-server API differs | Pre-flight validation script catches this |
| 1 | Restore timing >30s | Measured in smoke test; consider smaller ctx_size |
| 2 | Schema missing fields | Add llama_cpp_version, save_time_ms, restore_time_ms to Commit |
| 3 | Token counts read from wrong field | Use .cacheflow/api_schema.json as source of truth |
| 3 | Mock doesn't match reality | Build mock to match api_schema.json exactly |
| 3 | Snapshot atomicity broken | Rename tmpfile AFTER DB transaction succeeds |
| 5 | Consolidation loses knowledge | Tests validate save/load works, not that knowledge is preserved |
| 6 | Demo runs too slowly | I/O time dominates; measure and document |

---

## What's Deferred to v2

- [ ] Merge operation (depends on consolidation)
- [ ] Multiple concurrent agents (needs slot pool)
- [ ] Distributed agents
- [ ] Snapshot compression (CoW, tiered paging)
- [ ] Advanced dashboard features
