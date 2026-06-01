# Multi-Agent Orchestration Design

## Overview

CacheFlow now supports concurrent execution of multiple agents through a **slot-based KV cache architecture**. Instead of blocking all agents behind an exclusive file lock, the system allocates independent KV cache slots from a shared llama.cpp server instance.

## Architecture

### Single Server, Multiple Slots

```
┌─ LlamaServer (one model in memory) ─────────────┐
│                                                  │
│  Slot 0: Agent A (KV cache)   ← Agent A state  │
│  Slot 1: Agent B (KV cache)   ← Agent B state  │
│  Slot 2: Agent C (KV cache)   ← Agent C state  │
│  Slot 3: [free]                                 │
│  ...                                            │
│                                                  │
│ All agents run concurrently, swapping slots     │
└──────────────────────────────────────────────────┘
```

### Benefits

1. **Memory Efficient**: Single model instance shared across agents (vs N model copies)
2. **True Concurrency**: Multiple agents run in parallel without blocking
3. **Deterministic Token Savings**: Each agent's baseline is immutable and per-slot
4. **Acyclic DAG**: No agent can rewind another agent's history
5. **Backward Compatible**: Existing single-agent code works unchanged

## SlotPool

The `SlotPool` class manages slot allocation and LRU eviction:

```python
from cacheflow.slot_pool import SlotPool

pool = SlotPool(max_slots=8)  # Typical for llama.cpp

# Acquire a slot for an agent
agent_id = uuid4()
lease = pool.acquire_slot(agent_id)
slot_id = lease.slot_id  # Use this slot_id in server calls

# Mark slot as dirty (has unsaved changes)
pool.mark_dirty(lease.slot_id)

# Record what was loaded into the slot
pool.load_commit(slot_id, commit_id, agent_id)

# Release when done (marks as available for eviction)
pool.release_slot(lease.slot_id)
```

### Allocation Strategy

1. **Reuse**: If agent already has a slot, return it (and update access_time for LRU tracking)
2. **Free Slot**: If a free slot exists, allocate it
3. **LRU Eviction**: If all slots full, evict the least-recently-used agent's slot and reuse it

### Concurrency Safety

- All operations protected by `threading.RLock()`
- Multiple threads can safely call `acquire_slot()` simultaneously
- No deadlocks: each agent gets a guaranteed slot (eviction if needed)

## Integration with AgentSession

The `AgentSession` class now uses SlotPool:

```python
class AgentSession:
    def __init__(self, agent_name: str, base_path: Path):
        ...
        self.slot_lease = None  # SlotLease from global pool
        self.slot_id = None     # Which slot this agent is using
        
    def _acquire_lock(self):
        # Acquire a slot from global _SLOT_POOL
        # Maintains backward compatibility with old lock API
        agent = self.store.get_agent(self.agent_name)
        self.slot_lease = _SLOT_POOL.acquire_slot(agent.id)
        self.slot_id = self.slot_lease.slot_id
        
    def run(self, task, ...):
        self.server.completion(..., slot_id=self.slot_id, ...)
        self.server.restore_slot(..., slot_id=self.slot_id)
        self.server.save_slot(slot_id=self.slot_id)
```

All server calls use `self.slot_id` instead of hardcoded `0`.

## Multi-Agent Workflow Example

```python
from cacheflow.agent import AgentSession
from pathlib import Path
import threading

base_path = Path(".")

# Create multiple agent sessions
agent_a = AgentSession("agent-research", base_path)
agent_b = AgentSession("agent-implement", base_path)
agent_c = AgentSession("agent-test", base_path)

def run_agent(agent, task):
    result = agent.run(task)
    print(f"{agent.agent_name}: {result.response[:100]}")

# Run all three concurrently
threads = [
    threading.Thread(target=run_agent, args=(agent_a, "Research the architecture")),
    threading.Thread(target=run_agent, args=(agent_b, "Implement the design")),
    threading.Thread(target=run_agent, args=(agent_c, "Write tests")),
]

for t in threads:
    t.start()
for t in threads:
    t.join()
```

Each agent:
- Gets its own KV cache slot from the pool
- Runs independently
- Saves/restores its own snapshot
- Tracks its own token baseline
- Completes without blocking others

## Token Savings Across Agents

Token savings computation is per-agent and immutable:

```
Agent A baseline: 100 tokens (established in first session)
Agent A session 2: 40 tokens evaluated → saves 60 tokens
Agent A session 3: 35 tokens evaluated → saves 65 tokens

Agent B baseline: 150 tokens (different prompt/codebase)
Agent B session 2: 50 tokens evaluated → saves 100 tokens
```

Each agent tracks its own `baseline_tokens_evaluated`, set on first session and used for all subsequent savings calculations.

## Slot Eviction Behavior

When slots are full and a new agent needs one:

1. **Find LRU slot**: Identify agent with oldest `access_time`
2. **Remove mapping**: Delete `agent_id -> slot_id` mapping for evicted agent
3. **Clear slot**: Set `loaded_commit_id = None`, `is_dirty = False`
4. **Reuse slot**: Assign to new agent with fresh state

**Important**: Evicted agents can request a slot again later (allocated a different slot if available).

## Database Schema

The `Agent` and `Commit` tables remain unchanged:

- `Agent.head_commit_id`: Points to agent's latest snapshot
- `Commit.parent_id`: Forms the DAG for each agent
- `Commit.forked_from_id`: Tracks forks between agents

No slot information is stored in the database because slots are **ephemeral runtime state**. Snapshots are loaded into slots as needed.

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| `acquire_slot()` | O(1) avg, O(n) worst | n = max_slots; worst case requires LRU search |
| `release_slot()` | O(1) | Just updates access_time |
| `mark_dirty()` | O(1) | State update |
| Concurrent acquire | O(1) per thread | Protected by threading.RLock |

## Testing

Multi-agent functionality is tested in `tests/test_multi_agent.py`:

- Slot allocation and reuse
- LRU eviction correctness
- Concurrent acquisition from multiple threads
- Slot state tracking (loaded commits, dirty flags)
- Statistics and monitoring

All 75 tests pass, including:
- 11 original agent tests (now using slots instead of locks)
- 10 new multi-agent tests
- 54 tests across store, config, CLI, RAG, compression

## Future Enhancements

1. **Slot Priority Queue**: Give certain agents (e.g., critical paths) higher eviction priority
2. **Heterogeneous Models**: Support different model versions in different slots
3. **Slot Pinning**: Prevent eviction of frequently-used agents
4. **Metrics**: Export slot pool stats to monitoring (Prometheus, etc.)
5. **Async Support**: Enable true async/await multi-agent orchestration

## Migration Notes

**Existing single-agent code**: No changes needed. AgentSession maintains backward-compatible API:
- `_acquire_lock()` / `_release_lock()` still work
- File lock replaced with slot lease (transparent)
- All tests pass without modification

**Backward Compatibility**: 
- Old `AgentSession` code works unchanged
- SlotPool is a drop-in replacement for file locking
- Can mix old (lock-based) and new (slot-based) workflows
