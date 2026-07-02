"""Reproduce W2's 865s cold-prime anomaly and pinpoint which step is slow.

Baseline: W1's fresh engine primed itsdangerous in ~19s.
Anomaly:  W2's post-teardown re-init primed click in ~865s (~14 min).

Both workloads run on separate corpora with `stop_global_engine()` between
them. The prime itself takes ~15-30s (which we verified elsewhere), so the
865s figure includes something not in prime_slot proper. Suspects:

  1. Engine load time — Llama() constructor after teardown (Metal shader recompile?)
  2. prime_slot itself — first eval after fresh Metal context
  3. save_slot — writing the KV state to disk (could be huge)
  4. update_agent_snapshot / DB write

Times each phase, prints per-step ms.
"""
from __future__ import annotations
import os, shutil, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cacheflow.engine import get_global_engine, stop_global_engine
from cacheflow.agent import AgentSession

CORP1 = REPO_ROOT / "experiments" / "corpora" / "itsdangerous"
CORP2 = REPO_ROOT / "experiments" / "corpora" / "click"

def _stage(corpus_dir: Path) -> None:
    cf = corpus_dir / ".cacheflow"
    if cf.exists():
        shutil.rmtree(cf)
    cf.mkdir()
    (cf / "snapshots").mkdir()
    src = REPO_ROOT / ".cacheflow" / "config.json"
    shutil.copy(src, cf / "config.json")

def run_prime(label: str, corpus_dir: Path) -> None:
    os.chdir(corpus_dir)
    _stage(corpus_dir)

    print(f"\n== {label} ({corpus_dir.name}) ==")
    t = time.time()
    agent = AgentSession(f"diag_{label}", corpus_dir)
    print(f"  AgentSession() ctor:       {int((time.time()-t)*1000):>7d} ms")

    t = time.time()
    agent._acquire_lock()
    print(f"  _acquire_lock:             {int((time.time()-t)*1000):>7d} ms")

    t = time.time()
    agent.server = get_global_engine(
        model_path=agent.config.model_path,
        slot_save_path=str(agent.config.slot_save_path),
        ctx_size=agent.config.ctx_size,
        n_gpu_layers=agent.config.n_gpu_layers,
    )
    print(f"  get_global_engine:         {int((time.time()-t)*1000):>7d} ms")

    t = time.time()
    a = agent.store.get_agent(agent.agent_name)
    if a is None:
        a = agent.store.create_agent(
            agent.agent_name, agent.config.model_name,
            agent.config.model_hash, agent.config.ctx_size,
        )
    print(f"  create_agent (DB):         {int((time.time()-t)*1000):>7d} ms")

    t = time.time()
    stable_prefix = agent._build_stable_prefix("You are a code agent.")
    print(f"  _build_stable_prefix:      {int((time.time()-t)*1000):>7d} ms "
          f"({len(stable_prefix):,d} chars)")

    t = time.time()
    agent.server.prime_slot(stable_prefix, slot_id=agent.slot_id)
    print(f"  prime_slot (eval):         {int((time.time()-t)*1000):>7d} ms  <-- THIS one")

    t = time.time()
    save_result = agent.server.save_slot(slot_id=agent.slot_id)
    print(f"  save_slot (KV to disk):    {int((time.time()-t)*1000):>7d} ms "
          f"({save_result.get('filename', 'n/a')})")

    agent._release_lock()

def main():
    run_prime("W1_first",   CORP1)
    print("\n-- stop_global_engine() --")
    t = time.time()
    stop_global_engine()
    print(f"  stop_global_engine:        {int((time.time()-t)*1000):>7d} ms")

    # Simulate what run.py does between workloads: fresh SlotPool
    import cacheflow.agent as _am
    from cacheflow.slot_pool import SlotPool
    _am._SLOT_POOL = SlotPool(max_slots=8)

    run_prime("W2_second",  CORP2)

if __name__ == "__main__":
    main()
