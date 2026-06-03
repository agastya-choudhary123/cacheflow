#!/usr/bin/env python3
"""
Run CacheFlow against its own source code and stream Q&A into session_log.md.
Progressive chunk ingestion happens automatically inside session.run() on first call.
"""

import sys
import time
from pathlib import Path
from datetime import datetime
from sqlalchemy.orm import Session as SQLSession

sys.path.insert(0, str(Path(__file__).parent))

from cacheflow.agent import AgentSession, DEFAULT_SYSTEM_PROMPT
from cacheflow.store import CacheFlowStore, Agent, Commit, SessionLog

BASE_PATH = Path(__file__).parent
OUTPUT_FILE = BASE_PATH / "session_log.md"

QUESTIONS = [
    "Walk me through the full architecture of this codebase. What are the key modules and how do they connect?",
    "How does KV cache persistence work end to end? Trace a single `cf run` from CLI call to snapshot saved on disk.",
    "What does the SlotPool do and what happens when all 8 slots are occupied? Be specific about the LRU eviction code.",
    "How does the compressor decide when to consolidate? Walk through the exact threshold logic and what it does.",
    "How does semantic search work across snapshots? What does the retriever do and how are embeddings stored?",
]

def clear_agent(store, name):
    a = store.get_agent(name)
    if not a:
        return
    with SQLSession(store.engine) as s:
        s.query(SessionLog).filter(SessionLog.agent_id == a.id).delete()
        s.query(Commit).filter(Commit.agent_id == a.id).delete()
        s.query(Agent).filter(Agent.id == a.id).delete()
        s.commit()
    print(f"Cleared '{name}'")

def write(f, text):
    f.write(text)
    f.flush()
    print(text, end="", flush=True)

def run():
    store = CacheFlowStore(BASE_PATH / ".cacheflow" / "agents.db")
    store.init_db()
    clear_agent(store, "codebase-reader")

    with open(OUTPUT_FILE, "w") as f:
        write(f, f"# CacheFlow Self-Analysis\n")
        write(f, f"**Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n")
        write(f, "---\n\n")

        session = AgentSession("codebase-reader", BASE_PATH)

        for i, question in enumerate(QUESTIONS, 1):
            write(f, f"## Q{i}\n\n**{question}**\n\n")

            t0 = time.time()
            try:
                result = session.run(
                    task=question,
                    system_prompt=DEFAULT_SYSTEM_PROMPT,
                    max_tokens=700,
                )
                elapsed = int(time.time() - t0)

                write(f, f"**Tokens used:** {result.tokens_this_session} &nbsp;|&nbsp; ")
                write(f, f"**Saved:** {result.tokens_saved} &nbsp;|&nbsp; ")
                write(f, f"**Time:** {elapsed}s")
                if result.is_first_session:
                    write(f, " &nbsp;|&nbsp; *first session — full codebase ingested across chunks*")
                write(f, "\n\n")
                write(f, result.response.strip())
                write(f, "\n\n---\n\n")

            except Exception as e:
                write(f, f"**Error:** {e}\n\n---\n\n")
                import traceback; traceback.print_exc()
                break

        write(f, f"\n**Done:** {datetime.now().strftime('%H:%M:%S')}\n")

if __name__ == "__main__":
    run()
