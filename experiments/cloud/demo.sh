#!/bin/bash
# Cloud experiment: demonstrate the thinking-block / knowledge-pool reuse method.
#
#   Agent A (implementer) reads cacheflow/engine.py and submits a summary.
#   Agent B (reviewer) queries the pool BEFORE reading the file. Gets Agent A's
#   summary, saves the tokens Agent B would have burned re-reading engine.py.
#
# Both agents are spawned as separate `claude -p` non-interactive processes so
# you can see exactly what the feature-branch cloud method looks like end-to-end.

set -eu
cd /Users/agastya/Desktop/agentgit

RESULTS=experiments/results/cloud_$(date +%s).json
TARGET_FILE=cacheflow/engine.py
REGION_HASH=$(git hash-object "$TARGET_FILE")
TARGET_TOKENS=$(python3 -c "print(len(open('$TARGET_FILE').read().split()))")
POOL_BEFORE=$(cf knowledge query "$TARGET_FILE" --region-hash "$REGION_HASH" 2>&1 | wc -c | tr -d ' ')

echo "== Cloud Experiment: Knowledge-Pool Reuse =="
echo "  Target file:      $TARGET_FILE"
echo "  Region hash:      $REGION_HASH"
echo "  File size (words): $TARGET_TOKENS"
echo "  Pool state before: $POOL_BEFORE chars (empty = no hit)"
echo ""

# --- Agent A: read + summarize + submit ---
echo "-- Agent A (implementer) — reads file, writes summary to pool --"
T0=$(python3 -c "import time;print(time.time())")
AGENT_A_OUT=$(claude -p "Read the file cacheflow/engine.py. Then write a dense summary (max 200 tokens) of what LlamaEngine does — focus on: model loading, slot management, save/restore of KV state. Output ONLY the summary, no preamble." 2>&1 || echo "AGENT_A_FAILED")
T1=$(python3 -c "import time;print(time.time())")
AGENT_A_MS=$(python3 -c "print(int(($T1 - $T0) * 1000))")
echo "  Agent A duration: ${AGENT_A_MS}ms"
echo "  Summary chars:    ${#AGENT_A_OUT}"
echo ""
echo "$AGENT_A_OUT" | cf knowledge submit "$TARGET_FILE" \
    --region-hash "$REGION_HASH" \
    --role implementer \
    --summary-file - \
    --source-agent agent_a_cloud_demo \
    --token-count 200 2>&1 | tail -3

# --- Agent B: query pool BEFORE opening the file ---
echo ""
echo "-- Agent B (reviewer) — queries pool BEFORE reading anything --"
T2=$(python3 -c "import time;print(time.time())")
POOL_HIT=$(cf knowledge query "$TARGET_FILE" --region-hash "$REGION_HASH" --role implementer 2>&1)
T3=$(python3 -c "import time;print(time.time())")
POOL_QUERY_MS=$(python3 -c "print(int(($T3 - $T2) * 1000))")
echo "  Pool query duration: ${POOL_QUERY_MS}ms"
echo "  Pool hit chars:      ${#POOL_HIT}"
echo ""

POOL_HIT_WORDS=$(printf '%s' "$POOL_HIT" | wc -w | tr -d ' ')
if [ ${#POOL_HIT} -gt 100 ]; then
    echo "  Agent B skips reading the file — uses Agent A's summary directly."
    AGENT_B_TASK_INPUT_TOKENS=$POOL_HIT_WORDS
else
    echo "  ⚠ Pool query returned nothing useful — Agent B would have to read the file."
    AGENT_B_TASK_INPUT_TOKENS=$TARGET_TOKENS
fi

# --- Report ---
BASELINE_TOKENS=$TARGET_TOKENS
POOL_TOKENS=$AGENT_B_TASK_INPUT_TOKENS
SAVED=$((BASELINE_TOKENS - POOL_TOKENS))
SAVED_PCT=$(python3 -c "print(f'{100*$SAVED/max(1,$BASELINE_TOKENS):.1f}')")

echo ""
echo "== Result =="
echo "  Agent B input tokens WITHOUT pool: ~$BASELINE_TOKENS (whole $TARGET_FILE)"
echo "  Agent B input tokens WITH pool:    ~$POOL_TOKENS  (Agent A's summary)"
echo "  Tokens saved for Agent B:          ~$SAVED  (${SAVED_PCT}%)"

python3 - <<PY
import json, time
out = {
  "backend": "cloud",
  "method": "knowledge_pool_reuse",
  "target_file": "$TARGET_FILE",
  "region_hash": "$REGION_HASH",
  "agent_a_duration_ms": $AGENT_A_MS,
  "pool_query_duration_ms": $POOL_QUERY_MS,
  "agent_b_tokens_baseline": $BASELINE_TOKENS,
  "agent_b_tokens_with_pool": $POOL_TOKENS,
  "agent_b_tokens_saved": $SAVED,
  "agent_b_reduction_pct": float("$SAVED_PCT"),
  "timestamp": int(time.time()),
}
open("$RESULTS", "w").write(json.dumps(out, indent=2))
print(f"\n  Saved: $RESULTS")
PY
