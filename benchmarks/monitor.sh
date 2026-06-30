#!/bin/bash
# Real-time benchmark monitoring script

set -e

RESULTS_DIR="benchmarks/results/raw"
LOGS_DIR="benchmarks/results"

clear
echo "======================================================================"
echo "CacheFlow Benchmark Monitor — Tier 2 Full Run"
echo "======================================================================"
echo "Time: $(date)"
echo ""

echo "AXIS 1: Scaling (HumanEval across 8 repos)"
echo "  Expected: 50 tasks × 8 repos = 400 total (cold + warm)"
echo "  Status:"
humaneval_files=$(ls -t ${RESULTS_DIR}/humaneval_local_*.jsonl 2>/dev/null | head -3)
if [ -n "$humaneval_files" ]; then
  for f in $humaneval_files; do
    name=$(basename "$f")
    rows=$(wc -l < "$f" 2>/dev/null || echo "0")
    size=$(du -h "$f" 2>/dev/null | cut -f1)
    printf "    %-50s %5d rows  %6s\n" "$name" "$rows" "$size"
  done
else
  echo "    (no results yet)"
fi

echo ""
echo "AXIS 2: Quality (SWE-bench Lite on 3 repos)"
echo "  Expected: 30 tasks × 3 repos = 90 total (cold + warm)"
echo "  Status:"
swebench_files=$(ls -t ${RESULTS_DIR}/swebench-lite_local_*.jsonl 2>/dev/null | head -3)
if [ -n "$swebench_files" ]; then
  for f in $swebench_files; do
    name=$(basename "$f")
    rows=$(wc -l < "$f" 2>/dev/null || echo "0")
    size=$(du -h "$f" 2>/dev/null | cut -f1)
    printf "    %-50s %5d rows  %6s\n" "$name" "$rows" "$size"
  done
else
  echo "    (no results yet)"
fi

echo ""
echo "Markdown Logs:"
for log in ${LOGS_DIR}/tier2_*.md; do
  if [ -f "$log" ]; then
    name=$(basename "$log")
    lines=$(wc -l < "$log" 2>/dev/null || echo "0")
    printf "  %-60s %5d lines\n" "$name" "$lines"
  fi
done

echo ""
echo "======================================================================"
echo "To see detailed logs:"
echo "  tail -50 benchmarks/results/tier2_scaling_humaneval_*.md"
echo "  tail -50 benchmarks/results/tier2_quality_swebench_*.md"
echo ""
echo "To generate report after runs complete:"
echo "  python3 benchmarks/report.py --results-dir benchmarks/results/raw --out benchmarks/results/summary"
echo "======================================================================"
