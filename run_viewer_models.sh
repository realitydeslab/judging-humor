#!/bin/bash
# Extract 25 vectors + word-level joke scores for each model, then build viewer.html
cd /Users/amber/Projects/Oxford/Judging-Humor
PY=.venv-es/bin/python
T=data/bill-burr-drop-dead-years.txt

MODELS=(
  "google/gemma-2-9b-it"
  "Qwen/Qwen2.5-14B-Instruct"
  "Qwen/Qwen3-14B"
  "Qwen/Qwen3.5-9B"
  "google/gemma-3-12b-it"
  "google/gemma-4-12b-it"
)

for M in "${MODELS[@]}"; do
  echo "############################################################"
  echo "### MODEL $M  $(date +%H:%M:%S)"
  echo "############################################################"
  $PY joke_pipeline.py --model "$M" --transcript "$T" \
      2>&1 | tr '\r' '\n' | grep -avE "it/s|s/it|%\||Loading checkpoint" \
    && echo "### OK $M" || echo "### FAILED $M"
  echo "### rebuilding viewer with whatever is done so far"
  $PY build_viewer.py 2>&1 | tail -2
done
echo "### ALL DONE $(date +%H:%M:%S)"
