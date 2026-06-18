#!/bin/bash
# Sequential humor-family runs (avoids MPS contention). Per-model error isolation.
cd /Users/amber/Projects/Oxford/Judging-Humor
PY=.venv-es/bin/python
T=data/bill-burr-drop-dead-years.txt
N="Bill Burr — Drop Dead Years"

MODELS=(
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen3-8B"
  "Qwen/Qwen3.5-9B"
  "google/gemma-4-12b-it"
)

for M in "${MODELS[@]}"; do
  echo "############################################################"
  echo "### RUN $M  $(date +%H:%M:%S)"
  echo "############################################################"
  $PY emotionscope_humor.py --model "$M" --transcript "$T" --name "$N" \
      2>&1 | tr '\r' '\n' | grep -avE "it/s|s/it|%\|" \
    && echo "### OK $M" || echo "### FAILED $M"
done
echo "### ALL DONE $(date +%H:%M:%S)"
