#!/bin/bash
# Wait for the main 6-model run to finish, then backfill scenario matrices for
# the two models that ran before the heatmap code existed, and rebuild viewer.
cd /Users/amber/Projects/Oxford/Judging-Humor
PY=.venv-es/bin/python
F=/private/tmp/claude-501/-Users-amber-Projects-Oxford-Judging-Humor/05b09ae8-1a34-42e0-a664-00c603ae3cae/tasks/b0k9oudum.output
echo "waiting for main run to finish..."
until grep -qa "ALL DONE" "$F" 2>/dev/null; do sleep 30; done
echo "main run done; backfilling scenario matrices"
for M in "google/gemma-2-9b-it" "Qwen/Qwen2.5-14B-Instruct"; do
  $PY backfill_scenarios.py "$M" 2>&1 | tr '\r' '\n' | grep -avE "it/s|s/it|%\||Loading checkpoint" \
    && echo "### BACKFILL OK $M" || echo "### BACKFILL FAILED $M"
done
$PY build_viewer.py 2>&1 | tail -1
echo "### BACKFILL ALL DONE $(date +%H:%M:%S)"
