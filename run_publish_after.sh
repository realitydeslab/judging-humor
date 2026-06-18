#!/bin/bash
# After the deferred backfill finishes (all 6 models + scenario matrices), rebuild
# the viewer and push so the live GitHub Pages site updates to the full set.
cd /Users/amber/Projects/Oxford/Judging-Humor
BF=/private/tmp/claude-501/-Users-amber-Projects-Oxford-Judging-Humor/05b09ae8-1a34-42e0-a664-00c603ae3cae/tasks/bye1m369j.output
echo "waiting for backfill to finish..."
until grep -qa "BACKFILL ALL DONE" "$BF" 2>/dev/null; do sleep 30; done
.venv-es/bin/python build_viewer.py 2>&1 | tail -1
git add -A
git commit -q -F - <<'EOF'
Update hosted viewer: all 6 models + probe×scenario matrices
EOF
git push -q origin main && echo "### PUBLISHED $(date +%H:%M:%S) https://realitydeslab.github.io/judging-humor/"
