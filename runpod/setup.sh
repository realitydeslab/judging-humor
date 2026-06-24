#!/usr/bin/env bash
# One-shot RunPod setup for the Judging-Humor emotion-LLM pipeline.
# Run this INSIDE the pod, from the repo root, after `git clone`.
#   bash runpod/setup.sh
set -euo pipefail

cd "$(dirname "$0")/.."
REPO="$(pwd)"
echo "[setup] repo at $REPO"

# 1. System deps (git-lfs handy for model caches; usually preinstalled)
apt-get update -qq && apt-get install -y -qq git-lfs >/dev/null 2>&1 || true

# 2. Put HF + torch caches on the persistent volume (survives pod restart,
#    avoids re-downloading 10s of GB of weights). RunPod mounts /workspace.
export HF_HOME="${HF_HOME:-/workspace/hf}"
mkdir -p "$HF_HOME"
echo "[setup] HF_HOME=$HF_HOME"

# 3. Python deps. torch comes from the PyTorch template; everything else pinned.
pip install --no-cache-dir -r runpod/requirements.txt

# 4. EmotionScope (the probing library) — pinned to the same commit as the Mac env.
pip install --no-cache-dir \
  "git+https://github.com/AidanZach/EmotionScope.git@8e8a2aeef40d0461c60aee6451ec95c7b2fb8856#egg=emotion_scope"

# 5. Sanity: GPU visible + key imports work.
python - <<'PY'
import torch, transformers, emotion_scope
print("torch", torch.__version__, "cuda avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0),
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
print("transformers", transformers.__version__, "emotion_scope OK")
PY

echo "[setup] done. Next:"
echo "  huggingface-cli login   # paste a token that has accepted the Gemma license"
echo "  python emotionscope_humor.py --model Qwen/Qwen2.5-14B-Instruct \\"
echo "      --transcript data/bill-burr-drop-dead-years.txt --name 'Bill Burr — Drop Dead Years'"
