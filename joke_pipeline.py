"""Per-model stage of the multi-model humor viewer.

For one model:
  1. extract all 25 EmotionScope emotion vectors  -> data/vectors/<tag>.npz
  2. read per-token cosine for ALL 25 emotions across the joke
  3. word-align the token scores (so every model shares the SAME display words)
     -> data/joke/<tag>.json   {tag, probe_layer, word_scores:{emotion:[z per word]}}
  4. (once) write the shared word stream + laughter positions -> data/joke/_words.json

Usage:  .venv-es/bin/python joke_pipeline.py --model <hf_id> [--transcript ...]
"""
from __future__ import annotations
import argparse, json, os, bisect
import numpy as np
import torch
from bisect import bisect_right

from emotion_scope import EmotionExtractor, EmotionProbe, load_model
from emotion_scope.config import ExtractionConfig
import emotionscope_humor as H   # load_gemma4, get_layers, read_per_token, EMOTIONS_ALL
import scenarios as SC

LAUGH = "[LAUGHTER]"


def model_tag(model_id):
    return model_id.split("/")[-1]


def build_word_stream(text):
    """Spoken-only words (laughter removed) + the word index after which each
    laugh occurred. Returns (spoken_text, words[list of (w,start,end)], laugh_after_word)."""
    parts = text.split(LAUGH)
    spoken = ""
    laugh_char_pos = []
    for i, chunk in enumerate(parts):
        chunk = chunk.strip()
        if chunk:
            if spoken:
                spoken += " "
            spoken += chunk
        if i < len(parts) - 1:
            laugh_char_pos.append(len(spoken))
    import re
    words = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", spoken)]
    starts = [w[1] for w in words]
    laugh_after_word = []
    for pos in laugh_char_pos:
        j = bisect_right(starts, pos) - 1
        if j >= 0:
            laugh_after_word.append(j)
    laugh_after_word = sorted(set(laugh_after_word))
    return spoken, words, laugh_after_word


def load_any(model_id, device):
    from transformers import AutoConfig
    mm = getattr(AutoConfig.from_pretrained(model_id), "text_config", None) is not None
    if mm:
        return H.load_gemma4(model_id, device, dtype="bfloat16")
    return load_model(model_id, device=device, dtype="bfloat16", backend="huggingface")


def extract_and_save_vectors(model, tok, backend, info, tag):
    os.makedirs("data/vectors", exist_ok=True)
    path = f"data/vectors/{tag}.npz"
    ex = EmotionExtractor(model, tok, backend, info, emotions=H.EMOTIONS_ALL)
    vectors = ex.extract()                      # uses consolidated 25-emotion templates
    names = [e["name"] for e in H.EMOTIONS_ALL if e["name"] in vectors]
    np.savez(path, probe_layer=info["probe_layer"], emotions=np.array(names),
             **{f"vec_{n}": vectors[n].cpu().float().numpy() for n in names})
    print(f"saved {len(names)} vectors -> {path}")
    return vectors, names


@torch.no_grad()
def word_level_scores(model, tok, spoken, words, vectors, names, layer, device):
    enc = tok(spoken, return_offsets_mapping=True, add_special_tokens=False)
    token_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    D = torch.stack([vectors[n].cpu().float() for n in names])      # (25, d)
    raw = H.read_per_token(model, token_ids, D, layer, device)       # (25, n_tok)

    # z-score each emotion across tokens (winsorized)
    z = np.empty_like(raw)
    for i in range(raw.shape[0]):
        r = raw[i]; fin = r[np.isfinite(r)]
        lo, hi = np.percentile(fin, [0.5, 99.5]); r = np.clip(r, lo, hi)
        z[i] = (r - np.nanmean(r)) / (np.nanstd(r) + 1e-8)

    # map each token to a word (by char midpoint)
    wstarts = [w[1] for w in words]; wends = [w[2] for w in words]
    n_words = len(words)
    sums = np.zeros((len(names), n_words)); cnts = np.zeros(n_words)
    for t, (cs, ce) in enumerate(offsets):
        if ce <= cs:
            continue
        mid = (cs + ce) // 2
        j = bisect_right(wstarts, mid) - 1
        if j < 0:
            j = 0
        if j >= n_words:
            j = n_words - 1
        # if midpoint past this word's end and a next word starts before ce, prefer overlap
        sums[:, j] += z[:, t]; cnts[j] += 1
    cnts_safe = np.where(cnts > 0, cnts, 1)
    wscore = sums / cnts_safe                                        # (25, n_words)
    # z-score across words for display contrast
    out = {}
    for i, n in enumerate(names):
        v = wscore[i]
        v = (v - v.mean()) / (v.std() + 1e-8)
        out[n] = [round(float(x), 3) for x in v]
    return out


def scenario_matrix(model, tok, backend, info, vectors):
    """Emotion-probe x scenario cosine matrix (EmotionScope probe, response-prep
    position) — the 'probes respond to implicit emotional content' heatmap."""
    probe = EmotionProbe(model, tok, backend, info, vectors,
                         emotions_metadata=H.EMOTIONS_ALL)
    rows = [r for r in SC.ROWS if r in vectors]
    cols = [lbl for lbl, _ in SC.SCENARIOS]
    vals = [[0.0] * len(SC.SCENARIOS) for _ in rows]
    for cj, (_lbl, txt) in enumerate(SC.SCENARIOS):
        st = probe.analyze(txt)
        for ri, r in enumerate(rows):
            vals[ri][cj] = round(float(st.scores.get(r, 0.0)), 4)
    return {"rows": rows, "cols": cols, "values": vals}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--transcript", default="data/bill-burr-drop-dead-years.txt")
    args = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    tag = model_tag(args.model)

    text = open(args.transcript).read()
    spoken, words, laugh_after_word = build_word_stream(text)
    os.makedirs("data/joke", exist_ok=True)
    if not os.path.exists("data/joke/_words.json"):
        json.dump({"words": [w[0] for w in words],
                   "laugh_after_word": laugh_after_word,
                   "transcript": os.path.basename(args.transcript)},
                  open("data/joke/_words.json", "w"))
        print(f"wrote shared word stream: {len(words)} words, {len(laugh_after_word)} laughs")

    model, tok, backend, info = load_any(args.model, device)
    layer = info["probe_layer"]
    vectors, names = extract_and_save_vectors(model, tok, backend, info, tag)
    smatrix = scenario_matrix(model, tok, backend, info, vectors)
    print(f"scenario matrix: {len(smatrix['rows'])} emotions x {len(smatrix['cols'])} scenarios")
    wscores = word_level_scores(model, tok, spoken, words, vectors, names, layer, device)
    out = {"tag": tag, "model": args.model, "probe_layer": int(layer),
           "emotions": names, "word_scores": wscores, "scenario_matrix": smatrix}
    json.dump(out, open(f"data/joke/{tag}.json", "w"))
    print(f"saved word-level scores -> data/joke/{tag}.json  ({len(names)} emotions x {len(words)} words)")


if __name__ == "__main__":
    main()
