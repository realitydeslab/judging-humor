"""Can an LLM's humor emotion vectors predict the AST-detected laughs in the
Ricky Gervais clip (location AND strength)?

Reads the ASR transcript through a model using its SAVED emotion vectors,
word-aligns the signal, maps each laugh timestamp to its punchline word, and
tests: permutation (signal before laughs vs random), detection AUC, peri-laughter
average, and Spearman correlation of the pre-laugh signal with laugh strength.
"""
from __future__ import annotations
import json
import numpy as np
import torch
from bisect import bisect_right

from emotion_scope import load_model
import emotionscope_humor as H
import align_and_report as ar

MODEL = "google/gemma-2-9b-it"
TAG = "gemma-2-9b-it"
MONITOR = ["amused", "playful", "delighted", "mirthful", "surprised", "bored"]
PRE = 6  # words before a laugh to average (the punch-in window)


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    asr = json.load(open("audio/gervais_fatpeople_asr.json"))
    lg = json.load(open("audio/gervais_fatpeople_laughter.json"))
    words = asr["words"]
    spoken = " ".join(w["w"] for w in words)

    # map laugh start-times -> punchline word index (last word ending before laugh)
    word_end = [w["end"] for w in words]
    laugh_pos, strengths = [], []
    for iv in lg["intervals"]:
        j = bisect_right(word_end, iv["start"]) - 1
        if j < 0:
            j = 0
        laugh_pos.append(j); strengths.append(iv["strength"])
    # dedupe identical word indices (keep max strength)
    bypos = {}
    for j, s in zip(laugh_pos, strengths):
        bypos[j] = max(bypos.get(j, 0), s)
    laugh_pos = sorted(bypos); strengths = np.array([bypos[j] for j in laugh_pos])
    print(f"{len(words)} words, {len(lg['intervals'])} laughs -> {len(laugh_pos)} punchline words")

    # load model + its saved emotion vectors
    npz = np.load(f"data/vectors/{TAG}.npz")
    names = [str(n) for n in npz["emotions"]]
    vectors = {n: torch.tensor(npz[f"vec_{n}"]).float() for n in names}
    layer = int(npz["probe_layer"])
    model, tok, backend, info = load_model(MODEL, device=dev, dtype="bfloat16",
                                           backend="huggingface")

    # read per-token cosine for the monitored emotions, aggregate to words
    enc = tok(spoken, return_offsets_mapping=True, add_special_tokens=False)
    tok_ids, offs = enc["input_ids"], enc["offset_mapping"]
    D = torch.stack([vectors[m] for m in MONITOR])
    raw = H.read_per_token(model, tok_ids, D, layer, dev)         # (n_mon, n_tok)

    # word starts for token->word mapping
    wstart, n_words = [], len(words)
    pos = 0
    for w in words:
        wstart.append(pos); pos += len(w["w"]) + 1
    word_scores = {}
    for mi, emo in enumerate(MONITOR):
        r = raw[mi]; fin = r[np.isfinite(r)]
        lo, hi = np.percentile(fin, [0.5, 99.5]); r = np.clip(r, lo, hi)
        z = (r - np.nanmean(r)) / (np.nanstd(r) + 1e-8)
        sums = np.zeros(n_words); cnts = np.zeros(n_words)
        for t, (cs, ce) in enumerate(offs):
            if ce <= cs:
                continue
            j = min(bisect_right(wstart, (cs + ce) // 2) - 1, n_words - 1)
            if j < 0:
                j = 0
            sums[j] += z[t]; cnts[j] += 1
        wv = sums / np.where(cnts > 0, cnts, 1)
        word_scores[emo] = (wv - wv.mean()) / (wv.std() + 1e-8)

    # ---- tests ----
    def spearman(a, b):
        ar_ = np.argsort(np.argsort(a)); br_ = np.argsort(np.argsort(b))
        return float(np.corrcoef(ar_, br_)[0, 1])

    print("\n=== can the model's emotions predict the laughs? ===")
    print(f"{'emotion':11s} {'detAUC':>7s} {'perm p':>7s} {'z>null':>7s} {'r(strength)':>11s}")
    results = {}
    for emo in MONITOR:
        sc = word_scores[emo]
        det = ar.detection_auc(sc, laugh_pos, pre_window=PRE)
        perm = ar.permutation_test(sc, laugh_pos, pre_window=PRE)
        # per-laugh signal = mean z in PRE words up to punchline
        per_laugh = np.array([np.mean(sc[max(0, j - PRE + 1):j + 1]) for j in laugh_pos])
        r = spearman(per_laugh, strengths)
        results[emo] = {"det_auc": round(det, 3), "perm_p": round(perm["p_value"], 4),
                        "z_above_null": round(perm["z_above_null"], 2), "r_strength": round(r, 3)}
        print(f"{emo:11s} {det:7.3f} {perm['p_value']:7.4f} {perm['z_above_null']:+7.2f} {r:+11.3f}")

    json.dump({"model": MODEL, "n_words": n_words, "n_laughs": len(laugh_pos),
               "results": results}, open("audio/gervais_prediction.json", "w"), indent=1)
    print("\nsaved -> audio/gervais_prediction.json")
    print("detAUC>0.5 & perm p<.05 => emotion rises into laughs; r(strength)>0 => bigger laughs predicted")


if __name__ == "__main__":
    main()
