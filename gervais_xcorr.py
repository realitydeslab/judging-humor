"""Lagged cross-correlation: does a model's emotion-over-time signal predict the
laughter-strength-over-time signal in the Gervais clip?

Puts per-word emotion z on a time axis (word timestamps), resamples onto the
laughter timeline's 0.25s grid, and correlates emotion(t) with strength(t+lag)
across lags. Positive lag = emotion LEADS laughter. Significance via circular-
shift permutation (accounts for autocorrelation).
"""
from __future__ import annotations
import argparse, json
import numpy as np, torch
from bisect import bisect_right
from emotion_scope import load_model
import emotionscope_humor as H

MONITOR = ["amused", "playful", "delighted", "mirthful", "surprised", "curious", "bored"]


def read_word_scores(model, tok, dev, spoken, words, vectors, layer):
    enc = tok(spoken, return_offsets_mapping=True, add_special_tokens=False)
    raw = H.read_per_token(model, enc["input_ids"],
                           torch.stack([vectors[m] for m in MONITOR]), layer, dev)
    wstart, n = [], len(words); pos = 0
    for w in words:
        wstart.append(pos); pos += len(w["w"]) + 1
    out = {}
    for mi, emo in enumerate(MONITOR):
        r = raw[mi]; fin = r[np.isfinite(r)]
        lo, hi = np.percentile(fin, [0.5, 99.5]); r = np.clip(r, lo, hi)
        z = (r - np.nanmean(r)) / (np.nanstd(r) + 1e-8)
        s = np.zeros(n); c = np.zeros(n)
        for t, (cs, ce) in enumerate(enc["offset_mapping"]):
            if ce <= cs: continue
            j = min(max(0, bisect_right(wstart, (cs + ce)//2) - 1), n - 1)
            s[j] += z[t]; c[j] += 1
        wv = s / np.where(c > 0, c, 1)
        out[emo] = (wv - wv.mean()) / (wv.std() + 1e-8)
    return out


def xcorr(emotion_t, strength_t, max_lag_steps, lead_window):
    """r(lag)=corr(emotion[t], strength[t+lag]); return lags(steps), r array."""
    lags = np.arange(-max_lag_steps, max_lag_steps + 1)
    rs = []
    for L in lags:
        if L >= 0:
            a, b = emotion_t[:len(emotion_t) - L], strength_t[L:]
        else:
            a, b = emotion_t[-L:], strength_t[:len(strength_t) + L]
        rs.append(np.corrcoef(a, b)[0, 1] if len(a) > 5 else 0.0)
    return lags, np.array(rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--tag", default="gemma-2-9b-it")
    args = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    asr = json.load(open("audio/gervais_fatpeople_asr.json"))
    lg = json.load(open("audio/gervais_fatpeople_laughter.json"))
    words = asr["words"]; spoken = " ".join(w["w"] for w in words)
    wt = np.array([(w["start"] + w["end"]) / 2 for w in words])     # word center times
    gt = np.array(lg["timeline"]["t"]); strength = np.array(lg["timeline"]["strength"])
    dt = gt[1] - gt[0]                                              # 0.25s

    npz = np.load(f"data/vectors/{args.tag}.npz")
    names = [str(n) for n in npz["emotions"]]
    vectors = {n: torch.tensor(npz[f"vec_{n}"]).float() for n in names}
    layer = int(npz["probe_layer"])
    model, tok, backend, info = load_model(args.model, device=dev, dtype="bfloat16",
                                           backend="huggingface")
    ws = read_word_scores(model, tok, dev, spoken, words, vectors, layer)

    max_lag = int(round(4.0 / dt))            # +/- 4 s
    lead_lo, lead_hi = 0, int(round(2.5 / dt))  # emotion leads laughter by 0..2.5 s
    rng = np.random.default_rng(0)
    print(f"\n=== lagged cross-correlation (emotion LEADS laughter, model {args.tag}) ===")
    print(f"{'emotion':11s} {'peak r':>7s} {'lag(s)':>7s} {'perm p':>7s}")
    out = {}
    for emo in MONITOR:
        em_t = np.interp(gt, wt, ws[emo])      # emotion on the 0.25s grid
        lags, rs = xcorr(em_t, strength, max_lag, (lead_lo, lead_hi))
        # search the "emotion leads" window for the peak |r| (signed, expect +)
        win = (lags >= lead_lo) & (lags <= lead_hi)
        peak_i = np.argmax(rs[win]); peak_r = rs[win][peak_i]; peak_lag = lags[win][peak_i] * dt
        # circular-shift permutation null on strength
        null = []
        for _ in range(2000):
            sh = rng.integers(20, len(strength) - 20)
            srt = np.roll(strength, sh)
            _, rr = xcorr(em_t, srt, max_lag, (lead_lo, lead_hi))
            null.append(rr[win].max())
        p = (np.sum(np.array(null) >= peak_r) + 1) / (len(null) + 1)
        out[emo] = {"peak_r": round(float(peak_r), 3), "lag_s": round(float(peak_lag), 2),
                    "p": round(float(p), 4)}
        print(f"{emo:11s} {peak_r:+7.3f} {peak_lag:+7.2f} {p:7.4f}")
    json.dump({"model": args.model, "results": out},
              open(f"audio/gervais_xcorr_{args.tag}.json", "w"), indent=1)
    print(f"\nsaved -> audio/gervais_xcorr_{args.tag}.json")
    print("peak r>0 at lag>0 with p<.05 => the emotion rises ~that many seconds BEFORE the laughs")


if __name__ == "__main__":
    main()
