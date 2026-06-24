"""Multivariate regression: do a model's emotion signals JOINTLY predict the
laughter-strength-over-time signal in the Gervais clip?

Unlike gervais_xcorr.py (one emotion at a time) this fits ALL monitored emotions
together, optionally with leading lags (emotion at t-lag predicts strength at t).

Both signals are heavily autocorrelated and this is a single clip, so we DON'T
trust in-sample R^2 or OLS p-values. Instead:
  * Ridge regression with blocked (contiguous) time-series CV -> out-of-sample R^2.
  * Circular-shift permutation null on strength -> p for the CV R^2.
  * Full-data ridge coefficients (standardized) to read which emotions carry signal.

Run: .venv-es/bin/python gervais_regress.py --model google/gemma-2-9b-it --tag gemma-2-9b-it
"""
from __future__ import annotations
import argparse, json
import numpy as np, torch
from bisect import bisect_right
from sklearn.linear_model import RidgeCV
from emotion_scope import load_model
import emotionscope_humor as H

MONITOR = ["amused", "playful", "delighted", "mirthful", "surprised", "curious", "bored"]
LAGS_S = [0.0, 0.5, 1.0, 1.5, 2.0]   # seconds the emotion LEADS the laughter
N_FOLDS = 5
N_PERM = 1000
ALPHAS = np.logspace(-2, 4, 25)


def read_word_scores(model, tok, dev, spoken, words, vectors, layer):
    """Per-word z-scored cosine for each monitored emotion (same as xcorr)."""
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
            j = min(max(0, bisect_right(wstart, (cs + ce) // 2) - 1), n - 1)
            s[j] += z[t]; c[j] += 1
        wv = s / np.where(c > 0, c, 1)
        out[emo] = (wv - wv.mean()) / (wv.std() + 1e-8)
    return out


def build_design(ws, wt, gt, dt):
    """Columns = emotion x lag. emotion(t - lag) aligned to strength(t)."""
    cols, names = [], []
    for emo in MONITOR:
        em_t = np.interp(gt, wt, ws[emo])          # emotion on the 0.25s grid
        for lag in LAGS_S:
            k = int(round(lag / dt))
            shifted = np.concatenate([np.full(k, em_t[0]), em_t[:len(em_t) - k]]) if k else em_t
            cols.append((shifted - shifted.mean()) / (shifted.std() + 1e-8))
            names.append(f"{emo}@{lag:g}s")
    return np.column_stack(cols), names


def blocked_cv_r2(X, y, n_folds=N_FOLDS):
    """Out-of-sample R^2 with contiguous folds (respects autocorrelation)."""
    n = len(y); bounds = np.linspace(0, n, n_folds + 1).astype(int)
    pred = np.full(n, np.nan)
    for f in range(n_folds):
        te = np.zeros(n, bool); te[bounds[f]:bounds[f + 1]] = True; tr = ~te
        if tr.sum() < 10 or te.sum() < 2:
            continue
        mdl = RidgeCV(alphas=ALPHAS).fit(X[tr], y[tr])
        pred[te] = mdl.predict(X[te])
    m = np.isfinite(pred)
    ss_res = np.sum((y[m] - pred[m]) ** 2)
    ss_tot = np.sum((y[m] - y[m].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    r = np.corrcoef(y[m], pred[m])[0, 1]
    return float(r2), float(r), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--tag", default="gemma-2-9b-it")
    args = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    asr = json.load(open("audio/gervais_fatpeople_asr.json"))
    lg = json.load(open("audio/gervais_fatpeople_laughter.json"))
    words = asr["words"]; spoken = " ".join(w["w"] for w in words)
    wt = np.array([(w["start"] + w["end"]) / 2 for w in words])
    gt = np.array(lg["timeline"]["t"]); strength = np.array(lg["timeline"]["strength"])
    dt = gt[1] - gt[0]

    npz = np.load(f"data/vectors/{args.tag}.npz")
    names = [str(n) for n in npz["emotions"]]
    vectors = {n: torch.tensor(npz[f"vec_{n}"]).float() for n in names}
    layer = int(npz["probe_layer"])
    model, tok, backend, info = load_model(args.model, device=dev, dtype="bfloat16",
                                           backend="huggingface")
    ws = read_word_scores(model, tok, dev, spoken, words, vectors, layer)

    X, feat_names = build_design(ws, wt, gt, dt)
    y = strength.copy()
    print(f"\n=== ridge regression: emotions -> laughter strength ({args.tag}) ===")
    print(f"grid points={len(y)}  features={X.shape[1]} ({len(MONITOR)} emo x {len(LAGS_S)} lags)")

    cv_r2, cv_r, pred = blocked_cv_r2(X, y)

    # circular-shift permutation null on strength (preserves its autocorrelation)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(N_PERM):
        sh = rng.integers(20, len(y) - 20)
        r2p, _, _ = blocked_cv_r2(X, np.roll(y, sh))
        null.append(r2p)
    null = np.array(null)
    p = (np.sum(null >= cv_r2) + 1) / (N_PERM + 1)

    # full-data ridge for interpretable standardized coefficients
    full = RidgeCV(alphas=ALPHAS).fit(X, y)
    coefs = full.coef_
    # collapse lags -> per-emotion strongest lag
    by_emo = {}
    for emo in MONITOR:
        idx = [i for i, fn in enumerate(feat_names) if fn.startswith(emo + "@")]
        best = max(idx, key=lambda i: abs(coefs[i]))
        by_emo[emo] = {"coef": round(float(coefs[best]), 3),
                       "lag_s": float(feat_names[best].split("@")[1][:-1])}

    print(f"\nblocked-CV out-of-sample R^2 = {cv_r2:+.3f}   (Pearson r = {cv_r:+.3f})")
    print(f"circular-shift permutation p = {p:.4f}   (null R^2 mean={null.mean():+.3f}, "
          f"95th={np.percentile(null,95):+.3f})")
    print(f"\nstandardized ridge coefficients (best lag per emotion):")
    print(f"{'emotion':11s} {'coef':>7s} {'lag(s)':>7s}")
    for emo in MONITOR:
        print(f"{emo:11s} {by_emo[emo]['coef']:+7.3f} {by_emo[emo]['lag_s']:+7.2f}")

    out = {"model": args.model, "tag": args.tag, "n_grid": int(len(y)),
           "n_features": int(X.shape[1]), "lags_s": LAGS_S,
           "cv_r2": round(cv_r2, 4), "cv_pearson_r": round(cv_r, 4),
           "perm_p": round(float(p), 4),
           "null_r2_mean": round(float(null.mean()), 4),
           "null_r2_p95": round(float(np.percentile(null, 95)), 4),
           "coef_by_emotion": by_emo}
    json.dump(out, open(f"audio/gervais_regress_{args.tag}.json", "w"), indent=1)
    print(f"\nsaved -> audio/gervais_regress_{args.tag}.json")
    print("CV R^2>0 with perm p<.05 => emotions jointly carry real, generalizing signal about laugh size")


if __name__ == "__main__":
    main()
