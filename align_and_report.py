"""Framework-agnostic glue for the humor-monitoring experiment.

  * parse a cleaned transcript (with inline [LAUGHTER] markers) into a
    SPOKEN-ONLY token stream + the token indices where the audience laughed
    (the markers are NOT fed to the model -> no leakage).
  * given any per-token "amusement" score array aligned to that token stream,
    test + plot whether the score rises into the laughs.
"""
from __future__ import annotations
import numpy as np

LAUGH = "[LAUGHTER]"


def build_token_stream(text: str, tok):
    """Split on [LAUGHTER]; tokenize the spoken chunks only; record, for each
    laugh event, the index of the spoken token it immediately follows.

    Returns:
      token_ids        : list[int]   spoken-only token ids (no special tokens)
      token_strs       : list[str]   decoded per-token strings (for inspection)
      laugh_positions  : list[int]   token indices AFTER which a laugh occurred
    """
    parts = text.split(LAUGH)
    token_ids: list[int] = []
    laugh_positions: list[int] = []
    for i, chunk in enumerate(parts):
        chunk = chunk.strip()
        if chunk:
            ids = tok(chunk, add_special_tokens=False)["input_ids"]
            token_ids.extend(ids)
        # a laugh marker sits between chunk i and chunk i+1
        if i < len(parts) - 1:
            if len(token_ids) > 0:
                laugh_positions.append(len(token_ids) - 1)  # last spoken token before laugh
    token_strs = [tok.decode([t]) for t in token_ids]
    # de-duplicate consecutive identical laugh positions (back-to-back markers)
    laugh_positions = sorted(set(laugh_positions))
    return token_ids, token_strs, laugh_positions


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = np.nanmean(x); s = np.nanstd(x)
    return (x - m) / (s + 1e-8)


def smooth(x: np.ndarray, k: int = 15) -> np.ndarray:
    """Centered moving average, NaN-safe."""
    x = np.asarray(x, dtype=np.float64)
    valid = ~np.isnan(x)
    xf = np.where(valid, x, 0.0)
    ker = np.ones(k)
    num = np.convolve(xf, ker, mode="same")
    den = np.convolve(valid.astype(float), ker, mode="same")
    return num / np.clip(den, 1e-8, None)


def peri_laughter_average(scores: np.ndarray, laugh_positions, half: int = 40):
    """Average the (z-scored) signal in a window [-half, +half] tokens around
    each laugh onset. Returns (offsets, mean_curve, sem_curve, n)."""
    z = zscore(scores)
    offsets = np.arange(-half, half + 1)
    stack = []
    n = len(scores)
    for p in laugh_positions:
        lo, hi = p - half, p + half
        if lo < 0 or hi >= n:
            continue
        stack.append(z[lo:hi + 1])
    stack = np.array(stack)
    if len(stack) == 0:
        return offsets, np.zeros_like(offsets, float), np.zeros_like(offsets, float), 0
    mean = np.nanmean(stack, axis=0)
    sem = np.nanstd(stack, axis=0) / np.sqrt(len(stack))
    return offsets, mean, sem, len(stack)


def permutation_test(scores, laugh_positions, pre_window=12, n_perm=5000, seed=0):
    """Is the mean amusement score in the `pre_window` tokens leading up to a
    laugh higher than for random positions? Returns dict of stats."""
    rng = np.random.default_rng(seed)
    z = zscore(scores)
    n = len(z)

    def pre_mean(positions):
        vals = []
        for p in positions:
            lo = max(0, p - pre_window + 1)
            seg = z[lo:p + 1]
            seg = seg[~np.isnan(seg)]
            if len(seg):
                vals.append(seg.mean())
        return np.mean(vals) if vals else np.nan

    observed = pre_mean(laugh_positions)
    k = len(laugh_positions)
    null = np.empty(n_perm)
    for i in range(n_perm):
        fake = rng.integers(pre_window, n, size=k)
        null[i] = pre_mean(fake)
    p_val = (np.sum(null >= observed) + 1) / (n_perm + 1)
    return {
        "observed_pre_laugh_z": float(observed),
        "null_mean_z": float(np.mean(null)),
        "null_std_z": float(np.std(null)),
        "z_above_null": float((observed - np.mean(null)) / (np.std(null) + 1e-8)),
        "p_value": float(p_val),
        "n_laughs": int(k),
        "pre_window": pre_window,
    }


def detection_auc(scores, laugh_positions, pre_window=12):
    """Treat 'is this token within `pre_window` before a laugh' as the label and
    the amusement z-score as the predictor; report ROC AUC (rank-based)."""
    z = zscore(scores)
    n = len(z)
    label = np.zeros(n, dtype=bool)
    for p in laugh_positions:
        lo = max(0, p - pre_window + 1)
        label[lo:p + 1] = True
    valid = ~np.isnan(z)
    pos = z[valid & label]
    neg = z[valid & ~label]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # rank-based AUC
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    r_pos = ranks[:len(pos)].sum()
    auc = (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def plot_report(scores, laugh_positions, token_strs, out_path, title,
                framework_name, best_layer, model_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = zscore(scores)
    sm = smooth(z, k=21)
    n = len(z)
    offsets, mean_c, sem_c, n_used = peri_laughter_average(scores, laugh_positions)

    fig, axes = plt.subplots(2, 1, figsize=(15, 9),
                             gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    ax.plot(np.arange(n), sm, lw=1.0, color="#1f77b4",
            label="amusement signal (z, smoothed)")
    ax.fill_between(np.arange(n), sm, 0, where=sm > 0, color="#1f77b4", alpha=0.15)
    for j, p in enumerate(laugh_positions):
        ax.axvline(p, color="#d62728", alpha=0.30, lw=0.8,
                   label="audience laughter" if j == 0 else None)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("token index (comedian's words only — laughter markers removed before the model)")
    ax.set_ylabel("amusement (z)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, n)

    ax2 = axes[1]
    ax2.plot(offsets, mean_c, color="#2ca02c", lw=2)
    ax2.fill_between(offsets, mean_c - sem_c, mean_c + sem_c, color="#2ca02c", alpha=0.25)
    ax2.axvline(0, color="#d62728", lw=1.5, label="laugh onset")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_xlabel("tokens relative to laugh onset")
    ax2.set_ylabel("mean amusement (z)")
    ax2.set_title(f"Peri-laughter average over {n_used} laughs  "
                  f"[{framework_name}, {model_name}, layer {best_layer}]")
    ax2.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"saved plot -> {out_path}")
    return out_path
