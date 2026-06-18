"""E1 — conditional / in-context epiplexity, per token, vs audience laughter.

Epiplexity (Finzi et al. 2026, arXiv:2601.03220) splits the information in data
into a *structured / learnable* part S_T (what a computationally bounded observer
can compress) and *irreducible entropy* H_T. The paper's practical estimator is
the prequential code length: the area between a less-adapted and a more-adapted
observer's loss curve.

With a single FIXED pretrained model the faithful analog is the **in-context
compression gain**: give the model a short local context vs the full preceding
context, and the per-token nats the long context saves = structured information
it absorbed by reading the setup.

    s_short(t) = -log p(x_t | last K tokens)        weak/bounded observer  (total)
    s_full(t)  = -log p(x_t | all prior tokens)     adapted observer       (residual H_T)
    E1(t)      = s_short(t) - s_full(t)              structured info / epiplexity (S_T)

Sign reading for humor: at a genuine incongruity (the punchline word the setup
made *less* likely) s_full spikes and E1 can dip; the resolvable structure of a
joke (premises, callbacks, running bits) shows up as positive E1. We let the
data show which.

The [LAUGHTER] markers are stripped before the model sees the text (no leakage);
we keep the token index each laugh follows and ask whether E1 rises into it.

Run:  .venv/bin/python epiplexity_e1.py \
        --transcript data/bill-burr-drop-dead-years.txt --name "Bill Burr — Drop Dead Years"
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import align_and_report as ar

MODEL = "Qwen/Qwen2.5-1.5B"   # base, not instruct: instruct-tuning distorts next-token probs


def device():
    return "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")


def load(model_id):
    dev = device()
    print(f"Loading {model_id} on {dev} ...")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32).to(dev).eval()
    return model, tok, dev


def start_token(tok):
    return tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id


@torch.no_grad()
def nll_full(model, tok, ids, dev, window=1024, overlap=256):
    """Per-token -log p(x_t | all prior), via overlapping sliding windows so each
    scored token carries >= `overlap` tokens of left context (first window: all)."""
    bos = start_token(tok)
    n = len(ids)
    out = np.full(n, np.nan, np.float32)
    step = window - overlap
    for start in range(0, n, step):
        end = min(start + window, n)
        chunk = ids[start:end]
        seq = ([bos] + chunk) if bos is not None else chunk
        off = 1 if bos is not None else 0
        inp = torch.tensor([seq], device=dev)
        logp = torch.log_softmax(model(inp).logits[0].float(), dim=-1)  # (L, V)
        core = 0 if start == 0 else overlap
        for j in range(core, len(chunk)):
            si = off + j                  # seq index of chunk[j]
            if si - 1 < 0:                # no predictor for the very first token
                continue
            out[start + j] = -logp[si - 1, chunk[j]].item()
        if end == n:
            break
    return out


@torch.no_grad()
def nll_short(model, tok, ids, dev, C=16, batch=64):
    """Per-token -log p(x_t | last C tokens). Each token's window is short, so we
    score only the final position; windows are grouped by length (<=C+1) to avoid
    any padding, and batched."""
    bos = start_token(tok)
    n = len(ids)
    out = np.full(n, np.nan, np.float32)
    by_len = defaultdict(list)            # len(seq) -> list of (seq, target, t)
    for t in range(n):
        ctx = ids[max(0, t - C):t]
        seq = ([bos] + ctx) if bos is not None else (ctx if ctx else [bos])
        by_len[len(seq)].append((seq, ids[t], t))
    for L, items in by_len.items():
        for b in range(0, len(items), batch):
            grp = items[b:b + batch]
            inp = torch.tensor([s for s, _, _ in grp], device=dev)
            logp = torch.log_softmax(model(inp).logits[:, -1, :].float(), dim=-1)
            for r, (_, target, t) in enumerate(grp):
                out[t] = -logp[r, target].item()
    return out


def winsorize(x, lo=0.5, hi=99.5):
    fin = x[np.isfinite(x)]
    a, b = np.percentile(fin, [lo, hi])
    return np.clip(x, a, b)


def report(name, scores, laugh_pos, pre_window):
    perm = ar.permutation_test(scores, laugh_pos, pre_window=pre_window)
    det = ar.detection_auc(scores, laugh_pos, pre_window=pre_window)
    print(f"\n--- {name} ---")
    print(f"  punchline-token (offset 0):", end=" ")
    off, mean_c, sem_c, nU = ar.peri_laughter_average(scores, laugh_pos, half=40)
    z0 = mean_c[len(mean_c) // 2]
    print(f"z={z0:+.2f}  ({nU} laughs)")
    print(f"  {pre_window}-tok run-up: observed z={perm['observed_pre_laugh_z']:+.3f}, "
          f"{perm['z_above_null']:+.1f}σ above null, p={perm['p_value']:.4f}")
    print(f"  detection AUC (pre-laugh vs rest): {det:.3f}")
    return perm, det, (off, mean_c, sem_c, nU)


def plot(s_full, s_short, e1, laugh_pos, out_path, title, pre_window):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(e1)
    ze = ar.zscore(e1)
    sm = ar.smooth(ze, k=31)
    fig, axes = plt.subplots(2, 2, figsize=(17, 9),
                             gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0, 0]
    ax.plot(np.arange(n), sm, lw=0.9, color="#6a3d9a", label="epiplexity E1 (z, smoothed)")
    ax.fill_between(np.arange(n), sm, 0, where=sm > 0, color="#6a3d9a", alpha=0.15)
    for j, p in enumerate(laugh_pos):
        ax.axvline(p, color="#d62728", alpha=0.25, lw=0.6,
                   label="audience laughter" if j == 0 else None)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("token index (comedian's words only — laughter stripped before model)")
    ax.set_ylabel("E1 epiplexity (z)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, n)

    # peri-laughter panels for E1 and raw surprisal, on the same axes
    ax2 = axes[0, 1]
    for scores, color, lbl in [(e1, "#6a3d9a", "E1 epiplexity"),
                               (s_full, "#1f77b4", "raw surprisal s_full"),
                               (s_short, "#ff7f0e", "weak-obs s_short")]:
        off, mc, sc, nU = ar.peri_laughter_average(scores, laugh_pos, half=40)
        ax2.plot(off, mc, lw=1.8, color=color, label=lbl)
        ax2.fill_between(off, mc - sc, mc + sc, color=color, alpha=0.12)
    ax2.axvline(0, color="#d62728", lw=1.2, label="laugh onset")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_xlabel("tokens relative to laugh onset")
    ax2.set_ylabel("mean signal (z)")
    ax2.set_title("Peri-laughter averages")
    ax2.legend(loc="upper left", fontsize=8)

    # zoom: a 600-token slice with the most laughs, all three signals
    win = 600
    if laugh_pos:
        counts = [sum(1 for p in laugh_pos if s <= p < s + win) for s in range(0, n - win, 50)]
        zs = int(np.argmax(counts) * 50) if counts else 0
    else:
        zs = 0
    ze_ = ar.smooth(ar.zscore(e1), 9)
    zf_ = ar.smooth(ar.zscore(s_full), 9)
    sl = slice(zs, zs + win)
    axz = axes[1, 0]
    axz.plot(np.arange(zs, zs + win), ze_[sl], color="#6a3d9a", lw=1.0, label="E1")
    axz.plot(np.arange(zs, zs + win), zf_[sl], color="#1f77b4", lw=0.8, alpha=0.7, label="raw surprisal")
    for p in laugh_pos:
        if zs <= p < zs + win:
            axz.axvline(p, color="#d62728", alpha=0.4, lw=0.8)
    axz.axhline(0, color="k", lw=0.5)
    axz.set_title(f"zoom: tokens {zs}–{zs+win} (densest-laughter slice)")
    axz.set_xlabel("token index")
    axz.legend(loc="upper right", fontsize=8)

    axh = axes[1, 1]
    axh.hist(e1[np.isfinite(e1)], bins=80, color="#6a3d9a", alpha=0.8)
    axh.axvline(0, color="k", lw=0.8)
    axh.set_title("E1 distribution (nats: +structure / -incongruity)")
    axh.set_xlabel("E1 = s_short - s_full (nats)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"\nsaved plot -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", default="data/bill-burr-drop-dead-years.txt")
    ap.add_argument("--name", default="Bill Burr — Drop Dead Years")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--short-ctx", type=int, default=16)
    ap.add_argument("--pre-window", type=int, default=12)
    args = ap.parse_args()

    model, tok, dev = load(args.model)
    text = open(args.transcript).read()
    ids, token_strs, laugh_pos = ar.build_token_stream(text, tok)
    print(f"Transcript: {len(ids)} spoken tokens, {len(laugh_pos)} laughs")

    print("computing s_full (full-context surprisal) ...")
    s_full = nll_full(model, tok, ids, dev)
    print(f"computing s_short (K={args.short_ctx} context) ...")
    s_short = nll_short(model, tok, ids, dev, C=args.short_ctx)
    e1 = s_short - s_full            # structured / learnable information (nats)

    s_full = winsorize(s_full); s_short = winsorize(s_short); e1 = winsorize(e1)

    print("\n=== does it rise into the laughs? (z above a permutation null) ===")
    report("E1 epiplexity (structured info)", e1, laugh_pos, args.pre_window)
    report("raw surprisal s_full (H1 baseline)", s_full, laugh_pos, args.pre_window)
    report("weak-observer surprisal s_short", s_short, laugh_pos, args.pre_window)

    slug = args.transcript.split("/")[-1].replace(".txt", "")
    out_png = f"data/{slug}_epiplexity_e1.png"
    title = (f"E1 in-context epiplexity vs audience laughter — {args.name}\n"
             f"{args.model}, short-ctx K={args.short_ctx}  "
             f"(E1 = -log p(x|last K) + log p(x|all prior))")
    plot(s_full, s_short, e1, laugh_pos, out_png, title, args.pre_window)

    np.savez(f"data/{slug}_epiplexity_e1.npz", s_full=s_full, s_short=s_short,
             e1=e1, laugh_positions=np.array(laugh_pos), short_ctx=args.short_ctx)

    z = ar.zscore(e1)
    order = np.argsort(-np.nan_to_num(z))[:25]
    print("\nTop-25 highest-epiplexity tokens (z):")
    for i in sorted(order):
        near = any(0 <= p - i < 15 for p in laugh_pos)
        ctx = "".join(token_strs[max(0, i - 6):i + 1])
        print(f"  tok {i:5d}  E1z={z[i]:+.2f}  {'LAUGH<=15' if near else '         '}  …{ctx!r}")


if __name__ == "__main__":
    main()
