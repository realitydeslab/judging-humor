"""Monitor an 'amusement / funniness' emotion vector across a stand-up routine
and compare it to where the audience actually laughed.

Emotion-vector machinery: `steering-vectors` (github.com/steering-vectors),
the difference-of-means recipe behind Anthropic's "emotion concepts" research.
We do NOT hand-roll the vector: `train_steering_vector` builds the direction
from contrastive (funny, neutral) text pairs, and `record_activations` gives us
the per-token residual stream to project onto it.

Laughter alignment / stats / plotting: align_and_report.py (this repo).
The [LAUGHTER] markers are stripped before the model sees the text, so the test
is honest: does the model's internal amusement signal rise into the real laughs?
"""
from __future__ import annotations
import argparse, json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import train_steering_vector, record_activations, mean_aggregator

import emotion_data
import align_and_report as ar

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def device():
    return "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")


def load():
    dev = device()
    print(f"Loading {MODEL} on {dev} ...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32).to(dev).eval()
    return model, tok, dev


def split_idx(n, frac=0.7, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    k = max(1, int(frac * n))
    return idx[:k], idx[k:]


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    allv = np.concatenate([pos, neg]); order = allv.argsort()
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


@torch.no_grad()
def last_token_acts(model, tok, dev, texts, layers):
    """Last-token residual stream at each layer for each text -> {L: [n, d]}."""
    out = {L: [] for L in layers}
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=128).to(dev)
        with record_activations(model, layer_type="decoder_block",
                                layer_nums=list(layers)) as rec:
            model(**enc)
        for L in layers:
            out[L].append(rec[L][0][0, -1].float().cpu().numpy())
    return {L: np.array(v) for L, v in out.items()}


def build_amusement_vector(model, tok, dev):
    """Train the diff-of-means direction (steering-vectors) and pick the layer
    that best separates held-out funny vs. neutral text by projection AUC."""
    pos, neg = emotion_data.POS, emotion_data.NEG
    n_layers = model.config.num_hidden_layers
    layers = list(range(n_layers))

    # held-out split for honest layer selection
    ptr, pte = split_idx(len(pos)); ntr, nte = split_idx(len(neg))
    pairs_tr = [(pos[i], neg[j]) for i, j in zip(ptr, ntr)]

    print(f"Training steering vector on {len(pairs_tr)} contrastive pairs "
          f"across {n_layers} layers ...")
    sv_tr = train_steering_vector(model, tok, pairs_tr, layers=layers,
                                  layer_type="decoder_block",
                                  read_token_index=-1, aggregator=mean_aggregator(),
                                  show_progress=True)

    # evaluate per-layer separation on held-out funny/neutral passages
    te_pos = last_token_acts(model, tok, dev, [pos[i] for i in pte], layers)
    te_neg = last_token_acts(model, tok, dev, [neg[i] for i in nte], layers)
    aucs = {}
    for L in layers:
        d = sv_tr.layer_activations[L].float().cpu().numpy()
        d = d / (np.linalg.norm(d) + 1e-8)
        aucs[L] = auc(te_pos[L] @ d, te_neg[L] @ d)
    # Only consider semantically meaningful mid/late layers; if AUC ties at the
    # top, prefer the layer nearest ~0.6 depth (concepts live mid-to-late, not in
    # the surface-feature early layers).
    cand = list(range(max(1, n_layers // 3), n_layers - 1))
    best_auc = max(aucs[L] for L in cand)
    top = [L for L in cand if aucs[L] >= best_auc - 1e-9]
    target = int(round(0.6 * n_layers))
    best_layer = min(top, key=lambda L: abs(L - target))
    print(f"layer search restricted to {cand[0]}..{cand[-1]}; "
          f"top AUC {best_auc:.3f} at layers {top}; picked {best_layer}")
    print("per-layer held-out AUC (funny vs vivid-but-not-funny):")
    for L in layers:
        mark = "  <== best" if L == best_layer else ""
        print(f"  layer {L:2d}: AUC={aucs[L]:.3f}{mark}")

    # retrain on ALL pairs for the final, strongest direction
    all_pairs = [(pos[i], neg[i]) for i in range(min(len(pos), len(neg)))]
    sv_all = train_steering_vector(model, tok, all_pairs, layers=[best_layer],
                                   layer_type="decoder_block",
                                   read_token_index=-1, aggregator=mean_aggregator())
    direction = sv_all.layer_activations[best_layer].float().to(dev)
    direction = direction / direction.norm()
    return direction, best_layer, aucs


@torch.no_grad()
def read_per_token(model, tok, dev, token_ids, direction, layer,
                   window=768, overlap=128):
    """Per-token projection of the residual stream (at `layer`) onto `direction`,
    computed in overlapping windows so long transcripts fit in memory."""
    bos = tok.bos_token_id
    n = len(token_ids)
    scores = np.full(n, np.nan, dtype=np.float32)
    step = window - overlap
    for start in range(0, n, step):
        end = min(start + window, n)
        ids = token_ids[start:end]
        seq = ([bos] + ids) if bos is not None else ids
        off = 1 if bos is not None else 0
        inp = torch.tensor([seq], device=dev)
        with record_activations(model, layer_type="decoder_block",
                                layer_nums=[layer]) as rec:
            model(inp)
        h = rec[layer][0][0]                       # [off+len, d]
        proj = (h.float() @ direction).cpu().numpy()[off:]
        core = 0 if start == 0 else overlap
        for j in range(core, len(proj)):
            scores[start + j] = proj[j]
        if end == n:
            break
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", default="data/bill-burr-drop-dead-years.txt")
    ap.add_argument("--name", default="Bill Burr — Drop Dead Years")
    ap.add_argument("--pre-window", type=int, default=12)
    args = ap.parse_args()

    model, tok, dev = load()
    direction, best_layer, aucs = build_amusement_vector(model, tok, dev)

    text = open(args.transcript).read()
    token_ids, token_strs, laugh_pos = ar.build_token_stream(text, tok)
    print(f"\nTranscript: {len(token_ids)} spoken tokens, {len(laugh_pos)} laughs")

    scores = read_per_token(model, tok, dev, token_ids, direction, best_layer)
    # winsorize attention-sink / first-token outliers so they don't dominate z-scaling
    finite = scores[np.isfinite(scores)]
    lo, hi = np.percentile(finite, [0.5, 99.5])
    scores = np.clip(scores, lo, hi)

    perm = ar.permutation_test(scores, laugh_pos, pre_window=args.pre_window)
    det_auc = ar.detection_auc(scores, laugh_pos, pre_window=args.pre_window)
    print("\n=== RESULTS ===")
    print(json.dumps(perm, indent=2))
    print(f"detection AUC (pre-laugh tokens vs rest): {det_auc:.3f}")

    slug = args.transcript.split("/")[-1].replace(".txt", "")
    out_png = f"data/{slug}_amusement.png"
    title = (f"Internal 'amusement' emotion vector vs. audience laughter\n"
             f"{args.name}  —  steering-vectors diff-of-means, Qwen2.5-1.5B, layer {best_layer}  "
             f"(detection AUC {det_auc:.2f}, p={perm['p_value']:.4f})")
    ar.plot_report(scores, laugh_pos, token_strs, out_png, title,
                   "steering-vectors", best_layer, "Qwen2.5-1.5B-Instruct")

    np.savez(f"data/{slug}_scores.npz", scores=scores,
             laugh_positions=np.array(laugh_pos), best_layer=best_layer,
             layer_auc=np.array([aucs[L] for L in sorted(aucs)]))
    # human-readable: top amused tokens
    z = ar.zscore(scores)
    order = np.argsort(-np.nan_to_num(z))[:25]
    print("\nTop-25 most 'amused' tokens (z):")
    for i in sorted(order):
        near = any(0 <= p - i < 15 for p in laugh_pos)
        print(f"  tok {i:5d}  z={z[i]:+.2f}  {'LAUGH<=15' if near else '        '}  {token_strs[i]!r}")


if __name__ == "__main__":
    main()
