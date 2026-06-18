"""Monitor a family of humor-related emotion vectors across a stand-up routine
using EmotionScope (github.com/AidanZach/EmotionScope), a faithful open-source
replication of Anthropic's "Emotion Concepts and their Function in a Large
Language Model".

We extend EmotionScope's core 20 emotions with a humor family
(amused / playful / delighted / mirthful) and `bored`, extract all of them with
EmotionScope's pipeline (content-token diff-of-means, grand-mean contrast,
neutral-PCA denoise, L2 norm), then read each emotion's cosine activation
per-token across a comedy transcript and ask which emotions track the laughter.

Monitored:  amused, playful, delighted, mirthful, surprised, curious, bored
(`surprised` and `curious` already ship in EmotionScope's core 20.)

Run inside the EmotionScope env:  .venv-es/bin/python emotionscope_humor.py
"""
from __future__ import annotations
import argparse, json
import numpy as np
import torch

from emotion_scope import load_model, EmotionExtractor, EmotionProbe
from emotion_scope.config import CORE_EMOTIONS

import humor_data
import align_and_report as ar

NEW_EMOTIONS = [{"name": n, **m} for n, m in humor_data.META.items()]  # amused+family+bored
EMOTIONS_ALL = list(CORE_EMOTIONS) + NEW_EMOTIONS
MONITOR = ["amused", "playful", "delighted", "mirthful", "surprised", "curious", "bored"]


def get_layers(model):
    for path in ("model.layers", "transformer.h", "gpt_neox.layers",
                 "model.decoder.layers", "model.language_model.layers",
                 "language_model.model.layers", "blocks"):
        obj = model; ok = True
        for p in path.split("."):
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False; break
        if ok:
            return obj
    return find_decoder_layers(model)[1]


def find_decoder_layers(model):
    """Auto-locate the text decoder ModuleList (children have an attention
    submodule). Works for nested multimodal wrappers like Gemma 4."""
    import torch.nn as nn
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 10:
            child = mod[0]
            if any("attn" in n.lower() for n, _ in child.named_modules()):
                if best is None or len(mod) > len(best[1]):
                    best = (name, mod)
    if best is None:
        raise RuntimeError("could not locate decoder layer list")
    return best


def load_gemma4(model_id, device, dtype="bfloat16"):
    """Load Gemma 4 (multimodal Gemma4ForConditionalGeneration) for text-only
    residual reading and wire it so EmotionScope's `model.model.layers` hooks
    resolve to the real text decoder layers."""
    import transformers
    from transformers import AutoTokenizer, AutoConfig
    torch_dtype = getattr(torch, dtype)
    model = None
    for loader in ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModel"):
        try:
            model = getattr(transformers, loader).from_pretrained(
                model_id, dtype=torch_dtype, low_cpu_mem_usage=True)
            print(f"[gemma4] loaded {type(model).__name__} via {loader}")
            break
        except Exception as e:
            print(f"[gemma4] {loader} failed: {type(e).__name__}: {str(e)[:90]}")
    assert model is not None
    model = model.to(device).eval()

    name, layers = find_decoder_layers(model)
    print(f"[gemma4] decoder layers at {name!r}, n={len(layers)}")
    # Alias so EmotionScope's _get_layers_module ('model.layers') finds them.
    if hasattr(model, "model") and not hasattr(model.model, "layers"):
        model.model.layers = layers
    elif not hasattr(model, "model"):
        class _Shim:  # minimal holder
            pass
        model.model = _Shim(); model.model.layers = layers

    tc = AutoConfig.from_pretrained(model_id).text_config
    n_layers, d_model = tc.num_hidden_layers, tc.hidden_size
    from emotion_scope.config import ExtractionConfig
    probe_layer = round(n_layers * ExtractionConfig().probe_layer_fraction)
    tok = AutoTokenizer.from_pretrained(model_id)
    info = {"n_layers": n_layers, "d_model": d_model, "probe_layer": probe_layer,
            "model_name": model_id, "backend": "huggingface"}
    print(f"[gemma4] n_layers={n_layers} d_model={d_model} probe_layer={probe_layer}")
    return model, tok, "huggingface", info


@torch.no_grad()
def read_per_token(model, token_ids, dir_matrix, layer, device,
                   window=512, overlap=64):
    """Per-token cosine of the residual stream (at `layer`) against EACH row of
    `dir_matrix` (n_emotions, d). Returns (n_emotions, n_tokens). Windowed."""
    layers = get_layers(model)
    cap = {}

    def hook(_m, _i, o):
        cap["a"] = (o[0] if isinstance(o, tuple) else o).detach()

    h = layers[layer].register_forward_hook(hook)
    D = dir_matrix.to(device).float()
    D = D / (D.norm(dim=1, keepdim=True) + 1e-8)          # unit directions
    n = len(token_ids)
    scores = np.full((D.shape[0], n), np.nan, dtype=np.float32)
    step = window - overlap
    try:
        for start in range(0, n, step):
            end = min(start + window, n)
            inp = torch.tensor([token_ids[start:end]], device=device)
            model(inp)
            r = cap["a"][0].float()                        # (len, d)
            rn = r / (r.norm(dim=-1, keepdim=True) + 1e-8)
            proj = (rn @ D.T).cpu().numpy()                # (len, n_emotions)
            core = 0 if start == 0 else overlap
            for j in range(core, proj.shape[0]):
                scores[:, start + j] = proj[j]
            if end == n:
                break
    finally:
        h.remove()
    return scores


def sanity_check(model, tok, backend, info, vectors):
    probe = EmotionProbe(model, tok, backend, info, vectors,
                         emotions_metadata=EMOTIONS_ALL)
    tests = [
        "So a horse walks into a bar and the bartender says, why the long face.",
        "My uncle fought a vending machine over a Twix and lost by a knockout.",
        "He stared at the clock and watched the second hand crawl through the dull afternoon.",
        "Wait... what is that noise coming from inside the locked cupboard?",
    ]
    print("\n--- sanity: top-3 emotions per probe ---")
    for t in tests:
        st = probe.analyze(t)
        top = ", ".join(f"{n}:{s:+.2f}" for n, s in st.top_emotions[:3])
        print(f"  {top}   | {t[:46]!r}")


def multi_plot(scores_by_emotion, laugh_pos, out_path, title, model_tag, layer):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    half = 40
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5),
                             gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(scores_by_emotion)))
    rows = []
    for (emo, sc), c in zip(scores_by_emotion.items(), colors):
        offsets, mean_c, sem_c, n_used = ar.peri_laughter_average(sc, laugh_pos, half=half)
        ax.plot(offsets, mean_c, lw=2, color=c, label=emo)
        ax.fill_between(offsets, mean_c - sem_c, mean_c + sem_c, color=c, alpha=0.12)
        det = ar.detection_auc(sc, laugh_pos, pre_window=12)
        perm = ar.permutation_test(sc, laugh_pos, pre_window=12, n_perm=3000)
        rows.append((emo, det, perm["z_above_null"], perm["p_value"], float(mean_c[half])))
    ax.axvline(0, color="k", lw=1.2, ls="--", label="laugh onset")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("tokens relative to laugh onset")
    ax.set_ylabel("emotion activation (z)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    # bar chart: detection AUC per emotion (how well each predicts the laugh run-up)
    ax2 = axes[1]
    rows.sort(key=lambda r: r[1])
    emos = [r[0] for r in rows]; aucs = [r[1] for r in rows]
    bars = ax2.barh(emos, aucs, color=["#2ca02c" if a >= 0.5 else "#d62728" for a in aucs])
    ax2.axvline(0.5, color="k", lw=1, ls="--", label="chance")
    ax2.set_xlabel("detection AUC (pre-laugh tokens vs rest)")
    ax2.set_title(f"Which emotions predict the laugh?\n{model_tag}, layer {layer}")
    for b, r in zip(bars, rows):
        ax2.text(b.get_width() + 0.002, b.get_y() + b.get_height() / 2,
                 f"{r[1]:.3f} (p={r[3]:.3f})", va="center", fontsize=8)
    ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"saved plot -> {out_path}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--transcript", default="data/bill-burr-drop-dead-years.txt")
    ap.add_argument("--name", default="Bill Burr — Drop Dead Years")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--dtype", default="bfloat16",
                    help="float32 ok for <=3B; use bfloat16 for larger models")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    from transformers import AutoConfig
    multimodal = getattr(AutoConfig.from_pretrained(args.model), "text_config", None) is not None
    if multimodal:  # Gemma 4, Qwen3.5/3.6 — nested text decoder, needs the adapter
        model, tok, backend, info = load_gemma4(args.model, device, dtype="bfloat16")
    else:
        model, tok, backend, info = load_model(
            args.model, device=device, dtype=args.dtype, backend="huggingface")

    extractor = EmotionExtractor(model, tok, backend, info, emotions=EMOTIONS_ALL)
    if args.sweep:
        extractor.find_best_probe_layer()
        info["probe_layer"] = extractor.probe_layer
    layer = info["probe_layer"]
    print(f"\nExtracting {len(EMOTIONS_ALL)} emotion vectors at layer {layer}/{info['n_layers']} ...")
    vectors = extractor.extract()
    missing = [e["name"] for e in NEW_EMOTIONS if e["name"] not in vectors]
    if missing:
        print("WARNING missing vectors:", missing)
    sanity_check(model, tok, backend, info, vectors)

    # Read cosine to ALL emotion vectors so we can remove the common-mode
    # (sentence-boundary/punctuation) component that otherwise inflates EVERY
    # emotion at the punchline. The specific signal is each emotion MINUS the
    # per-token mean across all emotions.
    all_names = [e["name"] for e in EMOTIONS_ALL if e["name"] in vectors]
    dir_matrix = torch.stack([vectors[n].cpu().float() for n in all_names])  # (25, d)

    text = open(args.transcript).read()
    token_ids, token_strs, laugh_pos = ar.build_token_stream(text, tok)
    print(f"\nTranscript: {len(token_ids)} spoken tokens, {len(laugh_pos)} laughs")

    raw_all = read_per_token(model, token_ids, dir_matrix, layer, device)  # (25, N)
    # winsorize each row, then common mode = mean across all emotions per token
    for i in range(raw_all.shape[0]):
        r = raw_all[i]; fin = r[np.isfinite(r)]
        lo, hi = np.percentile(fin, [0.5, 99.5]); raw_all[i] = np.clip(r, lo, hi)
    common = np.nanmean(raw_all, axis=0)                       # (N,)
    idx = {n: i for i, n in enumerate(all_names)}

    raw_by_emotion = {e: raw_all[idx[e]] for e in MONITOR if e in idx}
    rel_by_emotion = {e: raw_all[idx[e]] - common for e in MONITOR if e in idx}

    print("\n=== per-emotion laughter alignment (RAW vs common-mode-removed) ===")
    print(f"  {'emotion':11s} {'raw AUC':>8s} {'rawZ':>6s}   {'rel AUC':>8s} {'relZ':>6s} {'rel p':>7s}")
    summary = {}
    for emo in MONITOR:
        if emo not in idx:
            continue
        rA = ar.detection_auc(raw_by_emotion[emo], laugh_pos, 12)
        rZ = ar.permutation_test(raw_by_emotion[emo], laugh_pos, 12)["z_above_null"]
        cA = ar.detection_auc(rel_by_emotion[emo], laugh_pos, 12)
        cp = ar.permutation_test(rel_by_emotion[emo], laugh_pos, 12)
        summary[emo] = {"raw_auc": round(rA, 3), "raw_z": round(rZ, 2),
                        "rel_auc": round(cA, 3), "rel_z": round(cp["z_above_null"], 2),
                        "rel_p": round(cp["p_value"], 4)}
        print(f"  {emo:11s} {rA:8.3f} {rZ:+6.2f}   {cA:8.3f} {cp['z_above_null']:+6.2f} {cp['p_value']:7.4f}")

    slug = args.transcript.split("/")[-1].replace(".txt", "")
    mtag = args.model.split("/")[-1]
    title = (f"Humor-family emotion vectors vs. laughter (common-mode removed)\n"
             f"{args.name}  —  {mtag}, layer {layer}")
    multi_plot(rel_by_emotion, laugh_pos,
               f"data/{slug}_humorfamily_{mtag}.png", title, mtag, layer)
    np.savez(f"data/{slug}_humorfamily_{mtag}.npz",
             **{f"raw_{e}": s for e, s in raw_by_emotion.items()},
             **{f"rel_{e}": s for e, s in rel_by_emotion.items()},
             laugh_positions=np.array(laugh_pos), probe_layer=layer)
    json.dump(summary, open(f"data/{slug}_humorfamily_{mtag}_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
