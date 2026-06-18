"""Open-source replication of Anthropic's "emotion vectors" method
(https://www.anthropic.com/research/emotion-concepts-function),
following the difference-of-means recipe used by the traitinterp replication
(github.com/ewernn/traitinterp).

An *emotion vector* for emotion E at layer L is:

    v_E,L  =  mean_{x in POS} h_L(x)  -  mean_{x in NEG} h_L(x)

where h_L(x) is the residual-stream activation at layer L, mean-pooled over the
content tokens of a short passage x.  POS passages express/evoke the emotion,
NEG passages are neutral matched controls.

To *read* the emotion over a new text we project each token's residual stream
onto the (unit-normalised) emotion direction:

    score_t = h_L(token_t) . v_hat_E,L

This module provides the model wrapper + activation extraction + vector build.
"""
from __future__ import annotations
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Reader:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.device = device or pick_device()
        # float32 on MPS is the most numerically reliable for activation reads.
        self.dtype = torch.float32
        print(f"Loading {model_name} on {self.device} ({self.dtype}) ...")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=self.dtype, output_hidden_states=True
        ).to(self.device).eval()
        self.n_layers = self.model.config.num_hidden_layers
        self.d_model = self.model.config.hidden_size
        print(f"  n_layers={self.n_layers}  d_model={self.d_model}")

    @torch.no_grad()
    def mean_pooled_activations(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        """Return [n_texts, n_layers+1, d_model]: residual stream mean-pooled
        over real (non-pad) tokens, for every layer (incl. embeddings = idx 0)."""
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tok(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=128).to(self.device)
            hs = self.model(**enc).hidden_states  # tuple len n_layers+1, each [B,T,d]
            mask = enc.attention_mask.unsqueeze(-1).float()  # [B,T,1]
            denom = mask.sum(1).clamp(min=1)                 # [B,1]
            layer_means = []
            for h in hs:
                pooled = (h * mask).sum(1) / denom           # [B,d]
                layer_means.append(pooled.float().cpu().numpy())
            out.append(np.stack(layer_means, axis=1))        # [B, n_layers+1, d]
        return np.concatenate(out, axis=0)

    @torch.no_grad()
    def token_projections(self, token_ids: list[int], direction: np.ndarray,
                          layer: int, window: int = 768, overlap: int = 128) -> np.ndarray:
        """Project every token's residual stream (at `layer`) onto `direction`.
        Runs in overlapping windows so long transcripts fit in memory; only the
        non-overlapping core of each window is kept. Returns [n_tokens] float32."""
        d = torch.tensor(direction, dtype=self.dtype, device=self.device)
        d = d / d.norm()
        bos = self.tok.bos_token_id
        n = len(token_ids)
        scores = np.full(n, np.nan, dtype=np.float32)
        step = window - overlap
        for start in range(0, n, step):
            end = min(start + window, n)
            ids = token_ids[start:end]
            if bos is not None:
                inp = torch.tensor([[bos] + ids], device=self.device)
                off = 1
            else:
                inp = torch.tensor([ids], device=self.device)
                off = 0
            h = self.model(inp).hidden_states[layer][0]       # [off+len, d]
            proj = (h @ d).float().cpu().numpy()[off:]        # [len]
            # keep only the core (drop the leading overlap except for first window)
            core_start = 0 if start == 0 else overlap
            for j in range(core_start, len(proj)):
                scores[start + j] = proj[j]
            if end == n:
                break
        return scores


def build_emotion_vector(reader: Reader, pos: list[str], neg: list[str],
                         seed: int = 0):
    """Difference-of-means emotion vector at every layer + held-out layer pick.

    Returns dict with: vectors [n_layers+1, d], best_layer (int),
    layer_auc [n_layers+1], and pos/neg projection stats at best layer.
    """
    rng = np.random.default_rng(seed)
    Ap = reader.mean_pooled_activations(pos)   # [np, Lp1, d]
    An = reader.mean_pooled_activations(neg)   # [nn, Lp1, d]
    n_layers = Ap.shape[1]

    # Held-out split to choose the layer that best separates pos/neg.
    def split(idx):
        idx = rng.permutation(idx)
        k = max(1, int(0.7 * len(idx)))
        return idx[:k], idx[k:]
    ptr, pte = split(np.arange(len(pos)))
    ntr, nte = split(np.arange(len(neg)))

    vectors = np.zeros((n_layers, Ap.shape[2]), dtype=np.float32)
    aucs = np.zeros(n_layers, dtype=np.float32)
    for L in range(n_layers):
        v = Ap[ptr, L].mean(0) - An[ntr, L].mean(0)
        vectors[L] = v
        vh = v / (np.linalg.norm(v) + 1e-8)
        sp = Ap[pte, L] @ vh
        sn = An[nte, L] @ vh
        aucs[L] = auc(sp, sn)
    best_layer = int(np.argmax(aucs))

    vh = vectors[best_layer] / (np.linalg.norm(vectors[best_layer]) + 1e-8)
    pos_proj = Ap[:, best_layer] @ vh
    neg_proj = An[:, best_layer] @ vh
    return {
        "vectors": vectors,
        "best_layer": best_layer,
        "layer_auc": aucs,
        "pos_proj": pos_proj,
        "neg_proj": neg_proj,
        # normalisation reference: midpoint & spread of the contrastive set
        "ref_mid": float((pos_proj.mean() + neg_proj.mean()) / 2),
        "ref_scale": float(np.std(np.concatenate([pos_proj, neg_proj])) + 1e-8),
    }


def auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Mann-Whitney AUC: P(pos score > neg score)."""
    pos_scores = np.asarray(pos_scores); neg_scores = np.asarray(neg_scores)
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5
    wins = 0.0
    for s in pos_scores:
        wins += np.sum(s > neg_scores) + 0.5 * np.sum(s == neg_scores)
    return wins / (len(pos_scores) * len(neg_scores))
