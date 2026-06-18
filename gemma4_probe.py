"""Probe Gemma 4 E4B structure so we can wire it into EmotionScope:
find the text decoder ModuleList, dims, and confirm a text-only forward +
residual capture works.
"""
import torch, transformers
from transformers import AutoConfig, AutoTokenizer

MID = "google/gemma-4-E4B-it"
print("transformers", transformers.__version__)
cfg = AutoConfig.from_pretrained(MID)
print("arch", cfg.architectures, "| model_type", cfg.model_type)
tc = cfg.text_config
print("text_config: n_layers", tc.num_hidden_layers, "hidden", tc.hidden_size)

# Try the most appropriate loader class.
model = None
for loader_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModel"):
    try:
        Loader = getattr(transformers, loader_name)
        model = Loader.from_pretrained(MID, dtype=torch.bfloat16, low_cpu_mem_usage=True)
        print(f"loaded via {loader_name}: {type(model).__name__}")
        break
    except Exception as e:
        print(f"{loader_name} failed: {type(e).__name__}: {str(e)[:120]}")

assert model is not None
# Find ModuleLists whose children look like decoder layers (have self_attn).
import torch.nn as nn
print("\n--- candidate decoder ModuleLists ---")
for name, mod in model.named_modules():
    if isinstance(mod, nn.ModuleList) and len(mod) > 0:
        child = mod[0]
        has_attn = any("attn" in n.lower() for n, _ in child.named_modules())
        if has_attn and len(mod) >= 10:
            print(f"  {name!r}  len={len(mod)}  child={type(child).__name__}")

tok = AutoTokenizer.from_pretrained(MID)
print("\ntokenizer:", type(tok).__name__, "bos", tok.bos_token_id)
