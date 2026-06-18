"""Add the scenario_matrix to an already-finished model's joke JSON, reusing the
saved emotion vectors (no re-extraction of word scores). Reloads the model only
to run the 12 scenario probes.

Usage: .venv-es/bin/python backfill_scenarios.py <hf_model_id>
"""
import json, sys
import numpy as np, torch
from joke_pipeline import load_any, scenario_matrix, model_tag


def main():
    model_id = sys.argv[1]
    tag = model_tag(model_id)
    jpath = f"data/joke/{tag}.json"
    d = json.load(open(jpath))
    if d.get("scenario_matrix", {}).get("values"):
        print(f"{tag}: already has scenario_matrix, skipping"); return
    npz = np.load(f"data/vectors/{tag}.npz")
    names = [str(n) for n in npz["emotions"]]
    vectors = {n: torch.tensor(npz[f"vec_{n}"]).float() for n in names}
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, tok, backend, info = load_any(model_id, device)
    info["probe_layer"] = int(npz["probe_layer"])
    sm = scenario_matrix(model, tok, backend, info, vectors)
    d["scenario_matrix"] = sm
    json.dump(d, open(jpath, "w"))
    print(f"backfilled {tag}: {len(sm['rows'])} emotions x {len(sm['cols'])} scenarios")


if __name__ == "__main__":
    main()
