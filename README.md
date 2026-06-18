# Judging Humor — monitoring an "amusement" emotion vector across stand-up comedy

Can a language model's *internal* representation of **funniness** tell us where a
stand-up audience will laugh? This project replicates Anthropic's
[*Emotion concepts and their function in a large language model*](https://www.anthropic.com/research/emotion-concepts-function)
on a small open model, builds an **amusement / funniness emotion vector**, reads
it **token-by-token** across real stand-up transcripts, and compares the signal to
the actual `[LAUGHTER]` markers in the transcript.

## The method (and the framework we use)

Anthropic's "emotion vectors" are **difference-of-means activation directions**:
generate text expressing an emotion, record the residual-stream activations,
and take `mean(emotion) − mean(control)`. Projecting new activations onto that
(unit) direction reads how strongly the emotion is "active" at each token.

We do **not** hand-roll this. We use the open-source
**[`steering-vectors`](https://github.com/steering-vectors/steering-vectors)**
library (`pip install steering-vectors`, v0.12.2):

- `train_steering_vector(...)` builds the diff-of-means direction from
  contrastive `(funny, not-funny)` text pairs across every layer.
- `record_activations(...)` exposes the per-token residual stream so we can
  project it onto the direction and get a per-token amusement signal.

This was selected after an exhaustive survey of emotion/representation-vector
frameworks (`steering-vectors`, RepE / `representation-engineering`, `repeng`,
[`traitinterp`](https://github.com/ewernn/traitinterp) — the explicit
Anthropic-paper replication, etc.). `steering-vectors` won on: pip-installable,
pure HF/PyTorch (works on Apple-Silicon MPS/CPU), and a first-class **reading**
API, not just steering.

**Model:** `Qwen/Qwen2.5-1.5B-Instruct` (ungated, runs on MPS). 28 layers.

## Making the experiment honest

1. **No leakage.** The `[LAUGHTER]` markers are *stripped before the model sees
   the text*. The model reads only the comedian's words; we keep the token index
   of every laugh and ask whether the internal amusement signal rises *into* it.
2. **Style-matched controls.** A first attempt used flat office-prose negatives,
   which produced a *punctuation/exclamation* detector (AUC saturated; top tokens
   were all `!`). The negatives were rewritten to be **equally vivid, emotional,
   exclamatory and dialogue-rich, but not funny** (anger / fear / grief / drama),
   so the direction isolates *funniness* rather than arousal or punctuation.
3. **Meaningful layer.** Layer selection is restricted to mid/late layers
   (concepts live there, not in surface-feature early layers).

## Files

| file | role |
|---|---|
| `download_transcript.py` | fetch + clean a [scrapsfromtheloft](https://scrapsfromtheloft.com) transcript, normalising laughter markers to `[LAUGHTER]` and stripping other stage directions |
| `emotion_data.py` | contrastive corpus: 40 funny vs 40 vivid-but-not-funny passages |
| `humor_steering.py` | build the amusement vector (`steering-vectors`), read it per-token over a transcript, run stats + plot |
| `align_and_report.py` | framework-agnostic glue: token/laughter alignment, peri-laughter average, permutation test, detection AUC, plotting |
| `emotion_lib.py` | a from-scratch reference implementation of the same diff-of-means method (not used by the main run; kept for comparison) |
| `data/*.png` | the result plots |

## Run

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch transformers steering-vectors numpy matplotlib accelerate
python download_transcript.py bill-burr-drop-dead-years
python humor_steering.py --transcript data/bill-burr-drop-dead-years.txt --name "Bill Burr — Drop Dead Years"
```

## Results

(see `RESULTS.md` / the PNGs — filled in by the run)
