# Results — does the model's internal "amusement" track audience laughter?

**Setup.** `Qwen/Qwen2.5-1.5B-Instruct`. Amusement direction built with
`steering-vectors` (diff-of-means) from 40 funny vs 40 vivid-but-not-funny
passages; layer **17** (of 28) selected on held-out separation. The direction is
projected onto every token of the comedian's words (laughter markers removed
first), z-scored, and aligned to the real `[LAUGHTER]` positions.

Two statistics:
- **Peri-laughter average** — the z-scored signal in a ±40-token window around
  each laugh onset (offset 0 = last spoken token before the audience laughs).
- **Permutation test** — is the mean signal in the tokens leading into a laugh
  higher than for the same number of random positions? (10k / 4k shuffles.)

## Headline

| | Bill Burr — *Drop Dead Years* | John Mulaney — *Kid Gorgeous* |
|---|---|---|
| spoken tokens / laughs | 15,935 / 221 | 11,611 / 78 |
| **punchline token** (offset 0) | **z=1.19, 17.4σ, p=0.0002** | **z=0.74, 6.6σ, p=0.0002** |
| 12-token run-up (aggregate) | **+0.17, 4.6σ, p=0.0002** | +0.00, 0.07σ, p=0.48 (n.s.) |
| run-up **excluding** punchline tok | +0.08, 2.0σ, p=0.018 | −0.06, n.s. |
| run-up, **content words only** | +0.04, 2.4σ, p=0.009 | −0.06, n.s. |
| detection AUC (pre-laugh vs rest) | 0.55 | 0.50 |
| fraction of punchline tokens that are pure punctuation | 0.99 | 1.00 |

## What this means

1. **The signal spikes exactly at the punchline boundary.** For *both*
   comedians the internal amusement signal jumps sharply at offset 0 — the last
   word before the audience laughs — and the spike is hugely significant
   (Burr 17σ, Mulaney 6.6σ). The model's "funny" direction lights up right where
   the joke lands. This is the clean, replicated effect (see the peri-laughter
   panels in `data/*_amusement.png`).

2. **…but a large part of that spike is timing/punctuation, not semantics.**
   ~100% of the offset-0 tokens are terminal punctuation (`!  .  ?  "`), and the
   layer-17 direction fires extremely hard on `!` (mean z = **4.4** at `!` tokens
   vs −0.05 elsewhere). The punchline is, almost by definition, the sentence end
   that the laugh follows — so "fires on the final beat of a sentence that gets a
   laugh" is partly a comedic-*timing* detector, partly an amusement one.

3. **The genuine *semantic* anticipation is real but small, and style-dependent.**
   Stripping the punctuation punchline token and looking only at content words in
   the run-up:
   - **Bill Burr:** still significantly elevated (content-words z=2.4, p=0.009).
     His punchy, one-liner, exclamatory style gives the model funny *content* to
     represent in the seconds before the laugh.
   - **John Mulaney:** nothing in the run-up (p≈0.8). His long-form storytelling
     humor is contextual/structural; a 1.5B model doesn't represent "this is about
     to be funny" from the setup words, only reacts at the punchline boundary.

## Honest bottom line

A small open model's internal **amusement emotion vector does track stand-up
comedy**: it spikes reliably on the punchline beat that triggers laughter
(replicated across two comedians, p=0.0002 each). But the effect decomposes into
(a) a strong **comedic-timing / terminal-punctuation** component at the exact
punchline boundary, and (b) a weaker, genuinely **semantic** "this is funny"
component that is present for a punchline-driven comedian (Burr) and essentially
absent for a storytelling comedian (Mulaney) — at least at 1.5B parameters.
Scaling the model and orthogonalizing the direction against a punctuation/
sentence-boundary direction are the obvious next steps to isolate (b) from (a).
