# Strand 4 — Mechanistic Interpretability, Concept/Emotion Vectors & Criticality in Neural Nets (raw research findings)

> Raw deliverable from the deep-research agent. Source for compiled `literature-review.md` §§7–8. All citations verified against primary records (transformer-circuits.pub / arXiv / DOI).

## Interpretability foundations
- **Olah et al. 2020 — Zoom In: An Introduction to Circuits.** *Distill.* https://doi.org/10.23915/distill.00024.001 — neurons detect features, circuits implement algorithms; the decomposition program. [VERIFIED]
- **Elhage et al. 2021 — A Mathematical Framework for Transformer Circuits.** https://transformer-circuits.pub/2021/framework/index.html — residual stream as linear maps; QK/OV decomposition. [VERIFIED]
- **Olsson et al. 2022 — In-context Learning and Induction Heads.** arXiv:2209.11895. https://arxiv.org/abs/2209.11895 — induction heads ([A][B]…[A]→[B]) as the prediction/pattern-completion mechanism. [VERIFIED]
- **Elhage et al. 2022 — Toy Models of Superposition.** arXiv:2209.10652. https://arxiv.org/abs/2209.10652 — polysemantic neurons from feature superposition; motivates SAEs. [VERIFIED]
- **Bricken et al. 2023 — Towards Monosemanticity.** https://transformer-circuits.pub/2023/monosemantic-features — sparse autoencoders recover monosemantic features on a 1-layer transformer. [VERIFIED]
- **Templeton et al. 2024 — Scaling Monosemanticity (Claude 3 Sonnet).** https://transformer-circuits.pub/2024/scaling-monosemanticity/ — 34M SAE features incl. abstract emotional/conceptual ones; causally steer behavior. [VERIFIED]

## Concept / emotion representation & probing
- **Park, Choe & Veitch 2023 — Linear Representation Hypothesis.** arXiv:2311.03658 (ICML 2024). https://arxiv.org/abs/2311.03658 — concepts as linear directions; links probing and steering. [VERIFIED]
- **Gurnee & Tegmark 2023 — Language Models Represent Space and Time.** arXiv:2310.02207. https://arxiv.org/abs/2310.02207 — linear "space/time neurons"; template for probing abstract concepts. [VERIFIED]
- **Zou et al. 2023 — Representation Engineering.** arXiv:2310.01405. https://arxiv.org/abs/2310.01405 — linear directions for honesty/fairness/emotions via contrastive pairs; steering. Operational method for emotion/humor vectors. [VERIFIED]
- **Turner et al. 2023 — Activation Addition.** arXiv:2308.10248. https://arxiv.org/abs/2308.10248 — steering vectors as contrastive activation differences (e.g., happy−sad) added at inference. [VERIFIED]

## Prediction-uncertainty machinery (the surprisal substrate)
- **Gurnee et al. 2024 — Universal Neurons in GPT2.** arXiv:2401.12181. https://arxiv.org/abs/2401.12181 — taxonomy incl. *entropy neurons* (modulate output-distribution entropy without changing the predicted token) and token-prediction neurons. [VERIFIED]
- **Stolfo et al. 2024 — Confidence Regulation Neurons in Language Models.** NeurIPS 2024. arXiv:2406.16254. https://arxiv.org/abs/2406.16254 — entropy/confidence neurons scale logits via the unembedding null space; mechanistic account of internal uncertainty. The clearest analog of a "surprisal computation." [VERIFIED]

## Emotion interpretability (EmotionScope-adjacent)
- **Sofroniew et al. 2026 — Emotion Concepts and their Function in a Large Language Model.** arXiv:2604.07729; https://transformer-circuits.pub/2026/emotions/index.html — 171 emotion concepts; linear *emotion vectors* that activate appropriately, steer behavior, and show internal-external decoupling. Closest match to the host project's "EmotionScope replication." [VERIFIED — recent]
- **Tak et al. 2025 — Mechanistic Interpretability of Emotion Inference in LLMs.** arXiv:2502.05489. https://arxiv.org/abs/2502.05489 — layer-wise emotion probing; mid-layer attention key; linear steering. [VERIFIED]
- **Wang et al. 2025 — Do LLMs "Feel"? Emotion Circuits Discovery and Control.** arXiv:2510.11328. https://arxiv.org/abs/2510.11328 — context-agnostic emotion directions; circuit-level control of six emotions (99.65% acc). [VERIFIED — recent]

## Surprisal/timing in jokes (cross-listed with Strands 2–3)
- **Xie, Li & Pu 2021 — Uncertainty and Surprisal Jointly Deliver the Punchline.** arXiv:2012.12007; https://aclanthology.org/2021.acl-short.6/ — GPT-2 entropy+surprisal for humor recognition. [VERIFIED]
- **Ma et al. 2026 — Timing is Everything.** arXiv:2605.00143 — token surprisal + temporal scaffolding on Chinese stand-up. [VERIFIED — recent]

## Criticality / edge of chaos
- **Poole et al. 2016 — Exponential Expressivity through Transient Chaos.** *NIPS 2016*, 3360–3368. https://papers.nips.cc/paper/6322 — order→chaos phase transition; max expressivity at the edge of chaos. [VERIFIED]
- **Schoenholz et al. 2017 — Deep Information Propagation.** ICLR 2017. arXiv:1611.01232. https://arxiv.org/abs/1611.01232 — trainability requires signal propagation at the edge of chaos. [VERIFIED]
- **Beggs & Plenz 2003 — Neuronal Avalanches.** *J. Neurosci. 23*(35), 11167–11177. https://doi.org/10.1523/JNEUROSCI.23-35-11167.2003 — power-law cortical avalanches; SOC in the brain. [VERIFIED]
- **Shew & Plenz 2013 — The Functional Benefits of Criticality in the Cortex.** *The Neuroscientist 19*(1), 88–100. https://doi.org/10.1177/1073858412445487 — dynamic range, info transmission & capacity maximized at criticality. [VERIFIED]
- **Kaplan et al. 2020 — Scaling Laws for Neural Language Models.** arXiv:2001.08361. https://arxiv.org/abs/2001.08361 — power-law loss scaling; echoes Zipfian/scale-free language statistics. [VERIFIED]

## Synthesis
Mechanistic interpretability provides the toolkit — circuits, linear probes, SAEs, steering vectors — to locate a possible "humor/incongruity/surprise" representation inside a transformer. The linear representation hypothesis (Park et al.) implies such a representation would be a residual-stream direction; SAEs (Bricken; Templeton) disentangle it; contrast-pair/elicitation methods (Zou; Turner; Sofroniew) isolate it. Confidence/entropy neurons (Gurnee; Stolfo) show transformers encode prediction uncertainty in identifiable components — exactly the surprisal signal Xie and Ma operationalize. The criticality literature is complementary: brains operate near criticality (Beggs & Plenz; Shew & Plenz), deep nets peak at the same edge (Poole; Schoenholz), and LLM scaling laws (Kaplan) mirror Zipfian language statistics. The criticality↔trained-LLM-humor link is well-established for biology and initialization theory but remains speculative for trained LLMs processing humor — an intriguing framing, not a confirmed mechanism.
