# Strand 3 — Computational Humor, NLP & LLMs, Laughter Prediction (raw research findings)

> Raw deliverable from the deep-research agent. Source for compiled `literature-review.md` §§4–6. All citations verified against primary records (ACL Anthology / arXiv / publisher).

## 1. Early computational humor
- **Binsted & Ritchie 1994/1997 — JAPE.** *Humor 10*(1), 25–76; AAAI-94. https://www.inf.ed.ac.uk/publications/online/0158.pdf — WordNet rule-based punning-riddle generator (word/syllable substitution, metathesis). [VERIFIED]
- **Stock & Strapparava 2003/2005 — HAHAcronym.** *ACL 2005 (Demo)*, 113–116. https://aclanthology.org/P05-3029/ — incongruous acronym re-expansions grounded in incongruity theory. [VERIFIED]
- **Mihalcea & Strapparava 2005 — Making Computers Laugh.** *HLT-EMNLP 2005*, 531–538. https://aclanthology.org/H05-1067/ — first large ML humor recognition (16K one-liners; SVM/NB). [VERIFIED]
- **Mihalcea & Strapparava 2006 — Learning to Laugh.** *Computational Intelligence 22*(2), 126–142. https://doi.org/10.1111/j.1467-8640.2006.00278.x — stylistic+semantic features; the 16K one-liner benchmark. [VERIFIED]

## 2. Humor recognition, classification & datasets
- **Yang, Lavie, Dyer & Hovy 2015 — Humor Recognition & Anchor Extraction.** *EMNLP 2015*, 2367–2376. https://doi.org/10.18653/v1/D15-1284 — latent structures (incongruity/ambiguity) + anchor extraction localizing the trigger phrase. [VERIFIED]
- **Miller, Hempelmann & Gurevych 2017 — SemEval-2017 Task 7 (puns).** https://aclanthology.org/S17-2005/ — detection/location/interpretation; standard Pun Corpus. [VERIFIED]
- **Potash, Romanov & Rumshisky 2017 — SemEval-2017 Task 6 (#HashtagWars).** https://aclanthology.org/S17-2004/ — humor as comparative ranking. [VERIFIED]
- **Hossain, Krumm, Gamon & Kautz 2020 — SemEval-2020 Task 7 (Humicroedit).** https://doi.org/10.18653/v1/2020.semeval-1.98 (arXiv:2008.00304) — funniness regression on edited headlines; 48 teams. [VERIFIED]
- **Annamoradnejad & Zoghi 2020 — ColBERT.** arXiv:2004.12765. https://arxiv.org/abs/2004.12765 — parallel BERT, ~98% F1 on 200K joke/non-joke set; standard benchmark. [VERIFIED]
- **Gultchin et al. 2019 — Humor in Word Embeddings.** *ICML 2019* PMLR 97. https://proceedings.mlr.press/v97/gultchin19a.html — humor properties as linear directions in embedding space. [VERIFIED]

## 3. New Yorker Caption Contest line
- **Radev et al. 2015 — Humor in Collective Discourse.** arXiv:1506.08126. https://arxiv.org/abs/1506.08126 — first computational study; 12 methods select funniest finalist. [VERIFIED]
- **Hessel et al. 2023 — Do Androids Laugh at Electric Sheep?** *ACL 2023*, 688–714. arXiv:2209.06293. https://arxiv.org/abs/2209.06293 — match/rank/explain tasks; top models lag humans, esp. explanation. The field's central benchmark. [VERIFIED]
- **Zhang et al. 2024 — Humor in AI.** arXiv:2406.10522. https://arxiv.org/abs/2406.10522 — 250M+ ratings over 2.2M captions; largest humor-preference dataset. [VERIFIED]

## 4. LLMs and humor (2022–2026)
- **Jentzsch & Kersting 2023 — ChatGPT is fun, but it is not funny!** arXiv:2306.04563. https://arxiv.org/abs/2306.04563 — recycles cached jokes; shallow comprehension (analysis of 1,008 jokes). [VERIFIED]
- **Horvitz et al. 2024 — Getting Serious about Humor.** arXiv:2403.00794. https://arxiv.org/abs/2403.00794 — LLM-generated "unfunny" contrastive pairs improve detectors. [VERIFIED]
- **Tikhonov & Shtykovskiy 2024 — Humor Mechanics.** arXiv:2405.07280 (ICCC 2024). https://arxiv.org/abs/2405.07280 — multistep CoT (topic→setup→incongruity→punchline) raises rated funniness. [VERIFIED]
- **Baluja 2024 — Text Is Not All You Need.** arXiv:2412.05315. https://arxiv.org/abs/2412.05315 — phonetic/rhythmic/timing cues improve LLM humor understanding; text-only perplexity misses delivery. [VERIFIED]
- **Romanowski, Valois & Fukui 2025 — From Punchlines to Predictions.** arXiv:2504.09049 (CMCL 2025). https://arxiv.org/abs/2504.09049 — metric for punchline identification in stand-up; ChatGPT/Claude/DeepSeek ~51% (humans ~41%). [VERIFIED]
- **Amin & Burghardt 2020 — Survey on Computational Humor Generation.** https://aclanthology.org/2020.latechclfl-1.4/ — flags absence of automatic funniness metrics. [VERIFIED]
- **Lemmens & De Marez 2026 — Computational Humor Modeling: A Survey.** *ACM Computing Surveys 58*(7), Art. 177. https://doi.org/10.1145/3778357 — most recent comprehensive SOTA survey. [VERIFIED]

## 5. Perplexity/surprisal as humor signal
- **Xie, Li & Pu 2021** (see Strand 2 / §13) — GPT-2 uncertainty+surprisal for humor recognition. [VERIFIED]
- **Ma, Peng, Lyu, Zhang & Zhu 2026 — Timing is Everything.** arXiv:2605.00143 (CogSci 2026). https://arxiv.org/abs/2605.00143 — 828 Chinese stand-up sets; Dual Prediction Violation; timing features outweigh semantic incongruity; pauses lengthen before high-surprisal punchlines. [VERIFIED — recent preprint]

## 6. Predicting audience laughter
- **Bertero & Fung 2016 — LSTM for Predicting Humor in Dialogues.** *NAACL-HLT 2016*, 130–135. https://doi.org/10.18653/v1/N16-1016 — predicts laugh-track timing on *Big Bang Theory*. [VERIFIED]
- **Chen & Lee 2017 — Predicting Audience's Laughter (TED, CNN).** *BEA @ EMNLP 2017*, 86–90. arXiv:1702.02584. https://aclanthology.org/W17-5009/ — TED-talk laughter prediction from lexical features. [VERIFIED]
- **Zribi, Cafiero, Lépinay & Vidal-Gorène 2026 — TIC-TALK.** arXiv:2603.21803. https://arxiv.org/abs/2603.21803 — multimodal stand-up DB (text/audio/laughter/kinesics), 90 sets, 5,400+ segments. [VERIFIED — recent preprint]
- **Barriere et al. 2025 — StandUp4AI.** arXiv:2505.18903. https://arxiv.org/abs/2505.18903 — 330+ hrs stand-up, 7 languages, humor annotations. [VERIFIED — recent]

## Synthesis
Progression: rule-based generators (JAPE, HAHAcronym) → statistical classifiers (Mihalcea & Strapparava; Yang et al.) → transformer detectors at near-ceiling on controlled data (ColBERT, SemEval). Key recent insight: LM surprisal at the punchline is a theoretically motivated, empirically validated humor signal (Xie et al. 2021; Ma et al. 2026). Three open gaps: (1) LLMs still fail humor *understanding* vs detection (Hessel et al.; Jentzsch & Kersting); (2) perplexity misses phonological/prosodic/visual incongruity (Baluja 2024); (3) predicting *when* audiences laugh remains near chance (Romanowski et al. 2025; human agreement ~41%).
