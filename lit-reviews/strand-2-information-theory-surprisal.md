# Strand 2 — Information Theory, Surprisal & Predictive Processing (raw research findings)

> Raw deliverable from the deep-research agent. Source for compiled `literature-review.md` §§2–3 and §8 (criticality). All citations verified against primary records.

## 1. Shannon 1948 — A Mathematical Theory of Communication
- *Bell System Technical Journal, 27*(3–4), 379–423, 623–659. https://doi.org/10.1002/j.1538-7305.1948.tb01338.x
- Entropy H = −Σ p log p; self-information (surprisal) = −log p(x). Quantitative seed of "surprise = information." [VERIFIED — Wiley]

## 2. Hale 2001 — A Probabilistic Earley Parser as a Psycholinguistic Model
- *NAACL 2001*, 159–166. https://aclanthology.org/N01-1021/
- Cognitive load at word w ∝ −log P(w | context). Predicts garden-path effects. [VERIFIED — ACL Anthology]

## 3. Levy 2008 — Expectation-Based Syntactic Comprehension
- *Cognition, 106*(3), 1126–1177. https://doi.org/10.1016/j.cognition.2007.05.006
- Generalizes surprisal to comprehension; difficulty = surprisal = belief-update cost. [VERIFIED — ScienceDirect/PubMed PMID 17662975]

## 4. Smith & Levy 2013 — Effect of Word Predictability on Reading Time Is Logarithmic
- *Cognition, 128*(3), 302–319. https://doi.org/10.1016/j.cognition.2013.02.006
- Reading time ∝ surprisal (logarithmic in word probability). Validates the linking assumption. [VERIFIED — ScienceDirect]

## 5. Wilcox et al. 2023 — Testing Surprisal Theory in 11 Languages
- *TACL, 11*, 1451–1470. https://doi.org/10.1162/tacl_a_00612 (arXiv:2307.03667)
- Neural-LM surprisal predicts reading time cross-linguistically; linking function is linear. [VERIFIED — ACL/MIT Press]

## 6. Shain et al. 2020 — fMRI Reveals Language-Specific Predictive Coding
- *Neuropsychologia, 138*, 107307. https://doi.org/10.1016/j.neuropsychologia.2019.107307
- Surprisal modulates the language network specifically (not multiple-demand). Neural substrate for setup→punchline. [VERIFIED — PubMed PMID 31874149]

## 7. Friston 2010 — The Free-Energy Principle: A Unified Brain Theory?
- *Nature Reviews Neuroscience, 11*(2), 127–138. https://doi.org/10.1038/nrn2787
- Brain minimizes variational bound on surprise. Humor = controlled free-energy spike then resolution. [VERIFIED — Nature]

## 8. Rao & Ballard 1999 — Predictive Coding in the Visual Cortex
- *Nature Neuroscience, 2*(1), 79–87. https://doi.org/10.1038/4580
- Hierarchical predictive coding: top-down predictions, bottom-up errors. Neural-circuit basis of free energy. [VERIFIED — Nature Neuroscience PMID 10195184]

## 9. Clark 2013 — Whatever Next? Predictive Brains, Situated Agents
- *Behavioral and Brain Sciences, 36*(3), 181–204. https://doi.org/10.1017/S0140525X12000477
- The "prediction machine" synthesis; mismatch is central to all cognition. [VERIFIED — Cambridge Core]

## 10. Suls 1972 — Two-Stage Model (see Strand 1)
- Information-processing incongruity-resolution; Stage 1 = high surprisal, Stage 2 = entropy reduction. [VERIFIED — Semantic Scholar; pre-DOI]

## 11. Hurley, Dennett & Adams 2011 — Inside Jokes
- MIT Press. ISBN 978-0-262-01582-0. https://mitpress.mit.edu/9780262518697/inside-jokes/
- Humor = reward signal for detecting/correcting false commitments in the generative model (prediction-error correction). [VERIFIED — MIT Press]

## 12. Kao, Levy & Goodman 2016 — A Computational Model of Linguistic Humor in Puns
- *Cognitive Science, 40*(5), 1270–1285. https://doi.org/10.1111/cogs.12269
- Defines *ambiguity* (entropy over meanings) and *distinctiveness* (KL-divergence); predict pun funniness (R²≈0.25). [VERIFIED — Wiley/PMC5042108]

## 13. Xie, Li & Pu 2021 — Uncertainty and Surprisal Jointly Deliver the Punchline
- *ACL-IJCNLP 2021 (Short)*, 33–39. arXiv:2012.12007. https://aclanthology.org/2021.acl-short.6/
- GPT-2 entropy (setup) + surprisal (punchline) outperform baselines for humor recognition. Closest precedent to "epiplexity." [VERIFIED — ACL/arXiv]

## 14. Mihalcea & Strapparava 2006 — Learning to Laugh (Automatically)
- *Computational Intelligence, 22*(2), 126–142. https://doi.org/10.1111/j.1467-8640.2006.00278.x
- 16,000 one-liners; humor has detectable statistical signatures. [VERIFIED — Wiley]

## 15. Bak, Tang & Wiesenfeld 1987 — Self-Organized Criticality
- *Physical Review Letters, 59*(4), 381–384. https://doi.org/10.1103/PhysRevLett.59.381
- SOC: systems evolve to a critical state (power-law/1/f) at the order–chaos boundary; max sensitivity/info transmission. [VERIFIED — APS]

## 16. Langton 1990 — Computation at the Edge of Chaos
- *Physica D, 42*(1–3), 12–37. https://doi.org/10.1016/0167-2789(90)90064-V
- Info storage/transmission/modification jointly maximized near the order–chaos phase transition. [VERIFIED — ScienceDirect]

## 17. Beggs & Plenz 2003 — Neuronal Avalanches in Neocortical Circuits
- *Journal of Neuroscience, 23*(35), 11167–11177. https://doi.org/10.1523/JNEUROSCI.23-35-11167.2003
- Cortex shows power-law avalanches — SOC signature; brains poised near criticality. [VERIFIED — J. Neurosci.]

## 18. Bilder & Knudsen 2014 — Creative Cognition and Systems Biology on the Edge of Chaos
- *Frontiers in Psychology, 5*, 1104. https://doi.org/10.3389/fpsyg.2014.01104
- Optimal creativity at the order–chaos boundary (inverted-U); ideas novel yet coherent. Most direct criticality↔higher-cognition link. [VERIFIED — Frontiers/PMC4179729]

## 19. Berlyne 1971 — Aesthetics and Psychobiology
- Appleton-Century-Crofts. [VERIFIED-BY-CITATION — pre-DOI monograph]
- Inverted-U: hedonic pleasure peaks at intermediate complexity/uncertainty (collative variables). Foreshadows "optimal surprisal."

## 20. Shain et al. 2024 — Large-Scale Evidence for Logarithmic Effects of Word Predictability
- *PNAS, 121*(10), e2307876121. https://doi.org/10.1073/pnas.2307876121
- Confirms logarithmic surprisal↔reading-time at scale with modern neural LMs. [VERIFIED — PNAS PMID 38422017]

## Synthesis
**Established:** Shannon's surprisal is the mathematical core; Hale→Levy→Smith&Levy→Wilcox→Shain show LM surprisal predicts human processing (behavior + fMRI). Suls/Hurley frame humor as prediction-error detection; Kao et al. and Xie et al. operationalize it (entropy, KL, LM surprisal) to predict funniness. Bak/Langton/Beggs&Plenz establish criticality as the info-optimal regime; Berlyne and Bilder & Knudsen connect optimal complexity to pleasure/creativity.
**Speculative / novel:** No prior work formally unifies these into "humor requires punchline surprisal at the edge-of-chaos critical point, operationalized by LM perplexity." Friston/Clark bridge criticality↔affective reward, but the explicit quantified claim is the host paper's potential contribution.
