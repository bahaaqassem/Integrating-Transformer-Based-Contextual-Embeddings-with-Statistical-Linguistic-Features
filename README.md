# AI-Generated Arabic News Detection — Notebooks

MSc thesis project (Islamic University of Gaza, Faculty of Information Technology):
**"Integrating Transformer-Based Contextual Embeddings with Statistical Linguistic
Features to Detect AI-Generated Arabic News Reports."**

A hybrid binary classifier that fuses a frozen/fine-tuned CAMeLBERT-MSA document
embedding (`V_neural ∈ R^768`) with a small vector of interpretable statistical
linguistic features (`V_stat`) to distinguish human-written from AI-generated
long-form Modern Standard Arabic (MSA) news.

- **Author:** Bahaa Khalil Kamal Qassem (120240332)
- **Supervisor:** Dr. Rebhi S. Baraka
- **Compute:** Kaggle (T4 GPU, 12-hour session limit)

---

## Table of contents

1. [Pipeline overview](#pipeline-overview)
2. [Stage 0 — Human corpus source fix](#stage-0--human-corpus-source-fix)
3. [Stage 1 — Corpus construction](#stage-1--corpus-construction)
4. [Stage 2 — Statistical features & encoder selection](#stage-2--statistical-features--encoder-selection)
5. [Stage 3 — Hybrid Track A (frozen) & feature revision](#stage-3--hybrid-track-a-frozen--feature-revision)
6. [Stage 4 — Track B (joint fine-tuning) & full LOGO](#stage-4--track-b-joint-fine-tuning--full-logo)
7. [Stage 5 — Stress test](#stage-5--stress-test)
8. [Stage 6 — External validation](#stage-6--external-validation)
9. [Reference results ladder](#reference-results-ladder)
10. [Conventions and contracts](#conventions-and-contracts)

---

## Pipeline overview

```
Stage 0   NB0 → NB0b                              human corpus (source-fix)
Stage 1   NB2b → NB2b-post → NB2c → NB2d–i → NB2j → NB3   dataset.parquet
Stage 2   NB4 → NB5a → NB5b → NB5b-LOGO → NB5c → NB8      features + encoder choice
Stage 3   NB6 → NB6b → NB6c → NB6e → NB6f → NB6g → NB7    frozen hybrid + feature search
Stage 4   NB9 → NB10 → NB14 → NB11 → NB12                 fine-tuned hybrid + LOGO
Stage 5   NB13 → NB15 → NB16                              robustness under text edits
Stage 6   NB17 → NB18                                     external / zero-shot baselines
```

Each stage's output is a Kaggle dataset consumed by the next stage; see
[Conventions and contracts](#conventions-and-contracts) for the alignment rules
enforced across all of them.

---

## Stage 0 — Human corpus source fix

| Notebook | Input | Output | Purpose |
|---|---|---|---|
| `NB0_harvest_culturax` | CulturaX (`ar` subset, streamed) | `culturax_raw_harvest.parquet` (~15k docs) | Raw harvest from trusted non-Western Arabic news domains (Al Jazeera, Al Arabiya, Al Mayadeen, RT Arabic, Anadolu, Arabi21, Al Quds, Al Araby, Maan, Wafa, Quds Press), filtered by ≥2 political/Mideast keyword hits and 400–6000 words. **Zero text stripping** beyond whitespace — built specifically to replace an earlier human corpus that had been over-sanitized (digits, quotes, colons and most punctuation stripped), which would have let a classifier cheat trivially. |
| `NB0b_select_corpus` | `culturax_raw_harvest.parquet` | `ha_corpus.parquet` (3,500 rows) | Light line-level wrapper removal (Hijri date headers, share buttons, pipe tails, photo captions) with **reject-on-artifact** for anything that still shows a trace (stray pipe, "آخر تحديث", GMT stamp, UUID, relative timestamps). Politics-core keyword filter, content dedup, length-stratified sampling to exactly 3,500, natural source distribution kept (no per-outlet cap). |

**Verified fix:** digit presence in the final human corpus is 90.0% (avg 22.3/article)
vs. 88.8% for AI (13.9/article) — consistent with raw text, not a sanitized corpus.

---

## Stage 1 — Corpus construction

| Notebook | Input | Output | Purpose |
|---|---|---|---|
| `NB2b_algorithmic_extraction` | `ha_corpus.parquet` | `extract_algo.parquet` | Model-free fact extraction per human article: CAMeL-NER entities (chunk+union, word-aligned boundaries, `aggregation_strategy='first'`), numbers/dates (regex), news agencies (dictionary), key sentences (TextRank/PageRank), quote **events** (reporting verb + short span — never verbatim, so generators can't copy quotes as a shortcut), and `n_fact_points = clip(round(words/400), 3, 7)`. |
| `NB2b_post_entity_merge` | `extract_algo.parquet` | `extract_algo.parquet` (overwritten) | Merges Arabic prefix-variant entity duplicates (و/ب/ل/ال…, e.g. حماس/وحماس/بحماس) with a safe rule: only strips a prefix when the stripped form already exists in the same list, so standalone names (e.g. بغداد) are never touched. ~9.7% of raw entity mentions merged. |
| `NB2c` | `extract_algo.parquet` | fact cards | Assembles the final per-article fact card used to constrain generation. |
| `NB2d`–`NB2i` | fact cards | per-generator AI articles | Counter-generation across six LLM generators (DeepSeek, Gemini, GPT-5 Mini, Claude Sonnet, Claude Opus, Qwen) from the same fact cards as their paired human article. |
| `NB2j` | all generator outputs | `aig_corpus.parquet` (3,601 rows) | Merge: keeps 104 duplicate cards (two AI articles), drops 258 failed generations. |
| `NB3_build_dataset` | `ha_corpus.parquet` + `aig_corpus.parquet` | `dataset.parquet` (7,101 rows) | Symmetric normalization applied to **both** classes (fixes an earlier newline-based leak), then a pair-aware 75/10/15 split by fact-card hash — no card straddles a split boundary. |

**Final corpus:** 3,500 human + 3,601 AI (deepseek 878, sonnet 840, qwen 622, gemini 441,
gpt 440, opus 380) — train 5,363 / val 645 / test 1,093.

---

## Stage 2 — Statistical features & encoder selection

| Notebook | Purpose |
|---|---|
| `NB4` | Extracts the original 5-feature `V_stat` (Targeted Perplexity via AraGPT2-base surrogate, Burstiness, TTR, Entity Density, Discourse Coherence). Scaler fit on train only. |
| `NB5a` | Statistical-only baseline classifiers; best is GradientBoosting at 84.7% macro-F1 (vs. a 68.8% cheap-signal control). |
| `NB5b_arabert` | AraBERTv2 two ways in one notebook: **(1)** frozen `[CLS]` + linear probe (Farasa preprocessing, sentence-aware chunking, mean-pooled) — the "frozen representation as-is" number; **(2)** full end-to-end fine-tuning — quantifies what adaptation adds. Caches `arabert_cls_frozen.npy`. |
| `NB5b_arabert_LOGO` | Six-fold Leave-One-Generator-Out on the **cached** frozen `[CLS]` vectors from NB5b (no re-encoding) — confirms the frozen representation generalizes to an unseen generator rather than memorizing per-generator fingerprints. |
| `NB5c` | Extends the comparison to AraELECTRA and CAMeLBERT-MSA under an identical frozen-probe protocol. **CAMeLBERT-MSA selected** as the neural core (best on both the standard split and the LOGO robustness profile; AraELECTRA collapses on the hardest held-out generator). |
| `NB8` | Adds XLM-RoBERTa as a fourth frozen probe for completeness against multilingual-encoder literature; CAMeLBERT-MSA remains the best fit for this topic-controlled, long-form corpus. |

---

## Stage 3 — Hybrid Track A (frozen) & feature revision

| Notebook | Purpose |
|---|---|
| `NB6` / `NB6b` | First frozen-hybrid attempt failed a pre-registered criterion; `NB6b` diagnoses the cause (an under-trained fusion head, not the features) and fixes the trainer. |
| `NB6c` | 27-configuration ablation of the original 5 features across two classifier arms — none convicted as harmful; TTR (despite a reversed class direction) turns out to be the single most valuable feature. |
| `NB6e` | Extracts 11 new candidate statistical features (quote behaviour, function-word ratio, POS entropy, voice, clause structure, compressibility, Zipf deviation, sentence-opener diversity…) and merges them with the original 5 into a 16-feature file. |
| `NB6f` | Two-stage search over 3,797 feature subsets (screen on validation LOGO, report finalists on test) to find the best-performing combination without overfitting the evaluation split. |
| `NB6g` | Fixes an implementation bug in one searched feature and locks the final adopted feature set. |
| `NB7` | Re-evaluates the statistical-only baseline (old vs. revised feature set) under full LOGO — the direct empirical answer to the literature's claim that stylometric features generalize poorly to unseen generators. |

---

## Stage 4 — Track B (joint fine-tuning) & full LOGO

| Notebook | Purpose |
|---|---|
| `NB9` | Selects the chunk-count cap `K` for long articles from the corpus's actual length distribution. |
| `NB10` | The final hybrid model: CAMeLBERT-MSA fine-tuned end-to-end, jointly trained with the statistical-feature fusion head. |
| `NB14` | A matched neural-only twin (identical training recipe, no statistical features) — the controlled comparison for the fusion ablation. |
| `NB11` | Full six-fold LOGO for the fine-tuned hybrid vs. its neural-only twin. |
| `NB12` | Error-overlap diagnosis — which articles the two models disagree on, and why. |

---

## Stage 5 — Stress test

| Notebook | Purpose |
|---|---|
| `NB13` | Generates the robustness stress set: human-then-AI-polished and AI-then-human-edited articles at graded intensities. |
| `NB15` | Validity checks (semantic-preservation) on the stress set, then evaluates all models against it. |
| `NB16` | Bootstrap confidence intervals for the stress-test results; prepares the external Ar-APT comparison. |

---

## Stage 6 — External validation

| Notebook | Purpose |
|---|---|
| `NB17` | Evaluates the final model against the public Ar-APT benchmark for cross-study comparability, plus a final model-positioning table. |
| `NB18` | Resolves a length-confound issue in the zero-shot Fast-DetectGPT baseline. |

---

## Reference results ladder

Macro-F1 on the pair-aware test split. **LOGO** = Leave-One-Generator-Out (train on
five generators, test on the sixth); GPT-5 Mini is the hardest held-out generator for
every model.

| Model | Standard F1 | LOGO mean | LOGO worst (GPT) |
|---|---|---|---|
| Cheap-signal control | 68.8 | — | — |
| Statistical-only (GradientBoosting) | 84.7 | — | see `NB7` |
| AraBERTv2 frozen probe | 99.5 | 97.6 | 92.8 |
| AraELECTRA frozen probe | 99.5 | 96.2 | 83.5 |
| CAMeLBERT-MSA frozen probe (neural core) | 99.7 | 98.1 | 92.7 |
| Hybrid Track A — frozen, official feature set | 99.6 | 98.0 | 94.2 |

Later stages (fine-tuned hybrid, stress test, external validation) are reported in
the thesis results chapter.

---

## Conventions and contracts

Enforced across the pipeline; see individual notebooks for details.

- **Split:** pair-aware, by fact-card hash, 75/10/15 (train/val/test) — no card
  straddles a split boundary.
- **Alignment:** any notebook fusing multiple cached artifacts (dataset, statistical
  features, neural embeddings) asserts identical `article_id` ordering across all
  sources before use, and aborts on mismatch.
- **Leakage discipline:** scalers, imputation statistics, and the perplexity
  surrogate are fit on the **train** split only.
- **Labels:** `0 = human`, `1 = AI`. `class_weight='balanced'` throughout.
- **Determinism:** `seed = 42` everywhere.
- **Text normalization:** identical `normalize_format` applied to both classes
  before any split (an earlier version leaked on this).
- **Subset/hyperparameter search:** two-stage protocol — screen on validation,
  report only finalists on test — to avoid overfitting the evaluation split.
- **Long runs:** incremental checkpointing every N articles/configs, with a
  `RESUME_PATH` pattern, to survive Kaggle's 12-hour session limit and local power
  interruptions.

---

## License / citation

Add your institution's or your own preferred license here before publishing.
If this repository accompanies the published thesis, cite the thesis rather than
this repository directly.
