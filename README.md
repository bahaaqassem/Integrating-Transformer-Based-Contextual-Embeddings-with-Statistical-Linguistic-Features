# AI-Generated Arabic News Detection — Notebooks

MSc thesis project (Islamic University of Gaza, Faculty of Information Technology):
**"Integrating Transformer-Based Contextual Embeddings with Statistical Linguistic
Features to Detect AI-Generated Arabic News Reports."**

A hybrid binary classifier that fuses a fine-tuned CAMeLBERT-MSA document embedding
(`V_neural ∈ R^768`) with a five-dimensional vector of interpretable statistical
linguistic features (`V_stat ∈ R^5`) to distinguish human-written from AI-generated
long-form Modern Standard Arabic (MSA) news.

- **Author:** Bahaa Khalil Kamal Qassem (120240332)
- **Supervisor:** Dr. Rebhi S. Baraka
- **Compute:** Kaggle (P100/T4 GPU, 12-hour session limit)

> **Read this before quoting any number.** Phases 0–6 build the system and measure it
> on one training run. **Phase 7 retrains both models under further random seeds and
> re-measures everything**, and it changes the conclusion: most of the fusion effects
> reported in Phases 5 and 6 do not survive replication. See
> [What Phase 7 changed](#what-phase-7-changed).

---

## Table of contents

1. [Repository layout](#repository-layout)
2. [Pipeline](#pipeline)
3. [Phase 0 — Human dataset source fix](#phase-0--human-dataset-source-fix)
4. [Phase 1 — Dataset construction](#phase-1--dataset-construction)
5. [Phase 2 — Statistical features and encoder selection](#phase-2--statistical-features-and-encoder-selection)
6. [Phase 3 — Frozen hybrid and feature revision](#phase-3--frozen-hybrid-and-feature-revision)
7. [Phase 4 — Joint fine-tuning and full LOGO](#phase-4--joint-fine-tuning-and-full-logo)
8. [Phase 5 — Stress test](#phase-5--stress-test)
9. [Phase 6 — External validation and baselines](#phase-6--external-validation-and-baselines)
10. [Phase 7 — Seed replication and the matched operating point](#phase-7--seed-replication-and-the-matched-operating-point)
11. [What Phase 7 changed](#what-phase-7-changed)
12. [Artifact map](#artifact-map)
13. [Which notebook produced which thesis object](#which-notebook-produced-which-thesis-object)
14. [Reference results ladder](#reference-results-ladder)
15. [Conventions and contracts](#conventions-and-contracts)
16. [Reproducing one phase](#reproducing-one-phase)

---

## Repository layout

Folders on disk carry the phase numbering used throughout this document.

```
Phase 0/  NB0, NB0b
Phase 1/  NB2, NB3
Phase 2/  NB4, NB5, NB8
Phase 3/  NB6, NB7
Phase 4/  NB9, NB10, NB11, NB12, NB14
Phase 5/  NB13_stress_generation_parallel, NB15, NB16
Phase 6/  NB17, NB18
Phase 7/  NB19, NB20, NB21, NB22
```

---

## Pipeline

An arrow means "writes an artifact that the next notebook reads". Branches that meet
again are drawn as branches, because three of them are independent runs rather than a
chain.

```
PHASE 0   NB0 ──► NB0b ─────────────────────────────────────────► ha_corpus.parquet
                                                                        │
PHASE 1   NB2b ─► NB2b-post ─► NB2c ─► NB2d…NB2i ─► NB2j ─► NB3 ◄───────┘
                                                             │
                                                    dataset.parquet
                                                             │
              ┌──────────────────────────────────────────────┼───────────────┐
              │                                              │               │
PHASE 2   NB4 ─► NB5a                     NB5b ─► NB5b-LOGO ─► NB5c ─► NB8   │
              │                                              │               │
        scaler.pkl                        encoder choice: CAMeLBERT-MSA      │
              │                                                              │
PHASE 3   NB6 ─► NB6b ─► NB6c ─► NB6e ─► NB6f ─► NB6g ─► NB7                 │
                            │                                                │
             vstat16_scaled.parquet + scaler11.pkl                           │
                            │                                                │
PHASE 4   NB9 ─►  ┌── NB10  (hybrid, seed 42) ──┐                            │
                  └── NB14  (neural, seed 42) ──┴─► NB11 ─► NB12             │
                            │                                                │
                    two checkpoints                                          │
                            │                                                │
PHASE 5   NB13 ◄────────────┼────────────────────────────────────────────────┘
            │               │        (NB13 needs only dataset.parquet)
            └──────────────►NB15 ─► NB16
                                      │
PHASE 6                            NB17 ─► NB18
                                      │
PHASE 7   NB19 (run twice: SEED=1, SEED=2) ─► NB20 ─┐
                    │                               ├─► NB22
                    └──────────────► NB21 ──────────┘
```

Three properties of this graph are easy to miss:

- **NB10 and NB14 are two independent training runs, not a chain.** They share the
  recipe and differ only in the presence of the statistical inputs; NB11 consumes
  both. This is what makes the fusion comparison a matched one.
- **NB13 does not depend on any trained model.** It rewrites articles taken from
  `dataset.parquet`, so it can run before, during or after Phase 4.
- **Phase 7 trains only in NB19.** NB21 loads the six existing checkpoints and
  rescores; NB20 and NB22 are CPU-only.

---

## Phase 0 — Human dataset source fix

| Notebook | Input | Output | Purpose |
|---|---|---|---|
| `NB0_harvest_culturax` | CulturaX (`ar` subset, streamed) | `culturax_raw_harvest.parquet` (~15k docs) | Raw harvest from non-Western Arabic news domains (Al Jazeera, Al Arabiya, Al Mayadeen, RT Arabic, Anadolu, Arabi21, Al Quds, Al Araby, Maan, Wafa, Quds Press), filtered by ≥2 political keyword hits and 400–6000 words. **Zero text stripping** beyond whitespace, to replace an earlier human set that had been over-sanitized (digits, quotes, colons and most punctuation removed) and would have let a classifier cheat trivially. |
| `NB0b_select_corpus` | `culturax_raw_harvest.parquet` | `ha_corpus.parquet` (3,500 rows) | Line-level wrapper removal (Hijri date headers, share buttons, pipe tails, photo captions) with **reject-on-artifact** for any residual trace (stray pipe, "آخر تحديث", GMT stamp, UUID, relative timestamps). Keyword filter, content dedup, length-stratified sampling to exactly 3,500. |

**Verified fix:** digits appear in 90.0% of human articles (22.3 per article) against
88.8% for AI (13.9) — consistent with raw text rather than a sanitized set.

---

## Phase 1 — Dataset construction

| Notebook | Input | Output | Purpose |
|---|---|---|---|
| `NB2b_algorithmic_extraction` | `ha_corpus.parquet` | `extract_algo.parquet` | Model-free fact extraction: CAMeL-NER entities (chunk and union, word-aligned, `aggregation_strategy='first'`), numbers and dates by regex, agencies by dictionary, key sentences by TextRank, quote **events** (reporting verb plus a short span, never verbatim, so a generator cannot copy quotes as a shortcut), and `n_fact_points = clip(round(words/400), 3, 7)`. |
| `NB2b_post_entity_merge` | `extract_algo.parquet` | same file, overwritten | Merges Arabic prefix variants (و/ب/ل/ال…) under a safe rule: a prefix is stripped only when the stripped form already exists in the same list, so standalone names are never touched. 9.7% of raw mentions merged. |
| `NB2c` | `extract_algo.parquet` | fact cards | Assembles the per-article card that constrains generation. |
| `NB2d`–`NB2i` | fact cards | per-generator articles | Counter-generation across six generators (DeepSeek, Gemini, GPT-5 Mini, Claude Sonnet, Claude Opus, Qwen) from the same card as the paired human article. |
| `NB2j` | generator outputs | `aig_corpus.parquet` (3,601 rows) | Merge: keeps 104 duplicate cards, drops 258 failed generations out of 266 gate failures. |
| `NB3_build_dataset` | `ha_corpus.parquet`, `aig_corpus.parquet` | `dataset.parquet` (7,101 rows) | Symmetric normalization applied to **both** classes — this closes a newline-based leak that separated the classes at 99.9% with no linguistic content — then a pair-aware 75/10/15 split by fact-card hash. |

**Final dataset:** 3,500 human + 3,601 AI (deepseek 878, sonnet 840, qwen 622,
gemini 441, gpt 440, opus 380); train 5,363 / val 645 / test 1,093. Cheap-signal
control after the fix: 68.80% accuracy, 68.54 Macro-F1.

---

## Phase 2 — Statistical features and encoder selection

| Notebook | Purpose |
|---|---|
| `NB4` | Extracts the original five-feature `V_stat` (targeted perplexity via an AraGPT2 surrogate, burstiness, TTR, entity density, discourse coherence). Scaler fit on train only; writes `scaler.pkl`. |
| `NB5a` | Statistical-only baselines; GradientBoosting reaches 84.7 Macro-F1 against the 68.54 control. |
| `NB5b_arabert` | AraBERTv2 two ways: frozen `[CLS]` with a linear probe, and full fine-tuning. Caches `arabert_cls_frozen.npy`. |
| `NB5b_arabert_LOGO` | Six-fold LOGO on the **cached** frozen vectors, with no re-encoding. |
| `NB5c` | The same frozen-probe protocol for AraELECTRA and CAMeLBERT-MSA. **CAMeLBERT-MSA is selected** as the neural core. |
| `NB8` | Adds XLM-RoBERTa as a fourth frozen probe, against the multilingual-encoder literature. |

---

## Phase 3 — Frozen hybrid and feature revision

| Notebook | Purpose |
|---|---|
| `NB6` / `NB6b` | The first frozen-hybrid attempt failed a pre-registered criterion; `NB6b` diagnoses the cause as an under-trained fusion head rather than the features, and replaces the trainer. |
| `NB6c` | 27-configuration ablation of the original five features across two classifier arms. |
| `NB6e` | Extracts 11 further candidates (quote behaviour, function-word ratio, POS entropy, voice, clause structure, compressibility, Zipf deviation, sentence-opener diversity) and merges them with the original five into `vstat16_scaled.parquet`; writes `scaler11.pkl`. |
| `NB6f` | Two-stage search over 3,797 feature subsets: screen on validation LOGO, report finalists on test. |
| `NB6g` | Fixes an implementation bug in one searched feature and locks the adopted five: burstiness, TTR, quote ratio, function-word ratio, compressibility. |
| `NB7` | Re-evaluates the statistical-only baseline, original against revised set, under full LOGO. |

---

## Phase 4 — Joint fine-tuning and full LOGO

| Notebook | Purpose |
|---|---|
| `NB9` | Sets the chunk cap `K = 9` from the measured token-length distribution: 99% coverage, where `K = 4` would truncate 6.9% of articles. |
| `NB10` | The hybrid production model: CAMeLBERT-MSA fine-tuned end to end with the fusion head. Writes `ckpt/last.pt` and the tokenized chunk cache. |
| `NB14` | The matched neural-only twin: identical recipe, no statistical inputs. Writes `ckpt_neural/last.pt`. |
| `NB11` | Six-fold LOGO for both models. |
| `NB12` | Error-overlap diagnosis: the two models disagree on six articles out of 3,788, with a net effect of zero. |

---

## Phase 5 — Stress test

| Notebook | Purpose |
|---|---|
| `NB13` | Generates the stress set: human text polished by a model at four intensities, and machine text adversarially rewritten. 820 transformed articles. |
| `NB15` | Loads both checkpoints and scores the stress set together with its untransformed controls. |
| `NB16` | Validity gate (garbled and no-op removed, 748 of 820 kept), the paired bootstrap, and preparation of the Ar-APT subset. |

---

## Phase 6 — External validation and baselines

| Notebook | Purpose |
|---|---|
| `NB17` | Published baselines and the Ar-APT external benchmark; the system-positioning table. |
| `NB18` | Diagnoses the zero-shot Fast-DetectGPT failure: its statistic is not length-normalized, and length alone reaches AUC 66.22 on the Ar-APT balanced set. |

---

## Phase 7 — Seed replication and the matched operating point

Phases 4 to 6 report one training run of each model. The paired bootstrap used there
resamples **articles**, so it quantifies the uncertainty of the evaluation sample and
not the uncertainty of training. Phase 7 measures the second kind, and then asks
whether what remains is a difference in the decision rule or only in the operating
point.

| Notebook | Hardware | Reads | Writes | Runtime |
|---|---|---|---|---|
| `NB19` | GPU | `dataset.parquet`, `stress_generations.jsonl` (NB13), `scaler.pkl`, `scaler11.pkl`, `vstat16_scaled.parquet`, `chunks_K9.npz` (NB10, optional) | `nb19_seed{S}_scores.parquet`, `nb19_hybrid_seed{S}.pt`, `nb19_neural_seed{S}.pt` | ~3.7 h per seed |
| `NB20` | CPU | `nb15_stress_results.parquet` (NB15), both `nb19_seed{S}_scores.parquet`, `stress_generations_meta.parquet` (NB16) | `nb20_all_seed_scores.parquet`, `nb20_per_seed_rates.parquet`, `nb20_bootstrap.parquet` | minutes |
| `NB21` | GPU | `arapt_prepared.parquet`, `arapt_vstat.parquet`, `arapt_chunks_K9.parquet` (all NB16), and the six checkpoints from NB10, NB14 and NB19 | `nb21_arapt_seed_scores.parquet`, `nb21_arapt_per_seed.parquet`, `nb21_arapt_bootstrap.parquet` | ~20 min |
| `NB22` | CPU | `nb20_all_seed_scores.parquet`, `nb21_arapt_seed_scores.parquet` | `nb22_threshold_drift.parquet`, `nb22_threshold_matched.parquet`, `nb22_threshold_bootstrap.parquet`, `nb22_disagreements.parquet` | minutes |

**Order of execution.**

1. `NB19` with `SEED = 1`, Save Version; then `SEED = 2`, Save Version. Seed 42 is
   already covered by NB10, NB14 and NB15 and is not retrained.
2. `NB20` once both seed files exist.
3. `NB21` after adding the two NB19 checkpoint files as inputs.
4. `NB22` last, since it needs NB20 and NB21 together.

**Guards.** `NB19` recomputes the five features on 150 articles and aborts unless
they reproduce `vstat16_scaled.parquet` exactly; it also asserts that the chunk cache
matches the dataset length and the value of `K`. `NB21` aborts if any (seed, model)
pair produced no scores. `NB22` aborts if the Ar-APT anchor holds fewer than 200
articles. These guards exist because two silent failures were caught this way: a
retyped function-word list, and a checkpoint missing from the input list.

---

## What Phase 7 changed

Both models were retrained from scratch under two further seeds and everything was
re-measured. Gaps below are hybrid minus neural, in percentage points.

| Axis | Seed 42 | Seed 1 | Seed 2 | Sign held |
|---|---|---|---|---|
| Polished human text, study dataset | −3.49 | −4.65 | +0.58 | no |
| Adversarially rewritten machine text | −2.73 | −3.96 | −1.24 | yes |
| Untransformed human text, Ar-APT | −7.25 | −9.25 | +5.50 | no |
| Untransformed machine text, Ar-APT | −9.50 | −1.50 | +4.50 | no |
| Polished human text, Ar-APT | −11.83 | −14.51 | +6.46 | no |

Four of the five axes reverse sign between seeds. Matching the neural threshold to
the hybrid on the 400 untransformed Ar-APT human articles reduces ten significant
differences out of twelve to four, and nine of the twelve shrink by at least half.
The threshold the hybrid model needs for a fixed false-alarm rate moves across seeds
by 4.7 to 9.3 times the movement of the neural threshold.

The one effect that keeps its sign in all three seeds is the cost the hybrid model
pays on adversarially rewritten text, and it is carried by a single rewriting model:
against Claude it is −15.19, −11.39 and −6.33 points; against Gemini, GPT and Qwen it
is exactly zero in every seed. The articles on which the two models disagree are also
not the same from seed to seed: 41 polished articles are involved across the three
seeds and 4 are common to all three, a Jaccard overlap of 0.098.

**What this means for the earlier notebooks.** Their figures are correct for the seed
that produced them and are not stable across training runs. Statistical fusion does
not change what the detector can separate; it shifts the decision boundary and makes
the position of that boundary less stable. A deployment should prefer the
neural-only model and obtain the false-alarm rate it needs from the operating point
rather than from added features.

---

## Artifact map

Every cached file, the notebook that writes it, and the notebooks that read it.

| Artifact | Written by | Read by |
|---|---|---|
| `ha_corpus.parquet` | NB0b | NB2b, NB3 |
| `extract_algo.parquet` | NB2b, NB2b-post | NB2c |
| `aig_corpus.parquet` | NB2j | NB3 |
| `dataset.parquet` | NB3 | NB4, NB5*, NB6*, NB9, NB10, NB13, NB14, NB19 |
| `scaler.pkl` | NB4 | NB15, NB19 |
| `arabert_cls_frozen.npy` | NB5b | NB5b-LOGO |
| `vstat16_scaled.parquet`, `scaler11.pkl` | NB6e | NB6f, NB6g, NB7, NB10, NB15, NB19 |
| `chunks_K9.npz` (keys `ch`, `nch`) | NB10 | NB14, NB19 |
| `ckpt/last.pt` — hybrid, seed 42 | NB10 | NB11, NB15, NB17, NB21 |
| `ckpt_neural/last.pt` — neural, seed 42 | NB14 | NB11, NB15, NB17, NB21 |
| `stress_generations.jsonl` | NB13 | NB15, NB19 |
| `nb15_stress_results.parquet` | NB15 | NB16, NB20 |
| `stress_generations_meta.parquet` | NB16 | NB20 |
| `arapt_prepared.parquet`, `arapt_vstat.parquet`, `arapt_chunks_K9.parquet` | NB16 | NB17, NB21 |
| `nb17_positioning_table.parquet`, `nb17_fastdetectgpt_arapt.parquet` | NB17 | NB18 |
| `nb19_seed{S}_scores.parquet`, `nb19_{hybrid,neural}_seed{S}.pt` | NB19 | NB20, NB21 |
| `nb20_all_seed_scores.parquet` | NB20 | NB22 |
| `nb21_arapt_seed_scores.parquet` | NB21 | NB22 |

---

## Which notebook produced which thesis object

| Thesis object | Notebook |
|---|---|
| Table 4.1 composition of the assembled dataset | NB3 |
| Table 4.2 token-length distribution and window coverage | NB9 |
| Table 4.3 training configuration | NB10 |
| Table 4.4 design of the stress test | NB13 |
| Table 5.1 frozen-probe comparison of four encoders | NB5c, NB8 |
| Table 5.2 result of the subset search | NB6f |
| Table 5.3 the statistical features alone | NB7 |
| Table 5.4 reference configurations under two fusion regimes | NB10 |
| Table 5.5 fusion benefit against a matched baseline | NB10, NB14 |
| Table 5.6 standard test split | NB10, NB14 |
| Table 5.7 LOGO results | NB11 |
| Table 5.8 agreement across the six folds | NB12 |
| Tables 5.9 to 5.12 stress test and low-false-alarm operating points | NB15, NB16 |
| Tables 5.13 to 5.15 Ar-APT control, polished subsets, feature means | NB16, NB17 |
| **Table 5.16 the two effects under three seeds** | **NB19, NB20, NB21** |
| **Table 5.17 default against matched operating point** | **NB22** |
| Table 5.18 Fast-DetectGPT with an open-weight surrogate | NB17 |
| Table 5.19 length diagnosis of the zero-shot statistic | NB18 |
| Table 5.20 all systems on the held-out test split | NB17 |
| Figure 4.1 formatting normalization and the pair-aware split | NB3 |
| Figure 4.2 the five adopted features as implemented | NB4, NB6e |
| Figure 4.3 sliding-window segmentation | NB10 |
| Figure 4.4 the hybrid model | NB10 |
| Figure 4.5 LOGO folds and the matched baseline | NB11 |
| Figure 4.6 the paired bootstrap | NB16 |
| Figures 5.1 to 5.4 result plots | NB11, NB15, NB17 |
| Figures C.1 to C.7 appendix listings | NB10, NB6e, NB16, NB18 |

---

## Reference results ladder

Macro-F1 on the pair-aware test split. **LOGO** = Leave-One-Generator-Out (train on
five generators, evaluate on the sixth).

| Model | Standard | LOGO mean | LOGO worst |
|---|---|---|---|
| Cheap-signal control | 68.54 | — | — |
| Statistical only, revised feature set | 88.75 | — | 74.37 |
| AraBERTv2 frozen probe | 99.5 | 97.6 | 92.8 |
| AraELECTRA frozen probe | 99.5 | 96.2 | 83.5 |
| CAMeLBERT-MSA frozen probe | 99.7 | 98.1 | 92.7 |
| Hybrid, frozen encoder | 99.6 | 98.0 | 94.2 |
| Neural only, fine-tuned (seed 42) | 99.88 | — | 98.36 |
| Hybrid, fine-tuned (seed 42) | 99.76 | — | 98.83 |

The last two differ on six articles out of 3,788 across the whole LOGO protocol, a
difference that sits at the noise floor of 0.42 points. Across the three seeds the
fine-tuned models reach 99.76, 100.00 and 99.91 on the standard split: in-distribution
accuracy is stable, the decision boundary is not.

---

## Conventions and contracts

- **Split:** pair-aware, by fact-card hash, 75/10/15 — no card straddles a boundary.
- **Alignment:** any notebook fusing cached artifacts asserts identical `article_id`
  ordering across all sources and aborts on mismatch.
- **Leakage discipline:** scalers, imputation statistics and the perplexity surrogate
  are fit on the **train** split only.
- **Labels:** `0 = human`, `1 = AI`; `class_weight='balanced'` throughout.
- **Seeds:** Phases 0–6 use `seed = 42` throughout; Phase 7 varies it deliberately.
- **Text normalization:** identical `normalize_format` applied to both classes before
  any split.
- **Search protocol:** screen on validation, report only finalists on test.
- **Long runs:** incremental checkpointing with a `RESUME_PATH` pattern, to survive
  the 12-hour session limit.
- **File format:** cached artifacts are `.parquet`, except the tokenized chunk cache
  (`.npz`, a large integer tensor read once) and model checkpoints (`.pt`).
- **Logging:** every notebook prints a line after each operation, so a committed run
  can be followed from the Kaggle log alone.

---

## Reproducing one phase

Each phase reads the previous phase's outputs as Kaggle datasets or notebook outputs.
To reproduce a single phase without rerunning the others, add the artifacts listed for
it in the [Artifact map](#artifact-map) as inputs and run the notebook top to bottom.
Every notebook is written for a single committed pass and never depends on cells being
run out of order.

---

## License and citation

Add the institution's or the author's preferred license here before publishing. If
this repository accompanies the published thesis, cite the thesis rather than the
repository.
