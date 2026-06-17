# Design: Correctness-First LLM Router

- **Date:** 2026-06-16
- **Author:** Chia-Chen Hsieh (with Claude)
- **Status:** Approved (design); pending implementation plan
- **Deadline:** 2026-06-26 23:59 (10 days)
- **Current public LB:** 0.46853 → target: clearly beat the strong baseline, optimize private LB

## 1. Problem & Objective

LLM Routing: given a user `query`, pick exactly one of 11 candidate models (`Model_A`…`Model_K`) to answer it. Submit `(ID, pred_model)` for 2,550 test rows.

**Metric:** `Reward_0.85 = 0.85 · mean(performance) − 0.15 · mean(cost) / denom`, where `denom = mean over train rows of (row-max cost) = 0.07721` (fixed constant). Higher is better.

**Objective:** maximize the **private** leaderboard reward (70% of the Kaggle grade; public is only 30% and on a different 50% split). Score-focused; methodology clarity is secondary but a CV methods table is kept as a near-free report byproduct.

**Constraints (validated from project spec):**
- "There is no limit to the method you can use." Open-source models allowed.
- Hardware: Colab A100 (strong GPU). Fine-tuning + vLLM inference feasible.
- Submissions: **max 3/day, choose 2 for private LB** → trust CV, do not LB-probe.
- Data source: NVIDIA (heterogeneous benchmark mixture).

## 2. Key empirical findings (diagnostics)

Per-model stats (train, N=10,182):

| model | accuracy | mean_cost | reward_if_always |
|---|---|---|---|
| Model_K | 0.532 | 0.0011 | **0.4504** |
| Model_H | 0.583 | 0.0371 | 0.4237 |
| Model_F | 0.589 | 0.0657 | 0.3731 |
| Model_B | 0.472 | 0.0151 | 0.3719 |
| Model_D | 0.435 | 0.0015 | 0.3667 |
| Model_J | 0.436 | 0.0025 | 0.3655 |
| Model_C | 0.408 | 0.0010 | 0.3451 |
| Model_E | 0.425 | 0.0085 | 0.3446 |
| Model_G | 0.423 | 0.0142 | 0.3322 |
| Model_I | 0.366 | 0.0035 | 0.3046 |
| Model_A | 0.383 | 0.0144 | 0.2977 |

**Routing ceilings:**
- Always Model_K: **0.4504** (best fixed model)
- Current public LB: **0.46853**
- Oracle (argmax true reward): **0.67307**
- Accuracy-oracle (cheapest *correct* model, else cheapest): **0.67249**
- True-perf + **constant per-model cost**: **0.67216**

**Decisive structural facts:**
1. **Oracle ≈ accuracy-oracle ≈ constant-cost oracle** (0.673 ≈ 0.672 ≈ 0.672). Per-query cost modeling is worthless; cost reduces to "prefer the cheaper model among likely-correct ones." Treat **cost as a per-model constant**.
2. **Difficulty is bimodal:** 17.5% of rows solved by all 11 models, 19.1% by none; 80.9% solvable. On both extremes the optimal route is just "a cheap model." Model identity matters only on the middle ~63%.
3. **Model_K is the oracle's pick 51% of the time** (correct 53% at ~zero cost). It is wrong on 46% of rows; on **2,750** of those some other model is right — routing those correctly is worth **+0.23 reward**. That is the entire baseline→oracle gap.
4. **Heterogeneous mixture, not pure math.** Composition (train): MCQ ~4.3% (avg model acc **0.142**, far harder), LaTeX math ~15%, code ~7.7%, long-doc (>5k chars) ~4.4% (acc 0.17–0.32), 64% short (<300 chars, easier). Difficulty correlates strongly with `is_mcq`, `is_long_doc`, and length.
5. **No train/test distribution shift** (median len 156 vs 150; matching tails). Skew is task-inherent → handled by features + stratified CV + bimodal-aware routing, not external data.

## 3. Core approach: correctness-first reframe

Replace the current notebook's multi-output **reward regression** with **per-model correctness classification**:

- **Predict** `p̂_i = P(model_i solves query)` for i=1..11 (multilabel binary).
- **Cost** = per-model constant `c̄_i` = train mean cost.
- **Route:** `argmax_i [ 0.85·p̂_i − 0.15·c̄_i/denom + b_i ]`, where `b_i` are per-model additive bias offsets **tuned on out-of-fold (OOF) predictions to maximize the exact metric**. The bias search is the explicit performance↔cost balance knob (answers report Q2).

Rationale: the metric is a correctness game with a cheap-model tiebreak (finding #1–#3 above). Calibrated probabilities + explicit cost tiebreak match the metric's structure; reward regression smears the binary signal and barely beats always-K.

## 4. Architecture & components (modular)

### (a) Data & CV layer — `data.py`, `cv.py`, `metric.py`
- Load train/test/sample; build `Y_perf` (N×11 binary), `c̄` (11,), difficulty `= Σ perf`, `denom`.
- **One fixed 5-fold `StratifiedKFold`** on a difficulty×`is_mcq` bucket; fold indices **saved to disk and shared by every base learner** (required for valid OOF stacking).
- `metric.py`: exact reward functions + **self-tests** asserting always-K ≈ 0.450376 and oracle ≈ 0.673071.

### (b) Feature/representation layer — `features.py`
- **Handcrafted** (cheap, high-signal): char/word/line counts + logs, `is_mcq`, `n_choices`, `is_long_doc`, latex/math signals, code signals, digit/question-mark counts, length buckets.
- **TF-IDF** (word 1–2, char 3–5), optionally Truncated-SVD reduced.
- **Frozen embeddings:** reuse cached **Qwen3-Embedding-4B** (general-domain); preserve cache-reuse logic from the current notebook.
- **LLM-difficulty features** (Phase 2): see (e).

### (c) Base learners — `models_classical.py`, `models_encoder.py`
Each emits OOF (N×11) + test (M×11) probability matrices on the shared folds:
1. **LightGBM** binary multilabel (one head per model) on `[embeddings ⊕ TF-IDF-SVD ⊕ handcrafted ⊕ LLM features]`. `objective='binary'` for probabilities.
2. **Logistic/linear** on embeddings (cheap, diverse member).
3. **Fine-tuned ModernBERT-large** (Phase 1): shared trunk, **11-logit multilabel head + auxiliary difficulty-regression head** (predict #correct/11), head+tail truncation (8192-token window covers ~p98; truncate the long tail), bf16 on A100, 5-fold OOF + final full-data fit for test predictions.

### (d) Calibration + ensemble — `ensemble_routing.py`
- Per-model **isotonic calibration** on OOF (so the `0.85·p̂ − cost` tradeoff is metrically correct).
- **OOF-tuned weighted average** of base learners (start here, robust); optional per-model **logistic meta-stacker** in Phase 3.
- Output final `p̂` (OOF + test).

### (e) LLM-as-difficulty — `llm_difficulty.py` (Phase 2)
General reasoner (**default Qwen2.5-7B-Instruct**, scalable to 14B/32B-AWQ) via **vLLM**; all outputs **cached to disk**:
- **Self-consistency:** k≈6 samples; extract final answer (numeric / MCQ-letter / short span) → agreement fraction, entropy, #distinct. Gold-free difficulty proxy.
- **LLM-as-Judge:** one call for a 1–10 difficulty rating + "would a mid-tier model solve this?" probability.
- **Generation stats:** solution length mean/std, refusal/uncertainty markers, mean sequence logprob (perplexity).
Produces ~10 features per query, fed into (c1) and (c3). General (not math-specialized) model because the data spans math/MCQ/code/trivia/long-doc.

### (f) Routing layer — `ensemble_routing.py`, `run.py`
Compute expected reward, run the `b_i` bias search on OOF, apply to test, write `submission.csv` with format assertions (ID order, valid model names, row count).

## 5. Data flow

```
query
  → [handcrafted ⊕ TF-IDF ⊕ Qwen3-emb ⊕ LLM-difficulty]
  → {LightGBM, linear, ModernBERT-large}        (each → p̂ per model)
  → per-model isotonic calibration
  → OOF-tuned weighted/stacked ensemble          (→ final p̂)
  → expected reward 0.85·p̂ − 0.15·c̄/denom + b_i  (b_i OOF-tuned to exact metric)
  → argmax → pred_model
```

## 6. CV & evaluation protocol

- All decisions judged by **OOF exact reward** on the shared 5-fold split.
- Submit only when OOF improves; record public LB **only** to monitor CV↔LB correlation. Optimize CV/private; never chase public.
- Maintain a `model_comparison.csv` (method × OOF reward, mean perf, mean cost, model distribution) — doubles as the report's comparison table.
- Budget: ≤3 submissions/day; reserve final 2 picks for the configs with best OOF reward + sane CV↔LB gap.

## 7. Phasing — each phase independently submittable

| Phase | Deliverable | Expected |
|---|---|---|
| **0. Harness + reframe** | data/CV/metric layer (+ baseline self-tests), handcrafted+TF-IDF+cached-emb → LGBM → calibrate → bias-tuned routing | Beat 0.469 clearly |
| **1. Fine-tuned encoder** | ModernBERT-large multilabel+difficulty head, ensembled with Phase 0 | Main jump |
| **2. LLM difficulty** | vLLM self-consistency/judge features folded into LGBM + encoder stack | High-ceiling push |
| **3. Polish** | meta-stacker, bias-offset refinement, pick the 2 private-LB submissions | Final |

## 8. Routing / metric optimization detail

- Expected reward per model: `e_i = 0.85·p̂_i − 0.15·c̄_i/denom`.
- Search per-model additive bias `b_i` (coordinate ascent / small grid) on OOF to maximize exact reward; equivalent to per-model decision thresholds. Apply learned `b_i` to test.
- Because cost is a constant tiebreak, expect `b_i` to encode "how confident must model_i be to beat the cheap default" — directly the perf/cost balance.

## 9. Robustness, caching, determinism

- **Disk caching**, keyed by config, for: embeddings, LLM generations, every OOF/test probability matrix. Colab disconnects don't lose work; each phase reruns cheaply.
- **Determinism:** fixed seeds; saved fold indices; deterministic feature builders.
- **Graceful degradation:** if vLLM/model download fails, proceed without Phase-2 features. Each phase runs standalone.
- Preserve Colab/Drive path handling and cache-reuse from the current notebook.

## 10. Testing

- **Metric self-tests:** assert always-K ≈ 0.450376 and oracle ≈ 0.673071.
- **Smoke mode:** tiny-subset end-to-end run before each full A100 run.
- **Submission-format assertions:** ID order matches sample, valid model names, exactly 2,550 rows.

## 11. Deliverable / repo structure

Develop as modular `src/` package **plus a thin orchestrating notebook** (keeps Colab/Drive paths; the notebook is the E3-friendly entry point):

```
src/
  config.py            # CFG dataclass, paths (Colab/Drive + local), seeds
  data.py              # load + targets + cost constants + denom
  metric.py            # exact reward + self-tests
  cv.py                # stratified folds, saved indices, OOF helpers
  features.py          # handcrafted + TF-IDF + frozen-embedding loaders
  llm_difficulty.py    # vLLM self-consistency / judge / gen-stats (cached)
  models_classical.py  # LightGBM + linear base learners
  models_encoder.py    # ModernBERT-large fine-tune (multilabel + difficulty head)
  ensemble_routing.py  # calibration, ensemble, bias-tuned routing, submission
  run.py               # orchestrator: phase selection, caching, comparison table
FinalProject_router.ipynb   # thin notebook importing src/, runs phases
```

A flatten step produces the single-file `FinalProject_<TeamID>.ipynb`/`.py` for E3 at the end.

## 12. Optional / stretch: external source-tagging (DEFERRED)

Not built initially. Documented for later if Phases 0–2 land with time and OOF headroom.

- **Idea:** near-duplicate match each query against reference corpora to tag its **probable source benchmark** (a difficulty/type *feature*, not labels). Generalizes if the private split draws from the same benchmarks.
- **Corpora by observed type:** Math → MATH/GSM8K/AIME/NuminaMath/Omni-MATH; hard MCQ → GPQA/MMLU-Pro/ARC-Challenge; code → HumanEval/MBPP/LiveCodeBench/APPS; trivia → TriviaQA/NaturalQuestions/BBH; long-doc → QuALITY/NarrativeQA.
- **Not pursued:** importing more labeled rows (impossible — labels are tied to the 11 anonymized models) and answer-key/model-identity leakage (gold answers don't reveal which anonymized model is correct; unreliable).
- **Gate:** only adopt if it lifts OOF reward and holds up in CV.

## 13. Report mapping (near-free byproduct)

- Q1 (implementation): packages, framework, losses (BCE + auxiliary difficulty MSE), hyperparameters — captured in `config.py` + run logs.
- Q2 (perf/cost balance): the per-model bias-offset search and its effect on mean perf vs mean cost.
- Q3 (methods comparison): `model_comparison.csv` (per-method OOF reward), best method + why.

## 14. Risks & mitigations

- **Encoder overfits noisy single-sample labels** → auxiliary difficulty head, calibration, ensemble, OOF-only model selection.
- **Public↔private gap** → trust OOF; monitor CV↔LB correlation; pick final 2 by OOF.
- **Phase-2 LLM inference too slow** → cap tokens, batch with vLLM, smaller model, cache; Phase-2 is optional over Phases 0–1.
- **Long-query tail** → head+tail truncation for encoder; embeddings already chunk.

## 15. Resolved decisions

- Base encoder: **ModernBERT-large** (8192 context; head+tail truncation).
- LLM-difficulty model: **Qwen2.5-7B-Instruct** default, scale to 14B/32B-AWQ if time allows.
- Deliverable: **modular `src/` + thin notebook**; flatten for E3.
- Source-tagging: **deferred** (documented stretch, §12).
- Cost: **per-model constant** (validated negligible loss vs true cost).
