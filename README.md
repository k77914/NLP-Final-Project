# INLP 2026 Final Project — LLM Routing

Route each query to one of **11 candidate LLMs** (`Model_A`…`Model_K`) to maximize a cost-aware
reward. Given a query, the router predicts which model gives the best **performance-vs-cost**
trade-off and sends the query there.

**Metric — `Reward₀.₈₅ = 0.85 · mean(performance) − 0.15 · mean(cost) / C̄ₘₐₓ`**, where `C̄ₘₐₓ` is the
mean per-row maximum cost (≈ 0.0772 on train). Anchors: always-`Model_K` ≈ **0.450** (simple
baseline), oracle ≈ **0.673** (upper bound).

**Best result: public LB 0.47007** — a cost-aware utility-regression ensemble with a `Model_K`
confidence fallback (see *Results*).

---

## Repository layout

```
├── FinalProject_router_a100.ipynb   # validated router (the method behind 0.470) — Colab/A100
├── FinalProject_router.ipynb        # combined pipeline (thin orchestrator over src/) — Colab/A100
├── src/                             # tested Python package (re-implementation + experiments)
├── tests/                           # unit tests (pytest)
├── scripts/                         # build leaderboard-candidate submissions from saved predictions
├── docs/superpowers/                # design spec + step-by-step implementation plan
├── report/                          # report draft (answers the 3 graded questions)
├── Output/                          # result artifacts (curated; see "Outputs" below)
├── Project Specs.md                 # competition description & metric
└── requirements.txt
```

`src/` modules: `config` (paths/hyperparams) · `metric` (Reward₀.₈₅) · `data` · `features`
(handcrafted + TF-IDF + Qwen3 embeddings) · `cv` (stratified folds) · `models_classical`
(Ridge / LightGBM) · `models_encoder` (ModernBERT utility encoder) · `llm_difficulty`
(LLM self-consistency / judge features, vLLM **or** transformers backend) · `ensemble_routing`
(calibration, routing, honest cross-fit policy selection) · `combine` (`run_combined`) · `run`.

---

## Notebooks

### `FinalProject_router_a100.ipynb` — the validated router ⭐
The pipeline behind the best score. For each model it **regresses the cost-aware utility**
`Yᵢ = 0.85·perfᵢ − 0.15·costᵢ/C̄ₘₐₓ` directly, with four base learners trained in 5-fold CV:

| learner | input | model |
|---|---|---|
| `tfidf_ridge` | word + char TF-IDF | Ridge |
| `qwen3_embedding_ridge` | Qwen3-Embedding-4B ⊕ numeric | Ridge |
| `qwen3_embedding_lgbm` | Qwen3-Embedding-4B ⊕ numeric | LightGBM |
| `qwen3_embedding_lgbm_two_head` | Qwen3-Embedding-4B ⊕ numeric | LightGBM (perf + cost heads) |

It then blends them (grid-searched weighted blend → **0.7 / 0.2 / 0.1**, + a rank blend) and routes
`argmax(utility)` with a **`Model_K` margin-fallback** (route to the cheap default when the top-2
predicted utilities are within a small confidence margin). Writes its artifacts to
`Output/router_a100_full/` (a.k.a. `router_a100_exact_metric_v2`).

### `FinalProject_router.ipynb` — the combined pipeline (experiment)
A thin orchestrator over `src/` that **stacks the validated ensemble with two new members** — a
fine-tuned **ModernBERT-large** utility encoder and a **LightGBM on LLM-difficulty features**
(self-consistency + an LLM-as-judge rating) — under a strict cross-fit gate. Steps: setup (runs from
local Colab disk, restores caches from Drive) → **Step 1** LLM difficulty features → **Step 2**
`run_combined` (cross-fit CV + gate). **Result: the gate rejected both new members** (zero weight in
all folds); the validated router above remains best. Writes to `Output/router_v2/`.

---

## Outputs

Only lightweight **result** artifacts are versioned (~3.6 MB). Embeddings (`.npy`), pickled models
(`.joblib`), the raw dataset, and a duplicate download folder are git-ignored (regenerable).

### `Output/router_a100_full/` — validated-router artifacts
- `*_oof_predictions.npz` / `*_test_predictions.npz` — per-learner out-of-fold & test **utility
  predictions** (`tfidf_ridge`, `qwen3_embedding_ridge`, `qwen3_embedding_lgbm`,
  `qwen3_embedding_lgbm_two_head`, `weighted_ensemble_best`, `rank_blend_oof`).
- `submission.csv` — the weighted-ensemble routing (public LB **0.46853**).
- `submission_candidate_k_fallback_q05.csv` — **best submission, public LB 0.47007** (ensemble +
  5%-confidence `Model_K` fallback). `q045` / `q075` are nearby fallback thresholds.
- `model_comparison.csv`, `candidate_probe_report.csv` — method & threshold comparisons.
- `run_config.json`, `*_model_distribution.txt` — config and routed-model distributions.

### `Output/router_v2/` — combined-pipeline artifacts
- `submission_combined.csv` — combined-pipeline routing (public LB **0.46853**; collapses to the base
  ensemble because the gate dropped the new members).
- `combined_report.csv` — gate decision: `cv_old_only` vs `cv_combined`, member weights, stability.
- `combined_crossfit_folds.csv`, `policy_crossfit_folds.csv` — per-fold honest CV details.
- `model_comparison.csv`, `cache/llm_feats_*.parquet`, `cache/folds_*.npz`.

---

## Results

| Method | CV (OOF) | Public LB |
|---|---|---|
| Simple baseline — always `Model_K` | 0.4504 | ≈0.450 |
| `router_a100` weighted-ensemble (`argmax`) | ≈0.478 | 0.46853 |
| **+ `Model_K` q05 confidence fallback** | 0.478 | **0.47007** ⭐ |
| Correctness-first reframe (predict perf → route) | 0.4795 (in-sample) | 0.44936 |
| Correctness-first, honest cross-fit | 0.4631 | 0.44590 |
| Combined (+ ModernBERT + LLM-difficulty), gated | 0.4776 (= ensemble) | 0.46853 |
| Oracle (upper bound) | 0.6731 | — |

**Why the simple-looking method wins:** regressing the *cost-aware utility directly* avoids a
winner's curse (predicting performance then `argmax`-routing systematically over-routes to expensive
models whose accuracy is over-estimated — that variant scored 0.479 CV but **0.449** LB). The
`Model_K` fallback adds robustness on low-confidence queries. A ModernBERT encoder + LLM-as-judge
difficulty signal were tried but added **no transferable signal** (rejected by an honest cross-fit
gate). See `report/` for the full write-up.

---

## Reproduce

**Unit tests** (CPU; needs `numpy pandas scikit-learn scipy lightgbm pytest`):
```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```

**Validated router** — open `FinalProject_router_a100.ipynb` on Colab (A100), mount Drive with the
`dataset/` folder, and run all cells → writes `submission.csv` + per-learner predictions to
`Output/`. Build the K-fallback candidates from saved predictions with
`python scripts/build_router_a100_full_candidates.py` (the `q05` file is the 0.47007 submission).

**Combined pipeline** — open `FinalProject_router.ipynb` on Colab; run setup → Step 1 → Step 2.
GPU-heavy artifacts (embeddings, encoder, LLM features) are cached to Drive and reused.

> Not in the repo (git-ignored, regenerable): `dataset/*.csv`, `dataset.zip`, embeddings (`.npy`),
> and pickled models (`.joblib`).
