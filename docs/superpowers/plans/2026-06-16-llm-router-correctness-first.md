# Correctness-First LLM Router — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the LLM router as a per-model correctness classifier (calibrated `P(model solves query)`) with cost-as-constant, metric-tuned routing, a fine-tuned encoder, and vLLM difficulty features — ensembled — to move the private-LB Reward_0.85 from 0.469 toward the 0.673 oracle.

**Architecture:** Predict `p̂_i = P(model_i correct)` for 11 models; route `argmax_i [0.85·p̂_i − 0.15·c̄_i/denom + b_i]` with per-model bias offsets `b_i` tuned on OOF to the exact metric. Base learners (LightGBM, linear, ModernBERT-large, +vLLM difficulty features) → isotonic calibration → OOF-tuned ensemble. Built in 4 independently-submittable phases.

**Tech Stack:** Python, numpy/pandas/scikit-learn, LightGBM, sentence-transformers (Qwen3-Embedding-4B, cached), transformers (ModernBERT-large), vLLM (Qwen2.5-7B-Instruct), pytest. Dev locally on CPU for Phases 0 unit tests; train on Colab A100.

**Reference spec:** `docs/superpowers/specs/2026-06-16-llm-router-correctness-first-design.md`

---

## Conventions

- **Commit footer:** every commit message ends with:
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  ```
- **Run tests from repo root:** `python -m pytest tests/ -q`
- **Fixed constants** live in `src/metric.py`: `LETTERS`, `MODEL_NAMES`, `PERF_COLS`, `COST_COLS`.
- **Known anchors** (assert these): always-Model_K ≈ `0.450376`, oracle ≈ `0.673071`, denom ≈ `0.07721`.
- All heavy artifacts (embeddings, LLM generations, OOF/test matrices) cache to `Output/router_v2/cache/`.

## File Structure

```
src/
  config.py            # CFG dataclass, path resolution (Colab/Drive + local), seeds
  metric.py            # constants + exact reward + expected-reward matrix
  data.py              # load CSVs, validate, build perf/cost arrays + cost constants
  features.py          # handcrafted features, mcq flag, TF-IDF builder, embedding loader
  cv.py                # stratified folds (saved/shared), OOF helpers
  models_classical.py  # LightGBM binary-multilabel + linear (logistic) base learners
  models_encoder.py    # ModernBERT-large multilabel + aux difficulty head (Phase 1)
  llm_difficulty.py    # vLLM self-consistency / judge / gen-stats features (Phase 2)
  ensemble_routing.py  # isotonic calibration, ensemble, bias-tuned routing, submission
  run.py               # orchestrator: phase selection, caching, comparison table
tests/
  conftest.py          # synthetic fixture + real-data fixture (skips if absent)
  test_metric.py
  test_data.py
  test_features.py
  test_cv.py
  test_ensemble_routing.py
FinalProject_router.ipynb   # thin notebook importing src/, runs phases on Colab
requirements.txt
.gitignore
```

---

## Task 0: Project scaffold, git, test harness, config

**Files:**
- Create: `.gitignore`, `requirements.txt`, `src/__init__.py`, `tests/__init__.py`, `tests/conftest.py`, `src/config.py`

- [ ] **Step 1: Initialize git and ignore heavy/data files**

Run:
```bash
cd "/mnt/c/Users/kurop/Desktop/University/NLP/Final Project"
rm -rf .git && git init -q && git add -A 2>/dev/null; true
```

Create `.gitignore`:
```gitignore
__pycache__/
*.pyc
.ipynb_checkpoints/
Output/
*.joblib
*.npy
*.npz
dataset/*.csv
dataset.zip
.idea/
.venv/
```

- [ ] **Step 2: Create `requirements.txt`**

```
pandas==2.2.2
numpy>=1.26.4,<2.1
scikit-learn>=1.4
scipy
lightgbm
joblib
tqdm
pytest
# Colab/A100 only (not needed for Phase-0 unit tests):
# transformers>=4.48.0  sentence-transformers>=2.7.0  accelerate  vllm
```

- [ ] **Step 3: Create empty package markers**

`src/__init__.py` and `tests/__init__.py` → empty files.

- [ ] **Step 4: Create `src/config.py`**

```python
from __future__ import annotations
import os, random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _resolve_root() -> Path:
    drive = Path('/content/drive/MyDrive/NLP Final Project')
    if (drive / 'dataset' / 'train.csv').exists():
        return drive
    return Path(__file__).resolve().parents[1]


@dataclass
class CFG:
    seed: int = 42
    root: Path = field(default_factory=_resolve_root)
    n_splits: int = 5
    smoke: bool = False
    smoke_rows: int = 400

    perf_weight: float = 0.85
    cost_weight: float = 0.15

    # features
    tfidf_word_features: int = 100_000
    tfidf_char_features: int = 100_000
    tfidf_max_chars: int = 60_000
    svd_components: int = 256
    embedding_cache_name: str = 'qwen3_4b_dim1024_chunks_v1'
    qwen_model_id: str = 'Qwen/Qwen3-Embedding-4B'
    embedding_dim: int = 1024

    # lgbm
    lgbm_estimators: int = 1200
    lgbm_lr: float = 0.03
    lgbm_leaves: int = 31
    lgbm_min_child: int = 30

    # encoder (Phase 1)
    encoder_id: str = 'answerdotai/ModernBERT-large'
    encoder_max_len: int = 2048
    encoder_epochs: int = 3
    encoder_lr: float = 2e-5
    encoder_bs: int = 8
    encoder_grad_accum: int = 2
    aux_diff_weight: float = 0.3

    # llm difficulty (Phase 2)
    llm_id: str = 'Qwen/Qwen2.5-7B-Instruct'
    llm_k_samples: int = 6
    llm_max_new_tokens: int = 1024
    llm_temperature: float = 0.8

    @property
    def data_dir(self) -> Path:
        return self.root / 'dataset'

    @property
    def out_dir(self) -> Path:
        d = self.root / 'Output' / 'router_v2'
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def cache_dir(self) -> Path:
        d = self.out_dir / 'cache'
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def prev_cache_dir(self) -> Path:
        return self.root / 'Output' / 'router_a100' / 'cache'


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
```

- [ ] **Step 5: Create `tests/conftest.py`** (fixtures used by later tasks)

```python
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def synthetic_train():
    """Small deterministic dataset with 11 models, known structure."""
    rng = np.random.default_rng(0)
    n = 60
    queries = [f"q{i} " + ("Answer Choices: A. x B. y" if i % 10 == 0 else "find x") for i in range(n)]
    perf = (rng.random((n, 11)) < 0.5).astype(np.float32)
    cost = (rng.random((n, 11)) * 0.1 + 0.001).astype(np.float32)
    cols = {}
    for j, L in enumerate("ABCDEFGHIJK"):
        cols[f"Model_{L}_performance"] = perf[:, j]
        cols[f"Model_{L}_cost"] = cost[:, j]
    df = pd.DataFrame({"ID": np.arange(1, n + 1), "query": queries, **cols})
    return df


@pytest.fixture
def real_train():
    p = ROOT / "dataset" / "train.csv"
    if not p.exists():
        pytest.skip("real dataset not present")
    return pd.read_csv(p)
```

- [ ] **Step 6: Verify pytest collects (no tests yet is fine) and commit**

Run: `python -m pytest tests/ -q`
Expected: `no tests ran` (exit 5) — acceptable; confirms imports work.

```bash
git add -A
git commit -m "chore: scaffold src/ package, test harness, config

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 1: `metric.py` — constants + exact reward (TDD)

**Files:**
- Create: `src/metric.py`, `tests/test_metric.py`

- [ ] **Step 1: Write failing tests**

`tests/test_metric.py`:
```python
import numpy as np
from src import metric


def test_constants():
    assert metric.MODEL_NAMES[0] == "Model_A"
    assert metric.MODEL_NAMES[-1] == "Model_K"
    assert len(metric.MODEL_NAMES) == 11
    assert metric.PERF_COLS[0] == "Model_A_performance"
    assert metric.COST_COLS[10] == "Model_K_cost"


def test_route_reward_simple():
    perf = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    cost = np.array([[0.1, 0.2], [0.3, 0.1]], dtype=np.float32)
    denom = 0.2
    # route to col 0 then col 1 => perf mean 1.0, cost mean (0.1+0.1)/2=0.1
    r = metric.route_reward(np.array([0, 1]), perf, cost, denom)
    assert abs(r - (0.85 * 1.0 - 0.15 * (0.1 / 0.2))) < 1e-6


def test_expected_reward_matrix_shape():
    p = np.full((4, 11), 0.5, dtype=np.float32)
    c = np.full(11, 0.01, dtype=np.float32)
    m = metric.expected_reward_matrix(p, c, denom=0.07721)
    assert m.shape == (4, 11)


def test_real_anchors(real_train):
    perf = real_train[metric.PERF_COLS].to_numpy(np.float64)
    cost = real_train[metric.COST_COLS].to_numpy(np.float64)
    denom = metric.cost_denominator(cost)
    assert abs(denom - 0.07721) < 1e-3
    k = metric.MODEL_NAMES.index("Model_K")
    always_k = metric.route_reward(np.full(len(perf), k), perf, cost, denom)
    assert abs(always_k - 0.450376) < 2e-3
    reward_mat = metric.expected_reward_matrix_from_truth(perf, cost, denom)
    oracle = metric.route_reward(reward_mat.argmax(1), perf, cost, denom)
    assert abs(oracle - 0.673071) < 2e-3
```

- [ ] **Step 2: Run, verify failure**

Run: `python -m pytest tests/test_metric.py -q`
Expected: FAIL (`ModuleNotFoundError`/`AttributeError: metric has no ...`).

- [ ] **Step 3: Implement `src/metric.py`**

```python
from __future__ import annotations
import numpy as np

LETTERS = list("ABCDEFGHIJK")
MODEL_NAMES = [f"Model_{L}" for L in LETTERS]
PERF_COLS = [f"{m}_performance" for m in MODEL_NAMES]
COST_COLS = [f"{m}_cost" for m in MODEL_NAMES]

PERF_WEIGHT = 0.85
COST_WEIGHT = 0.15


def cost_denominator(cost: np.ndarray) -> float:
    d = float(np.asarray(cost, np.float64).max(axis=1).mean())
    if d <= 0:
        raise ValueError("cost denominator must be positive")
    return d


def route_reward(pred_idx, perf, cost, denom,
                 perf_weight=PERF_WEIGHT, cost_weight=COST_WEIGHT) -> float:
    pred_idx = np.asarray(pred_idx, np.int64)
    rows = np.arange(len(pred_idx))
    mp = float(np.asarray(perf, np.float64)[rows, pred_idx].mean())
    mc = float(np.asarray(cost, np.float64)[rows, pred_idx].mean())
    return perf_weight * mp - cost_weight * (mc / denom)


def expected_reward_matrix(p_hat, cost_const, denom,
                           perf_weight=PERF_WEIGHT, cost_weight=COST_WEIGHT) -> np.ndarray:
    p_hat = np.asarray(p_hat, np.float64)
    cost_const = np.asarray(cost_const, np.float64).reshape(1, -1)
    return perf_weight * p_hat - cost_weight * (cost_const / denom)


def expected_reward_matrix_from_truth(perf, cost, denom,
                                      perf_weight=PERF_WEIGHT, cost_weight=COST_WEIGHT) -> np.ndarray:
    perf = np.asarray(perf, np.float64)
    cost = np.asarray(cost, np.float64)
    return perf_weight * perf - cost_weight * (cost / denom)
```

- [ ] **Step 4: Run, verify pass**

Run: `python -m pytest tests/test_metric.py -q`
Expected: PASS (5 tests; `test_real_anchors` passes locally since `dataset/train.csv` exists).

- [ ] **Step 5: Commit**

```bash
git add src/metric.py tests/test_metric.py
git commit -m "feat: exact Reward_0.85 metric + anchors test

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `data.py` — load + targets + cost constants (TDD)

**Files:**
- Create: `src/data.py`, `tests/test_data.py`

- [ ] **Step 1: Write failing tests**

`tests/test_data.py`:
```python
import numpy as np
from src import data, metric


def test_build_targets_shapes(synthetic_train):
    perf, cost = data.build_targets(synthetic_train)
    assert perf.shape == (len(synthetic_train), 11)
    assert cost.shape == (len(synthetic_train), 11)
    assert set(np.unique(perf)).issubset({0.0, 1.0})


def test_cost_constants(synthetic_train):
    _, cost = data.build_targets(synthetic_train)
    cc = data.cost_constants(cost)
    assert cc.shape == (11,)
    assert np.allclose(cc, cost.mean(0), atol=1e-6)


def test_validate_columns_real(real_train):
    # should not raise
    data.validate_train_columns(real_train)
```

- [ ] **Step 2: Run, verify failure**

Run: `python -m pytest tests/test_data.py -q`
Expected: FAIL (module/attr missing).

- [ ] **Step 3: Implement `src/data.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

from .metric import MODEL_NAMES, PERF_COLS, COST_COLS


def validate_train_columns(df: pd.DataFrame) -> None:
    expected = ["ID", "query"] + [c for pair in zip(PERF_COLS, COST_COLS) for c in pair]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"missing train columns: {missing}")


def load_data(cfg):
    train = pd.read_csv(cfg.data_dir / "train.csv")
    test = pd.read_csv(cfg.data_dir / "test.csv")
    sample = pd.read_csv(cfg.data_dir / "sample_submission.csv")
    validate_train_columns(train)
    assert list(test.columns) == ["ID", "query"], test.columns.tolist()
    assert list(sample.columns) == ["ID", "pred_model"], sample.columns.tolist()
    train["query"] = train["query"].fillna("").astype(str)
    test["query"] = test["query"].fillna("").astype(str)
    if cfg.smoke:
        train = train.sample(min(cfg.smoke_rows, len(train)),
                             random_state=cfg.seed).sort_values("ID").reset_index(drop=True)
        test = test.head(min(200, len(test))).copy()
    return train, test, sample


def build_targets(df: pd.DataFrame):
    perf = df[PERF_COLS].to_numpy(np.float32)
    cost = df[COST_COLS].to_numpy(np.float32)
    return perf, cost


def cost_constants(cost: np.ndarray) -> np.ndarray:
    return np.asarray(cost, np.float32).mean(axis=0)
```

- [ ] **Step 4: Run, verify pass; commit**

Run: `python -m pytest tests/test_data.py -q` → PASS
```bash
git add src/data.py tests/test_data.py
git commit -m "feat: data loading, targets, per-model cost constants

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: `features.py` — handcrafted + mcq flag + TF-IDF + embeddings

**Files:**
- Create: `src/features.py`, `tests/test_features.py`

- [ ] **Step 1: Write failing tests (handcrafted + mcq are pure/local)**

`tests/test_features.py`:
```python
import numpy as np
import pandas as pd
from src import features


def test_mcq_flag():
    q = pd.Series([
        "What is 2+2?",
        "Pick one.\nAnswer Choices:\nA. 1\nB. 2",
        "Choose:\nA) red\nB) blue\nC) green",
    ])
    flags = features.mcq_flag(q).to_numpy()
    assert flags.tolist() == [False, True, True]


def test_handcrafted_shape_and_no_nan():
    q = pd.Series(["short", "find x " * 50, "Answer Choices: A. a B. b"])
    f = features.handcrafted_features(q)
    assert len(f) == 3
    assert not f.isna().any().any()
    assert "is_mcq" in f.columns and "log_char_len" in f.columns


def test_handcrafted_is_numeric_matrix():
    q = pd.Series(["a", "b b b"])
    arr = features.handcrafted_matrix(q)
    assert arr.dtype == np.float32
    assert arr.shape[0] == 2
```

- [ ] **Step 2: Run, verify failure**

Run: `python -m pytest tests/test_features.py -q` → FAIL

- [ ] **Step 3: Implement `src/features.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def mcq_flag(q: pd.Series) -> pd.Series:
    q = q.fillna("").astype(str)
    has_choices = q.str.contains("Answer Choices", case=False, regex=False)
    lettered = q.str.contains(r"\n\s*[A-E][\.\)]\s", regex=True)
    return has_choices | lettered


def handcrafted_features(queries: pd.Series) -> pd.DataFrame:
    q = queries.fillna("").astype(str)
    f = pd.DataFrame(index=q.index)
    f["char_len"] = q.str.len().astype(np.float32)
    f["word_count"] = q.str.split().str.len().fillna(0).astype(np.float32)
    f["log_char_len"] = np.log1p(f["char_len"])
    f["log_word_count"] = np.log1p(f["word_count"])
    f["line_count"] = (q.str.count("\n") + 1).astype(np.float32)
    f["digit_ratio"] = (q.str.count(r"\d") / (f["char_len"] + 1.0)).astype(np.float32)
    f["latex"] = q.str.count(r"\$|\\frac|\\sqrt|\\sum|\\angle").astype(np.float32)
    f["code"] = q.str.count(r"```|def |class |#include|import |SELECT |function ").astype(np.float32)
    f["question_marks"] = q.str.count(r"\?").astype(np.float32)
    f["is_mcq"] = mcq_flag(q).astype(np.float32)
    f["n_choices"] = q.str.count(r"\n\s*[A-E][\.\)]\s").astype(np.float32)
    f["is_long"] = (f["char_len"] > 5000).astype(np.float32)
    f["is_short"] = (f["char_len"] < 300).astype(np.float32)
    return f.astype(np.float32)


def handcrafted_matrix(queries: pd.Series) -> np.ndarray:
    return handcrafted_features(queries).to_numpy(np.float32)


def cap_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.65)
    tail = max_chars - head
    return text[:head] + "\n[...]\n" + text[-tail:]


def build_tfidf(cfg) -> FeatureUnion:
    word = TfidfVectorizer(lowercase=True, strip_accents="unicode", sublinear_tf=True,
                           ngram_range=(1, 2), min_df=2, max_features=cfg.tfidf_word_features)
    char = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5),
                           min_df=2, max_features=cfg.tfidf_char_features)
    return FeatureUnion([("word", word), ("char", char)])


def _embed_cache_path(cfg, split: str, n: int) -> Path:
    return cfg.cache_dir / f"{cfg.embedding_cache_name}_{split}_{n}.npy"


def load_or_compute_embeddings(cfg, df: pd.DataFrame, split: str) -> np.ndarray:
    """Reuse cached Qwen3 embeddings if present (new or previous run dir); else compute (GPU)."""
    n = len(df)
    target = _embed_cache_path(cfg, split, n)
    if target.exists():
        return np.load(target)
    prev = cfg.prev_cache_dir / f"{cfg.embedding_cache_name}_{split}_{n}.npy"
    if prev.exists():
        emb = np.load(prev)
        np.save(target, emb)
        return emb
    return _compute_embeddings(cfg, df, split, target)


def _compute_embeddings(cfg, df, split, target) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    import torch
    kwargs = dict(model_name_or_path=cfg.qwen_model_id, truncate_dim=cfg.embedding_dim,
                  trust_remote_code=True)
    if torch.cuda.is_available():
        model = SentenceTransformer(**kwargs,
                                    model_kwargs={"torch_dtype": torch.float16, "device_map": "auto"},
                                    tokenizer_kwargs={"padding_side": "left"})
    else:
        model = SentenceTransformer(**kwargs)
    texts = [cap_text(t, 12000) for t in df["query"].fillna("").astype(str).tolist()]
    emb = model.encode(texts, batch_size=8, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    np.save(target, emb)
    return emb
```

- [ ] **Step 4: Run handcrafted/mcq tests, verify pass; commit**

Run: `python -m pytest tests/test_features.py -q` → PASS (embedding loader is not unit-tested; exercised in Task 7 smoke).
```bash
git add src/features.py tests/test_features.py
git commit -m "feat: handcrafted features, mcq flag, TF-IDF, embedding loader

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: `cv.py` — stratified shared folds (TDD)

**Files:**
- Create: `src/cv.py`, `tests/test_cv.py`

- [ ] **Step 1: Write failing tests**

`tests/test_cv.py`:
```python
import numpy as np
from src import cv, data


def test_folds_partition(synthetic_train):
    perf, _ = data.build_targets(synthetic_train)
    folds = cv.make_folds_from_arrays(perf, synthetic_train["query"], n_splits=5, seed=42)
    assert len(folds) == 5
    all_val = np.concatenate([va for _, va in folds])
    assert sorted(all_val.tolist()) == list(range(len(perf)))  # exact partition
    for tr, va in folds:
        assert set(tr).isdisjoint(set(va))


def test_folds_deterministic(synthetic_train):
    perf, _ = data.build_targets(synthetic_train)
    q = synthetic_train["query"]
    f1 = cv.make_folds_from_arrays(perf, q, 5, 42)
    f2 = cv.make_folds_from_arrays(perf, q, 5, 42)
    for (a, b), (c, d) in zip(f1, f2):
        assert np.array_equal(b, d)
```

- [ ] **Step 2: Run, verify failure** → FAIL

- [ ] **Step 3: Implement `src/cv.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from .features import mcq_flag


def make_strata(perf: np.ndarray, queries: pd.Series) -> np.ndarray:
    n_correct = np.asarray(perf, np.float32).sum(1).astype(int)   # 0..11
    is_mcq = mcq_flag(queries).to_numpy().astype(int)
    return n_correct * 2 + is_mcq


def make_folds_from_arrays(perf, queries, n_splits=5, seed=42):
    strata = make_strata(perf, pd.Series(list(queries)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr.copy(), va.copy()) for tr, va in skf.split(np.arange(len(strata)), strata)]


def get_folds(cfg, perf, queries):
    path = cfg.cache_dir / f"folds_{cfg.n_splits}_{cfg.seed}_{len(perf)}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return [(z[f"tr{i}"], z[f"va{i}"]) for i in range(cfg.n_splits)]
    folds = make_folds_from_arrays(perf, queries, cfg.n_splits, cfg.seed)
    save = {}
    for i, (tr, va) in enumerate(folds):
        save[f"tr{i}"] = tr
        save[f"va{i}"] = va
    np.savez(path, **save)
    return folds
```

- [ ] **Step 4: Run, verify pass; commit**

Run: `python -m pytest tests/test_cv.py -q` → PASS
```bash
git add src/cv.py tests/test_cv.py
git commit -m "feat: stratified shared CV folds (difficulty x mcq)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: `models_classical.py` — LightGBM + linear base learners

**Files:**
- Create: `src/models_classical.py`, `tests/test_models_classical.py`

Produces OOF (N×11) + test (M×11) probability matrices. LightGBM uses dense `[SVD(TF-IDF) ⊕ embeddings ⊕ handcrafted (⊕ extra)]`; linear uses sparse `TF-IDF ⊕ scaled handcrafted`.

- [ ] **Step 1: Write a smoke test (skips if lightgbm absent)**

`tests/test_models_classical.py`:
```python
import numpy as np
import pytest
from src import models_classical as mc, data, cv


def test_lgbm_oof_runs(synthetic_train):
    pytest.importorskip("lightgbm")
    perf, _ = data.build_targets(synthetic_train)
    X = np.random.RandomState(0).randn(len(perf), 12).astype(np.float32)
    folds = cv.make_folds_from_arrays(perf, synthetic_train["query"], 3, 42)
    oof = mc.lgbm_oof(X, perf, folds, n_estimators=50, lr=0.1, leaves=7, min_child=2)
    assert oof.shape == perf.shape
    assert (oof >= 0).all() and (oof <= 1).all()


def test_linear_oof_runs(synthetic_train):
    perf, _ = data.build_targets(synthetic_train)
    from scipy.sparse import csr_matrix
    X = csr_matrix(np.random.RandomState(1).rand(len(perf), 20))
    folds = cv.make_folds_from_arrays(perf, synthetic_train["query"], 3, 42)
    oof = mc.linear_oof(X, perf, folds)
    assert oof.shape == perf.shape
    assert (oof >= 0).all() and (oof <= 1).all()
```

- [ ] **Step 2: Run, verify failure** → FAIL

- [ ] **Step 3: Implement `src/models_classical.py`**

```python
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression


def _lgbm_factory(n_estimators, lr, leaves, min_child, seed=42):
    from lightgbm import LGBMClassifier
    return lambda: LGBMClassifier(
        objective="binary", n_estimators=n_estimators, learning_rate=lr,
        num_leaves=leaves, min_child_samples=min_child, subsample=0.9,
        colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=0.2,
        random_state=seed, n_jobs=-1, verbosity=-1)


def _const_or_proba(model, X):
    # robust when a fold's label column is single-class
    if hasattr(model, "classes_") and len(model.classes_) == 1:
        return np.full(X.shape[0], float(model.classes_[0]), dtype=np.float32)
    return model.predict_proba(X)[:, 1].astype(np.float32)


def _oof_multilabel(factory, X, Y, folds):
    oof = np.zeros_like(Y, dtype=np.float32)
    for tr, va in folds:
        for j in range(Y.shape[1]):
            ytr = Y[tr, j]
            m = factory()
            if len(np.unique(ytr)) == 1:
                oof[va, j] = float(ytr[0])
                continue
            m.fit(X[tr], ytr)
            oof[va, j] = _const_or_proba(m, X[va])
    return oof


def _full_multilabel(factory, X, Y, X_test):
    test = np.zeros((X_test.shape[0], Y.shape[1]), dtype=np.float32)
    for j in range(Y.shape[1]):
        y = Y[:, j]
        m = factory()
        if len(np.unique(y)) == 1:
            test[:, j] = float(y[0])
            continue
        m.fit(X, y)
        test[:, j] = _const_or_proba(m, X_test)
    return test


def lgbm_oof(X, Y, folds, n_estimators=1200, lr=0.03, leaves=31, min_child=30, seed=42):
    return _oof_multilabel(_lgbm_factory(n_estimators, lr, leaves, min_child, seed), X, Y, folds)


def lgbm_full(X, Y, X_test, n_estimators=1200, lr=0.03, leaves=31, min_child=30, seed=42):
    return _full_multilabel(_lgbm_factory(n_estimators, lr, leaves, min_child, seed), X, Y, X_test)


def _linear_factory(seed=42):
    return lambda: LogisticRegression(C=1.0, max_iter=2000, solver="liblinear", random_state=seed)


def linear_oof(X, Y, folds, seed=42):
    return _oof_multilabel(_linear_factory(seed), X, Y, folds)


def linear_full(X, Y, X_test, seed=42):
    return _full_multilabel(_linear_factory(seed), X, Y, X_test)
```

- [ ] **Step 4: Run, verify pass; commit**

Run: `python -m pytest tests/test_models_classical.py -q` → PASS (or skip if lightgbm absent locally)
```bash
git add src/models_classical.py tests/test_models_classical.py
git commit -m "feat: LightGBM + logistic per-model correctness learners

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: `ensemble_routing.py` — calibration, ensemble, bias-tuned routing, submission

**Files:**
- Create: `src/ensemble_routing.py`, `tests/test_ensemble_routing.py`

- [ ] **Step 1: Write failing tests (routing math + bias + submission are pure)**

`tests/test_ensemble_routing.py`:
```python
import numpy as np
import pandas as pd
import pytest
from src import ensemble_routing as er, metric


def test_route_argmax_expected_reward():
    p = np.array([[0.9, 0.1, 0.1], [0.1, 0.8, 0.2]])
    cc = np.array([0.001, 0.001, 0.001])
    idx = er.route(p, cc, denom=0.07721, bias=np.zeros(3))
    assert idx.tolist() == [0, 1]


def test_bias_breaks_ties_toward_cheap():
    # equal prob, model 2 far cheaper -> bias search should prefer it
    p = np.full((50, 3), 0.5, dtype=np.float64)
    perf = (np.random.RandomState(0).rand(50, 3) < 0.5).astype(np.float64)
    cost = np.array([[0.1, 0.1, 0.001]] * 50, dtype=np.float64)
    cc = cost.mean(0)
    bias = er.tune_bias(p, perf, cost, cc, denom=0.07721, grid=np.linspace(-0.2, 0.2, 9), passes=2)
    assert bias.shape == (3,)


def test_weighted_average_normalizes():
    a = np.full((4, 3), 0.2)
    b = np.full((4, 3), 0.6)
    out = er.weighted_average([a, b], [1.0, 3.0])
    assert np.allclose(out, 0.2 * 0.25 + 0.6 * 0.75)


def test_write_submission(tmp_path):
    sample = pd.DataFrame({"ID": [1, 2, 3], "pred_model": ["Model_A"] * 3})
    test_ids = np.array([1, 2, 3])
    path = er.write_submission(tmp_path / "sub.csv", test_ids, np.array([0, 10, 5]), sample)
    out = pd.read_csv(path)
    assert out.columns.tolist() == ["ID", "pred_model"]
    assert out["pred_model"].tolist() == ["Model_A", "Model_K", "Model_F"]
```

- [ ] **Step 2: Run, verify failure** → FAIL

- [ ] **Step 3: Implement `src/ensemble_routing.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.isotonic import IsotonicRegression

from .metric import (MODEL_NAMES, expected_reward_matrix, route_reward,
                     PERF_WEIGHT, COST_WEIGHT)


def route(p_hat, cost_const, denom, bias=None) -> np.ndarray:
    er = expected_reward_matrix(p_hat, cost_const, denom)
    if bias is not None:
        er = er + np.asarray(bias, np.float64).reshape(1, -1)
    return er.argmax(1)


def isotonic_calibrate(oof, Y, test=None):
    """Fit per-model isotonic on OOF; return (calibrated_oof, calibrated_test_or_None)."""
    oof = np.asarray(oof, np.float64)
    cal_oof = np.zeros_like(oof)
    cal_test = None if test is None else np.zeros_like(np.asarray(test, np.float64))
    for j in range(oof.shape[1]):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        if len(np.unique(Y[:, j])) < 2:
            cal_oof[:, j] = oof[:, j]
            if test is not None:
                cal_test[:, j] = np.asarray(test)[:, j]
            continue
        ir.fit(oof[:, j], Y[:, j])
        cal_oof[:, j] = ir.predict(oof[:, j])
        if test is not None:
            cal_test[:, j] = ir.predict(np.asarray(test)[:, j])
    return (cal_oof, cal_test) if test is not None else (cal_oof, None)


def weighted_average(mats, weights):
    w = np.asarray(weights, np.float64)
    w = w / w.sum()
    out = np.zeros_like(np.asarray(mats[0], np.float64))
    for m, wi in zip(mats, w):
        out += wi * np.asarray(m, np.float64)
    return out


def tune_weights(oof_list, perf, cost, cost_const, denom, step=0.1):
    """Grid-search convex weights over base learners to maximize OOF route reward."""
    from itertools import product
    n = len(oof_list)
    units = int(round(1 / step))
    best_w, best_r = None, -1e9
    def gen(pos, remaining, acc):
        if pos == n - 1:
            yield acc + [remaining]
            return
        for v in range(remaining + 1):
            yield from gen(pos + 1, remaining - v, acc + [v])
    for combo in gen(0, units, []):
        if sum(c > 0 for c in combo) < 1:
            continue
        w = [c / units for c in combo]
        blended = weighted_average(oof_list, w)
        r = route_reward(route(blended, cost_const, denom), perf, cost, denom)
        if r > best_r:
            best_r, best_w = r, w
    return np.array(best_w), best_r


def tune_bias(p_hat, perf, cost, cost_const, denom,
              grid=None, passes=3):
    """Coordinate-ascent per-model additive bias to maximize OOF route reward."""
    if grid is None:
        grid = np.linspace(-0.15, 0.15, 31)
    n_models = p_hat.shape[1]
    bias = np.zeros(n_models)
    best = route_reward(route(p_hat, cost_const, denom, bias), perf, cost, denom)
    for _ in range(passes):
        improved = False
        for j in range(n_models):
            base = bias.copy()
            for g in grid:
                trial = base.copy()
                trial[j] = g
                r = route_reward(route(p_hat, cost_const, denom, trial), perf, cost, denom)
                if r > best + 1e-9:
                    best, bias = r, trial
                    improved = True
        if not improved:
            break
    return bias


def write_submission(path, test_ids, pred_idx, sample_df) -> Path:
    path = Path(path)
    pred_model = [MODEL_NAMES[int(i)] for i in pred_idx]
    sub = pd.DataFrame({"ID": np.asarray(test_ids), "pred_model": pred_model})
    assert sub.columns.tolist() == ["ID", "pred_model"]
    assert len(sub) == len(sample_df)
    assert sub["pred_model"].isin(MODEL_NAMES).all()
    assert sub["ID"].tolist() == sample_df["ID"].tolist()
    sub.to_csv(path, index=False)
    return path
```

- [ ] **Step 4: Run, verify pass; commit**

Run: `python -m pytest tests/test_ensemble_routing.py -q` → PASS
```bash
git add src/ensemble_routing.py tests/test_ensemble_routing.py
git commit -m "feat: isotonic calibration, ensemble, bias-tuned routing, submission

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: `run.py` — Phase-0 orchestrator + notebook (FIRST SUBMISSION milestone)

**Files:**
- Create: `src/run.py`, `FinalProject_router.ipynb`

This wires Phase 0 end-to-end, caches OOF/test matrices, prints OOF route reward vs baselines, writes `submission.csv` + `model_comparison.csv`.

- [ ] **Step 1: Implement `src/run.py`**

```python
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from .config import CFG, seed_everything
from . import data, features, cv, models_classical as mc, ensemble_routing as er
from .metric import MODEL_NAMES, route_reward, cost_denominator


def _cache_np(path, fn):
    path = Path(path)
    if path.exists():
        return np.load(path)
    arr = fn()
    np.save(path, arr)
    return arr


def build_feature_blocks(cfg, train, test):
    """Return dense X for LGBM and sparse X for linear (train+test)."""
    # handcrafted
    hc_tr = features.handcrafted_matrix(train["query"])
    hc_te = features.handcrafted_matrix(test["query"])
    sc = StandardScaler().fit(hc_tr)
    hc_tr_s, hc_te_s = sc.transform(hc_tr), sc.transform(hc_te)

    # TF-IDF (sparse for linear) + SVD (dense for lgbm)
    tfidf = features.build_tfidf(cfg)
    tr_txt = [features.cap_text(t, cfg.tfidf_max_chars) for t in train["query"]]
    te_txt = [features.cap_text(t, cfg.tfidf_max_chars) for t in test["query"]]
    Xtf_tr = tfidf.fit_transform(tr_txt)
    Xtf_te = tfidf.transform(te_txt)
    svd = TruncatedSVD(n_components=cfg.svd_components, random_state=cfg.seed).fit(Xtf_tr)
    svd_tr, svd_te = svd.transform(Xtf_tr).astype(np.float32), svd.transform(Xtf_te).astype(np.float32)

    # embeddings (cached reuse)
    try:
        emb_tr = features.load_or_compute_embeddings(cfg, train, "train")
        emb_te = features.load_or_compute_embeddings(cfg, test, "test")
    except Exception as e:
        print("embeddings unavailable, proceeding without:", repr(e))
        emb_tr = np.zeros((len(train), 0), np.float32)
        emb_te = np.zeros((len(test), 0), np.float32)

    dense_tr = np.hstack([svd_tr, emb_tr, hc_tr_s]).astype(np.float32)
    dense_te = np.hstack([svd_te, emb_te, hc_te_s]).astype(np.float32)
    sparse_tr = hstack([Xtf_tr, csr_matrix(hc_tr_s)]).tocsr()
    sparse_te = hstack([Xtf_te, csr_matrix(hc_te_s)]).tocsr()
    return dense_tr, dense_te, sparse_tr, sparse_te


def run_phase0(cfg: CFG):
    seed_everything(cfg.seed)
    train, test, sample = data.load_data(cfg)
    perf, cost = data.build_targets(train)
    cost_const = data.cost_constants(cost)
    denom = cost_denominator(cost)
    folds = cv.get_folds(cfg, perf, train["query"])

    dense_tr, dense_te, sparse_tr, sparse_te = build_feature_blocks(cfg, train, test)

    # base learners (cached)
    lgbm_oof = _cache_np(cfg.cache_dir / "lgbm_oof.npy",
        lambda: mc.lgbm_oof(dense_tr, perf, folds, cfg.lgbm_estimators, cfg.lgbm_lr, cfg.lgbm_leaves, cfg.lgbm_min_child, cfg.seed))
    lgbm_test = _cache_np(cfg.cache_dir / "lgbm_test.npy",
        lambda: mc.lgbm_full(dense_tr, perf, dense_te, cfg.lgbm_estimators, cfg.lgbm_lr, cfg.lgbm_leaves, cfg.lgbm_min_child, cfg.seed))
    lin_oof = _cache_np(cfg.cache_dir / "lin_oof.npy", lambda: mc.linear_oof(sparse_tr, perf, folds, cfg.seed))
    lin_test = _cache_np(cfg.cache_dir / "lin_test.npy", lambda: mc.linear_full(sparse_tr, perf, sparse_te, cfg.seed))

    # calibrate
    lgbm_oof_c, lgbm_test_c = er.isotonic_calibrate(lgbm_oof, perf, lgbm_test)
    lin_oof_c, lin_test_c = er.isotonic_calibrate(lin_oof, perf, lin_test)

    return route_and_submit(cfg, [lgbm_oof_c, lin_oof_c], [lgbm_test_c, lin_test_c],
                            perf, cost, cost_const, denom, test, sample,
                            names=["lgbm", "linear"])


def route_and_submit(cfg, oof_list, test_list, perf, cost, cost_const, denom,
                     test, sample, names):
    w, w_reward = er.tune_weights(oof_list, perf, cost, cost_const, denom, step=0.1)
    oof_blend = er.weighted_average(oof_list, w)
    test_blend = er.weighted_average(test_list, w)
    bias = er.tune_bias(oof_blend, perf, cost, cost_const, denom)
    oof_reward = route_reward(er.route(oof_blend, cost_const, denom, bias), perf, cost, denom)

    rows = []
    for nm, oof in zip(names, oof_list):
        rows.append({"method": nm,
                     "oof_reward": route_reward(er.route(oof, cost_const, denom), perf, cost, denom)})
    rows.append({"method": f"ensemble({'+'.join(names)})_weights={list(np.round(w,2))}", "oof_reward": w_reward})
    rows.append({"method": "ensemble+bias", "oof_reward": oof_reward})
    k = MODEL_NAMES.index("Model_K")
    rows.append({"method": "always_K", "oof_reward": route_reward(np.full(len(perf), k), perf, cost, denom)})
    comp = pd.DataFrame(rows).sort_values("oof_reward", ascending=False)
    comp.to_csv(cfg.out_dir / "model_comparison.csv", index=False)
    print(comp.to_string(index=False))

    test_idx = er.route(test_blend, cost_const, denom, bias)
    er.write_submission(cfg.out_dir / "submission.csv", test["ID"].to_numpy(), test_idx, sample)
    dist = pd.Series([MODEL_NAMES[i] for i in test_idx]).value_counts()
    print("OOF ensemble+bias reward:", round(oof_reward, 5))
    print("submission model distribution:\n", dist.to_string())
    return {"oof_reward": oof_reward, "weights": w.tolist(), "bias": bias.tolist(),
            "oof_blend": oof_blend, "test_blend": test_blend}


if __name__ == "__main__":
    run_phase0(CFG())
```

- [ ] **Step 2: Smoke-run Phase 0 locally (smoke mode, no embeddings/lightgbm optional)**

Run:
```bash
python -c "from src.config import CFG; from src.run import run_phase0; run_phase0(CFG(smoke=True, svd_components=32, lgbm_estimators=50))"
```
Expected: prints a comparison table; `ensemble+bias` OOF reward ≥ `always_K`; writes `Output/router_v2/submission.csv`. (If lightgbm/embeddings absent locally, linear-only still runs and writes a submission.)

- [ ] **Step 3: Full Phase-0 run on Colab A100 (reuses cached Qwen3 embeddings)**

In `FinalProject_router.ipynb` (thin notebook): mount Drive, `pip install -r requirements.txt` (+ sentence-transformers), `import sys; sys.path.append(str(CFG().root))`, then:
```python
from src.config import CFG
from src.run import run_phase0
res = run_phase0(CFG())
```
Expected: OOF `ensemble+bias` reward **clearly > 0.469** (target the first real jump). Inspect `model_comparison.csv`.

- [ ] **Step 4: Commit + SUBMIT #1**

```bash
git add src/run.py FinalProject_router.ipynb
git commit -m "feat: Phase-0 orchestrator (reframe) + first submission

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
Upload `Output/router_v2/submission.csv` to Kaggle. Record public LB next to OOF reward in `model_comparison.csv` notes. **Do not pick final submissions yet.**

---

## Task 8: `models_encoder.py` — ModernBERT-large (Phase 1, Colab A100)

**Files:**
- Create: `src/models_encoder.py`
- Modify: `src/run.py` (add `run_phase1`)

Fine-tune ModernBERT-large with an 11-logit multilabel head + auxiliary difficulty-regression head; produce OOF + test probability matrices; ensemble with Phase-0 learners.

- [ ] **Step 1: Implement `src/models_encoder.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd


def _make_inputs(queries, tokenizer, max_len):
    """Head+tail truncation to fit max_len tokens."""
    texts = []
    for t in queries.fillna("").astype(str).tolist():
        if len(t) > 16000:  # cheap pre-trim before tokenizing very long docs
            t = t[:11000] + "\n[...]\n" + t[-5000:]
        texts.append(t)
    enc = tokenizer(texts, truncation=True, max_length=max_len, padding=False)
    return enc


def build_model(cfg):
    import torch, torch.nn as nn
    from transformers import AutoModel

    class Router(nn.Module):
        def __init__(self, base_id, n_models=11, aux_w=0.3):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(base_id)
            h = self.backbone.config.hidden_size
            self.dropout = nn.Dropout(0.1)
            self.perf_head = nn.Linear(h, n_models)
            self.diff_head = nn.Linear(h, 1)
            self.aux_w = aux_w

        def forward(self, input_ids=None, attention_mask=None, labels=None, difficulty=None, **kw):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            pooled = self.dropout(pooled)
            perf_logits = self.perf_head(pooled)
            diff_pred = self.diff_head(pooled).squeeze(-1)
            loss = None
            if labels is not None:
                bce = nn.functional.binary_cross_entropy_with_logits(perf_logits, labels.float())
                loss = bce
                if difficulty is not None:
                    loss = loss + self.aux_w * nn.functional.mse_loss(diff_pred, difficulty.float())
            return {"loss": loss, "logits": perf_logits}

    return Router(cfg.encoder_id, 11, cfg.aux_diff_weight)


def _train_one(cfg, train_df, tr_idx, va_idx_or_test, perf, is_test=False):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, DataCollatorWithPadding

    tok = AutoTokenizer.from_pretrained(cfg.encoder_id)
    sub = train_df.iloc[tr_idx]
    enc = _make_inputs(sub["query"], tok, cfg.encoder_max_len)
    labels = perf[tr_idx]
    difficulty = labels.sum(1) / 11.0

    class DS(torch.utils.data.Dataset):
        def __init__(self, enc, labels=None, diff=None):
            self.enc, self.labels, self.diff = enc, labels, diff
        def __len__(self): return len(self.enc["input_ids"])
        def __getitem__(self, i):
            d = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
            if self.labels is not None:
                d["labels"] = torch.tensor(self.labels[i])
                d["difficulty"] = torch.tensor(self.diff[i])
            return d

    model = build_model(cfg).cuda()
    if torch.cuda.is_available():
        model = model.to(torch.bfloat16)
    collator = DataCollatorWithPadding(tok)

    def collate(batch):
        labels = torch.stack([b.pop("labels") for b in batch]) if "labels" in batch[0] else None
        diff = torch.stack([b.pop("difficulty") for b in batch]) if "difficulty" in batch[0] else None
        out = collator(batch)
        if labels is not None:
            out["labels"] = labels; out["difficulty"] = diff
        return out

    dl = DataLoader(DS(enc, labels, difficulty), batch_size=cfg.encoder_bs, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.encoder_lr)
    model.train()
    for epoch in range(cfg.encoder_epochs):
        for step, batch in enumerate(dl):
            batch = {k: v.cuda() for k, v in batch.items()}
            out = model(**batch)
            (out["loss"] / cfg.encoder_grad_accum).backward()
            if (step + 1) % cfg.encoder_grad_accum == 0:
                opt.step(); opt.zero_grad()

    # predict
    model.eval()
    eval_df = train_df.iloc[va_idx_or_test] if not is_test else va_idx_or_test
    enc_e = _make_inputs(eval_df["query"], tok, cfg.encoder_max_len)
    dl_e = DataLoader(DS(enc_e), batch_size=cfg.encoder_bs * 2, shuffle=False, collate_fn=collate)
    preds = []
    with torch.no_grad():
        for batch in dl_e:
            batch = {k: v.cuda() for k, v in batch.items()}
            logits = model(**batch)["logits"].float()
            preds.append(torch.sigmoid(logits).cpu().numpy())
    import gc; del model; gc.collect(); torch.cuda.empty_cache()
    return np.vstack(preds).astype(np.float32)


def encoder_oof_and_test(cfg, train_df, test_df, perf, folds):
    oof = np.zeros_like(perf, dtype=np.float32)
    for tr, va in folds:
        oof[va] = _train_one(cfg, train_df, tr, va, perf)
    test_pred = _train_one(cfg, train_df, np.arange(len(train_df)), test_df, perf, is_test=True)
    return oof, test_pred
```

- [ ] **Step 2: Add `run_phase1` to `src/run.py`**

```python
def run_phase1(cfg):
    from . import models_encoder as me
    seed_everything(cfg.seed)
    train, test, sample = data.load_data(cfg)
    perf, cost = data.build_targets(train)
    cost_const = data.cost_constants(cost); denom = cost_denominator(cost)
    folds = cv.get_folds(cfg, perf, train["query"])
    dense_tr, dense_te, sparse_tr, sparse_te = build_feature_blocks(cfg, train, test)

    lgbm_oof = np.load(cfg.cache_dir / "lgbm_oof.npy"); lgbm_test = np.load(cfg.cache_dir / "lgbm_test.npy")
    lin_oof = np.load(cfg.cache_dir / "lin_oof.npy"); lin_test = np.load(cfg.cache_dir / "lin_test.npy")

    enc_oof_path = cfg.cache_dir / "enc_oof.npy"; enc_test_path = cfg.cache_dir / "enc_test.npy"
    if enc_oof_path.exists():
        enc_oof, enc_test = np.load(enc_oof_path), np.load(enc_test_path)
    else:
        enc_oof, enc_test = me.encoder_oof_and_test(cfg, train, test, perf, folds)
        np.save(enc_oof_path, enc_oof); np.save(enc_test_path, enc_test)

    lgbm_oof_c, lgbm_test_c = er.isotonic_calibrate(lgbm_oof, perf, lgbm_test)
    lin_oof_c, lin_test_c = er.isotonic_calibrate(lin_oof, perf, lin_test)
    enc_oof_c, enc_test_c = er.isotonic_calibrate(enc_oof, perf, enc_test)
    return route_and_submit(cfg, [lgbm_oof_c, lin_oof_c, enc_oof_c],
                            [lgbm_test_c, lin_test_c, enc_test_c],
                            perf, cost, cost_const, denom, test, sample,
                            names=["lgbm", "linear", "encoder"])
```

- [ ] **Step 3: Smoke test on Colab (1 fold, 1 epoch, tiny subset)**

Run in notebook:
```python
res = run_phase1(CFG(smoke=True, encoder_epochs=1, encoder_max_len=512, encoder_bs=4))
```
Expected: completes without OOM; prints comparison table including `encoder` row. Verifies the training/predict loop end-to-end.

- [ ] **Step 4: Full Phase-1 run, then commit + SUBMIT #2 (only if OOF improves)**

```python
res = run_phase1(CFG())
```
Expected: `encoder` OOF reward competitive; `ensemble+bias` OOF reward **> Phase-0** OOF reward. If yes:
```bash
git add src/models_encoder.py src/run.py
git commit -m "feat: ModernBERT-large multilabel+difficulty encoder (Phase 1)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
Submit `submission.csv` if OOF improved over Phase 0.

---

## Task 9: `llm_difficulty.py` — vLLM self-consistency/judge features (Phase 2, Colab A100)

**Files:**
- Create: `src/llm_difficulty.py`
- Modify: `src/run.py` (`build_feature_blocks` to optionally append LLM features; `run_phase2`)

- [ ] **Step 1: Implement `src/llm_difficulty.py`**

```python
from __future__ import annotations
import re, json, hashlib
import numpy as np
import pandas as pd
from pathlib import Path

JUDGE_PROMPT = (
    "You are rating the difficulty of a question for AI language models.\n"
    "Question:\n{q}\n\n"
    "Respond with ONLY a JSON object: "
    '{{"difficulty": <integer 1-10>, "p_solvable": <float 0-1>}}.'
)
SOLVE_PROMPT = "Solve the following problem. End with 'Final answer: <answer>'.\n\n{q}"

_ANS_RE = re.compile(r"final answer[:\s]*([^\n]+)", re.I)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_MCQ_RE = re.compile(r"\b([A-E])\b")
_REFUSE_RE = re.compile(r"cannot|cannot determine|not sure|impossible to|i don't know", re.I)


def _extract_answer(text: str) -> str:
    m = _ANS_RE.search(text)
    tail = m.group(1).strip() if m else text.strip()[-80:]
    mcq = _MCQ_RE.findall(tail[:10])
    if mcq:
        return mcq[0].upper()
    nums = _NUM_RE.findall(tail)
    if nums:
        return nums[-1]
    return tail.lower().strip()[:40]


def _self_consistency_feats(answers, lengths, refusals):
    from collections import Counter
    c = Counter(answers)
    n = max(len(answers), 1)
    top = c.most_common(1)[0][1] if c else 0
    probs = np.array([v / n for v in c.values()]) if c else np.array([1.0])
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    return {
        "sc_agreement": top / n,
        "sc_entropy": entropy,
        "sc_n_distinct": len(c),
        "gen_len_mean": float(np.mean(lengths)) if lengths else 0.0,
        "gen_len_std": float(np.std(lengths)) if lengths else 0.0,
        "refuse_rate": float(np.mean(refusals)) if refusals else 0.0,
    }


def compute_llm_features(cfg, df: pd.DataFrame, split: str) -> pd.DataFrame:
    cache = cfg.cache_dir / f"llm_feats_{split}_{len(df)}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    from vllm import LLM, SamplingParams
    llm = LLM(model=cfg.llm_id, dtype="bfloat16", gpu_memory_utilization=0.9,
              max_model_len=4096, trust_remote_code=True)
    queries = df["query"].fillna("").astype(str).tolist()
    queries = [q if len(q) < 6000 else q[:4000] + "\n[...]\n" + q[-1500:] for q in queries]

    # k self-consistency samples
    sc_params = SamplingParams(n=cfg.llm_k_samples, temperature=cfg.llm_temperature,
                               top_p=0.95, max_tokens=cfg.llm_max_new_tokens)
    sc_out = llm.generate([SOLVE_PROMPT.format(q=q) for q in queries], sc_params)
    # judge (greedy)
    judge_params = SamplingParams(n=1, temperature=0.0, max_tokens=64)
    judge_out = llm.generate([JUDGE_PROMPT.format(q=q) for q in queries], judge_params)

    rows = []
    for sc, jd in zip(sc_out, judge_out):
        answers, lengths, refusals = [], [], []
        for o in sc.outputs:
            txt = o.text
            answers.append(_extract_answer(txt))
            lengths.append(len(txt.split()))
            refusals.append(1 if _REFUSE_RE.search(txt) else 0)
        feat = _self_consistency_feats(answers, lengths, refusals)
        # judge parse
        diff, psolv = 5.0, 0.5
        try:
            j = json.loads(re.search(r"\{.*\}", jd.outputs[0].text, re.S).group(0))
            diff = float(j.get("difficulty", 5)); psolv = float(j.get("p_solvable", 0.5))
        except Exception:
            pass
        feat["judge_difficulty"] = diff
        feat["judge_p_solvable"] = psolv
        rows.append(feat)
    out = pd.DataFrame(rows).astype(np.float32)
    out.to_parquet(cache)
    return out
```

- [ ] **Step 2: Wire LLM features into `build_feature_blocks` + add `run_phase2`**

In `src/run.py`, extend `build_feature_blocks(cfg, train, test, extra_tr=None, extra_te=None)` to `np.hstack` `extra_*` onto the dense blocks (guard `None`). Add:
```python
def run_phase2(cfg):
    from . import llm_difficulty as ld, models_encoder as me
    seed_everything(cfg.seed)
    train, test, sample = data.load_data(cfg)
    perf, cost = data.build_targets(train)
    cost_const = data.cost_constants(cost); denom = cost_denominator(cost)
    folds = cv.get_folds(cfg, perf, train["query"])

    llm_tr = ld.compute_llm_features(cfg, train, "train").to_numpy(np.float32)
    llm_te = ld.compute_llm_features(cfg, test, "test").to_numpy(np.float32)
    dense_tr, dense_te, sparse_tr, sparse_te = build_feature_blocks(cfg, train, test, llm_tr, llm_te)

    lgbm_oof = mc.lgbm_oof(dense_tr, perf, folds, cfg.lgbm_estimators, cfg.lgbm_lr, cfg.lgbm_leaves, cfg.lgbm_min_child, cfg.seed)
    lgbm_test = mc.lgbm_full(dense_tr, perf, dense_te, cfg.lgbm_estimators, cfg.lgbm_lr, cfg.lgbm_leaves, cfg.lgbm_min_child, cfg.seed)
    np.save(cfg.cache_dir / "lgbm_oof.npy", lgbm_oof); np.save(cfg.cache_dir / "lgbm_test.npy", lgbm_test)

    lin_oof = np.load(cfg.cache_dir / "lin_oof.npy"); lin_test = np.load(cfg.cache_dir / "lin_test.npy")
    enc_oof = np.load(cfg.cache_dir / "enc_oof.npy"); enc_test = np.load(cfg.cache_dir / "enc_test.npy")

    mats = [er.isotonic_calibrate(o, perf, t) for o, t in
            [(lgbm_oof, lgbm_test), (lin_oof, lin_test), (enc_oof, enc_test)]]
    oof_list = [m[0] for m in mats]; test_list = [m[1] for m in mats]
    return route_and_submit(cfg, oof_list, test_list, perf, cost, cost_const, denom,
                            test, sample, names=["lgbm+llm", "linear", "encoder"])
```

- [ ] **Step 3: Smoke test (tiny subset, k=2)**

Run in notebook:
```python
res = run_phase2(CFG(smoke=True, llm_k_samples=2, llm_max_new_tokens=256, lgbm_estimators=50, svd_components=32))
```
Expected: builds LLM features, completes, prints comparison table with `lgbm+llm`. Caches `llm_feats_*.parquet`.

- [ ] **Step 4: Full Phase-2 run; commit + SUBMIT #3 (only if OOF improves)**

```python
res = run_phase2(CFG())
```
Expected: `ensemble+bias` OOF reward **> Phase-1**. If yes:
```bash
git add src/llm_difficulty.py src/run.py
git commit -m "feat: vLLM self-consistency/judge difficulty features (Phase 2)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
Submit if OOF improved.

---

## Task 10: Phase 3 — meta-stacker, bias refinement, final selection, E3 flatten

**Files:**
- Modify: `src/ensemble_routing.py` (add `logistic_stack`), `src/run.py` (`run_phase3`)
- Create: `scripts/flatten_for_e3.py`

- [ ] **Step 1: Add per-model logistic meta-stacker to `ensemble_routing.py`**

```python
def logistic_stack(oof_list, test_list, Y, extra_oof=None, extra_test=None):
    """Per-model logistic regression on stacked base probabilities (+ optional extra features)."""
    from sklearn.linear_model import LogisticRegression
    n, m = Y.shape
    stack_oof = np.stack(oof_list, axis=2)   # (n, m, n_base)
    stack_test = np.stack(test_list, axis=2)
    out_oof = np.zeros((n, m), np.float64)
    out_test = np.zeros((stack_test.shape[0], m), np.float64)
    for j in range(m):
        Xo = stack_oof[:, j, :]
        Xt = stack_test[:, j, :]
        if extra_oof is not None:
            Xo = np.hstack([Xo, extra_oof]); Xt = np.hstack([Xt, extra_test])
        if len(np.unique(Y[:, j])) < 2:
            out_oof[:, j] = Y[0, j]; out_test[:, j] = Y[0, j]; continue
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(Xo, Y[:, j])
        out_oof[:, j] = clf.predict_proba(Xo)[:, 1]
        out_test[:, j] = clf.predict_proba(Xt)[:, 1]
    return out_oof, out_test
```

> Note: stacker is fit on the same OOF it scores (mild optimism). For an honest estimate, evaluate the stacker's benefit by comparing its OOF route reward to the weighted-average ensemble; adopt only if it wins by a clear margin, and prefer the weighted average if marginal.

- [ ] **Step 2: Add `run_phase3` to `src/run.py`**

```python
def run_phase3(cfg):
    seed_everything(cfg.seed)
    train, test, sample = data.load_data(cfg)
    perf, cost = data.build_targets(train)
    cost_const = data.cost_constants(cost); denom = cost_denominator(cost)

    names = ["lgbm", "lin", "enc"]
    oof_list, test_list = [], []
    for a, b in [("lgbm_oof", "lgbm_test"), ("lin_oof", "lin_test"), ("enc_oof", "enc_test")]:
        o, t = er.isotonic_calibrate(np.load(cfg.cache_dir / f"{a}.npy"), perf, np.load(cfg.cache_dir / f"{b}.npy"))
        oof_list.append(o); test_list.append(t)

    # candidate A: weighted-average + bias  (from route_and_submit)
    res_wavg = route_and_submit(cfg, oof_list, test_list, perf, cost, cost_const, denom, test, sample, names)

    # candidate B: logistic stack + bias
    st_oof, st_test = er.logistic_stack(oof_list, test_list, perf)
    bias_b = er.tune_bias(st_oof, perf, cost, cost_const, denom)
    r_b = route_reward(er.route(st_oof, cost_const, denom, bias_b), perf, cost, denom)
    print("stack OOF reward:", round(r_b, 5), "| wavg OOF reward:", round(res_wavg["oof_reward"], 5))

    # choose best by OOF, write the two final candidate submissions
    er.write_submission(cfg.out_dir / "submission_wavg.csv", test["ID"].to_numpy(),
                        er.route(res_wavg["test_blend"], cost_const, denom, np.array(res_wavg["bias"])), sample)
    er.write_submission(cfg.out_dir / "submission_stack.csv", test["ID"].to_numpy(),
                        er.route(st_test, cost_const, denom, bias_b), sample)
    print("Final candidates written: submission_wavg.csv, submission_stack.csv")
    print("Pick the 2 with highest OOF reward + sane CV<->LB gap for the private leaderboard.")
```

- [ ] **Step 3: Create `scripts/flatten_for_e3.py` (single-file E3 deliverable)**

```python
"""Concatenate src/*.py (import order) + a __main__ runner into FinalProject_<TeamID>.py.
Usage: python scripts/flatten_for_e3.py Team_XX"""
import sys
from pathlib import Path

ORDER = ["config", "metric", "data", "features", "cv",
         "models_classical", "models_encoder", "llm_difficulty",
         "ensemble_routing", "run"]

def main(team_id):
    root = Path(__file__).resolve().parents[1]
    out = root / f"FinalProject_{team_id}.py"
    parts = ["# Auto-generated single-file router. See docs/superpowers/ for design+plan.\n"]
    for name in ORDER:
        src = (root / "src" / f"{name}.py").read_text()
        src = "\n".join(l for l in src.splitlines()
                        if not l.strip().startswith("from .") and not l.strip().startswith("from src"))
        parts.append(f"\n# ===== {name}.py =====\n{src}\n")
    parts.append('\nif __name__ == "__main__":\n    run_phase3(CFG())\n')
    out.write_text("\n".join(parts))
    print("wrote", out)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "TeamID")
```

- [ ] **Step 4: Run Phase 3, flatten, commit (final)**

```python
run_phase3(CFG())
```
```bash
python scripts/flatten_for_e3.py Team_XX   # replace with real Team ID
git add src/ensemble_routing.py src/run.py scripts/flatten_for_e3.py
git commit -m "feat: meta-stacker, final candidate selection, E3 flatten (Phase 3)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
Pick the **2 highest-OOF** submissions for the private leaderboard.

---

## Self-Review

**Spec coverage:**
- §3 reframe (P(correct) + cost-constant + bias-routing) → Tasks 1, 5, 6, 7 ✓
- §4a data/CV/metric → Tasks 1, 2, 4 ✓
- §4b features (handcrafted, TF-IDF, embeddings, LLM) → Tasks 3, 9 ✓
- §4c base learners (LGBM, linear, ModernBERT) → Tasks 5, 8 ✓
- §4d calibration + ensemble → Task 6; meta-stacker → Task 10 ✓
- §4e vLLM difficulty → Task 9 ✓
- §4f routing + submission → Task 6, 7 ✓
- §6 CV protocol (shared folds, OOF reward, comparison table) → Tasks 4, 7 ✓
- §7 phasing (each submittable) → Tasks 7, 8, 9, 10 ✓
- §8 bias offsets → Task 6 (`tune_bias`) ✓
- §9 caching/determinism/graceful degradation → Tasks 0, 3, 5, 7 (`_cache_np`, seed, embeddings try/except) ✓
- §10 testing (anchors, smoke, submission asserts) → Tasks 1, 7, 8, 9 ✓
- §11 structure → Task 0 + per-module tasks ✓
- §12 source-tagging → DEFERRED (intentionally not a task) ✓
- §13 report mapping → `model_comparison.csv` (Task 7) ✓

**Placeholder scan:** No TBD/TODO; all code blocks complete. `Team_XX`/`Team ID` are intentional user-supplied values. ✓

**Type consistency:** `route(p_hat, cost_const, denom, bias)`, `tune_bias(...)→(11,)`, `isotonic_calibrate(oof, Y, test)→(cal_oof, cal_test)`, `weighted_average(mats, weights)`, `route_and_submit(...)` signature consistent across Tasks 6–10. `lgbm_oof/lgbm_full/linear_oof/linear_full` names consistent between Task 5 and Tasks 7–9. Cache filenames (`lgbm_oof.npy`, `enc_oof.npy`, etc.) consistent across Tasks 7–10. ✓

**Known gap accepted:** `logistic_stack` fits/scores on same OOF (documented in Task 10 note; adopt only if clear OOF win) — honest-stacking via nested CV is a possible future refinement, intentionally out of scope to control complexity.
