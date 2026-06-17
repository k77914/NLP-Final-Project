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
