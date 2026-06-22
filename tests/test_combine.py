import numpy as np
import pandas as pd
from src import combine as cb


def test_route_margin_k_extremes():
    util = np.array([[0.5, 0.1, 0.1], [0.1, 0.6, 0.1]])
    # huge margin => low confidence everywhere => all fall back to k_idx
    assert (cb.route_margin_k(util, margin=10.0, k_idx=0) == 0).all()
    # zero margin => plain argmax
    assert (cb.route_margin_k(util, margin=0.0, k_idx=0) == cb.route_argmax(util)).all()


def test_blend_normalizes():
    a = np.full((4, 3), 0.2)
    b = np.full((4, 3), 0.6)
    out = cb.blend([a, b], [1.0, 3.0])
    assert np.allclose(out, 0.2 * 0.25 + 0.6 * 0.75)


def test_select_weights_margin_structure():
    rng = np.random.RandomState(0)
    util = rng.rand(80, 3)
    perf = (rng.rand(80, 3) < 0.5).astype(np.float64)
    cost = np.tile([0.001, 0.05, 0.05], (80, 1))
    r, w, mg = cb.select_weights_margin([util], perf, cost, denom=0.0772,
                                        margins=np.linspace(0, 0.03, 7), k_idx=0)
    assert len(w) == 1 and abs(sum(w) - 1.0) < 1e-9
    assert isinstance(mg, float) and np.isfinite(r)


def test_crossfit_cv_partition():
    rng = np.random.RandomState(1)
    n = 100
    util = rng.rand(n, 3)
    perf = (rng.rand(n, 3) < 0.5).astype(np.float64)
    cost = np.tile([0.001, 0.05, 0.05], (n, 1))
    folds = [(np.setdiff1d(np.arange(n), va), va) for va in np.array_split(np.arange(n), 5)]
    reward, rows = cb.crossfit_cv([util], perf, cost, 0.0772, folds, np.linspace(0, 0.03, 7), k_idx=0)
    assert len(rows) == 5
    assert np.isfinite(reward)
    assert all({"fold", "weights", "margin", "reward"} <= set(r) for r in rows)


def test_make_query_segments_cheap_rules():
    df = pd.DataFrame({
        "query": [
            "Write a Python function:\n```python\nreturn x\n```",
            "Answer Choices:\nA. one\nB. two",
            "Solve the equation x^2 + 3x + 2 = 0",
            "word " * 1200,
            "capital?",
            "Explain the tradeoffs of distributed databases.",
        ]
    })
    llm = pd.DataFrame({
        "judge_difficulty": [5, 5, 5, 5, 5, 9],
        "judge_p_solvable": [0.5, 0.5, 0.5, 0.5, 0.5, 0.2],
        "sc_agreement": [0.5, 0.5, 0.5, 0.5, 0.5, 0.2],
    })
    labels = cb.make_query_segments(df, llm).tolist()
    assert labels == ["code", "mcq", "math", "long", "short", "hard_general"]


def test_crossfit_segmented_cv_routes_all_rows():
    rng = np.random.RandomState(2)
    n = 90
    util_a = rng.rand(n, 3)
    util_b = rng.rand(n, 3)
    perf = (rng.rand(n, 3) < 0.5).astype(np.float64)
    cost = np.tile([0.001, 0.05, 0.05], (n, 1))
    folds = [(np.setdiff1d(np.arange(n), va), va) for va in np.array_split(np.arange(n), 5)]
    segments = np.array(["math"] * 30 + ["code"] * 30 + ["general"] * 30)
    reward, rows = cb.crossfit_segmented_cv(
        [util_a, util_b], perf, cost, 0.0772, folds, segments,
        np.array([0.0, 0.03]), min_segment_rows=5, k_idx=0,
    )
    assert np.isfinite(reward)
    assert sum(r["n_val"] for r in rows) == n
    assert any(r["source"] == "segment" for r in rows)
    assert all({"fold", "segment", "weights", "margin", "source"} <= set(r) for r in rows)


def test_route_segmented_test_shape_and_fallback():
    rng = np.random.RandomState(3)
    n, nt = 50, 8
    oof = rng.rand(n, 3)
    test = rng.rand(nt, 3)
    perf = (rng.rand(n, 3) < 0.5).astype(np.float64)
    cost = np.tile([0.001, 0.05, 0.05], (n, 1))
    train_segments = np.array(["math"] * 30 + ["general"] * 20)
    test_segments = np.array(["math"] * 4 + ["rare"] * 4)
    pred, rows = cb.route_segmented_test(
        [oof], [test], perf, cost, 0.0772, train_segments, test_segments,
        np.array([0.0, 0.03]), min_segment_rows=10, k_idx=0,
    )
    assert pred.shape == (nt,)
    assert ((pred >= 0) & (pred < 3)).all()
    assert {r["segment"] for r in rows} == {"math", "rare"}
    assert next(r for r in rows if r["segment"] == "rare")["source"] == "global"
