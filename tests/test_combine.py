import numpy as np
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
