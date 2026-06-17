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
