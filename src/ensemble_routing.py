from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.isotonic import IsotonicRegression

from .metric import MODEL_NAMES, expected_reward_matrix, route_reward


def route(p_hat, cost_const, denom, bias=None) -> np.ndarray:
    er = expected_reward_matrix(p_hat, cost_const, denom)
    if bias is not None:
        er = er + np.asarray(bias, np.float64).reshape(1, -1)
    return er.argmax(1)


def isotonic_calibrate(oof, Y, test=None):
    """Fit per-model isotonic on OOF; return (calibrated_oof, calibrated_test_or_None)."""
    oof = np.asarray(oof, np.float64)
    cal_oof = np.zeros_like(oof)
    test_arr = None if test is None else np.asarray(test, np.float64)
    cal_test = None if test is None else np.zeros_like(test_arr)
    for j in range(oof.shape[1]):
        if len(np.unique(Y[:, j])) < 2:
            cal_oof[:, j] = oof[:, j]
            if test is not None:
                cal_test[:, j] = test_arr[:, j]
            continue
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(oof[:, j], Y[:, j])
        cal_oof[:, j] = ir.predict(oof[:, j])
        if test is not None:
            cal_test[:, j] = ir.predict(test_arr[:, j])
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


def tune_bias(p_hat, perf, cost, cost_const, denom, grid=None, passes=3):
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
