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


def route_k_anchored(p_hat, cost_const, denom, k_idx, margin):
    """Argmax expected reward, but only leave Model_K when the best alternative beats K's
    expected reward by `margin`. A conservative, K-anchored policy (1 parameter)."""
    er = expected_reward_matrix(p_hat, cost_const, denom)
    idx = er.argmax(1)
    best = er[np.arange(len(er)), idx]
    keep = (best - er[:, k_idx]) > margin
    return np.where(keep, idx, k_idx).astype(np.int64)


def nested_calibrate(oof, perf, folds):
    """Honest per-model isotonic calibration: each fold's rows are calibrated by isotonic
    fit on the OTHER folds, removing the in-sample optimism of fitting+scoring on the same OOF."""
    oof = np.asarray(oof, np.float64)
    out = np.zeros_like(oof)
    n = len(oof)
    for _, va in folds:
        tr = np.setdiff1d(np.arange(n), va)
        _, cal_va = isotonic_calibrate(oof[tr], perf[tr], oof[va])
        out[va] = cal_va
    return out


def select_policy(p_hat, perf, cost, cost_const, denom, k_idx, margins=None):
    """Score candidate routing policies (always-K, plain argmax, K-anchored margin) and
    return their rewards plus the best pick. always-K is always a candidate, so the chosen
    policy can never score below the trivial baseline on the evaluation data."""
    if margins is None:
        margins = np.linspace(0.0, 0.25, 26)
    a_k = route_reward(np.full(len(perf), k_idx), perf, cost, denom)
    arg = route_reward(route(p_hat, cost_const, denom), perf, cost, denom)
    best_m, best_km = 0.0, -1e9
    for m in margins:
        r = route_reward(route_k_anchored(p_hat, cost_const, denom, k_idx, m), perf, cost, denom)
        if r > best_km:
            best_km, best_m = r, float(m)
    cands = {"always_K": a_k, "argmax": arg, "k_margin": best_km}
    return {**cands, "best_margin": best_m, "best_policy": max(cands, key=cands.get)}


def crossfit_policy_predictions(oof_list, perf, cost, cost_const, denom, folds, k_idx,
                                weight_step=0.5, margins=None):
    """Cross-fit policy selection on already-OOF base predictions.

    This removes calibration/weight/margin selection leakage from each evaluated
    fold. It is a policy-level estimate; a fully nested estimate would also retrain
    every base learner inside each outer fold.
    """
    oof_list = [np.asarray(o, np.float64) for o in oof_list]
    perf = np.asarray(perf)
    cost = np.asarray(cost)
    n = len(perf)
    pred_idx = np.empty(n, dtype=np.int64)
    details = []

    for outer_fold, (tr, va) in enumerate(folds):
        tr = np.asarray(tr, np.int64)
        va = np.asarray(va, np.int64)
        local_pos = np.full(n, -1, dtype=np.int64)
        local_pos[tr] = np.arange(len(tr))

        inner_calibrated = []
        outer_val_calibrated = []
        for raw_oof in oof_list:
            inner_oof = np.zeros((len(tr), raw_oof.shape[1]), dtype=np.float64)
            for inner_fold, (_, inner_va_all) in enumerate(folds):
                if inner_fold == outer_fold:
                    continue
                inner_va = np.intersect1d(tr, inner_va_all, assume_unique=False)
                inner_tr = np.setdiff1d(tr, inner_va, assume_unique=False)
                _, calibrated = isotonic_calibrate(
                    raw_oof[inner_tr], perf[inner_tr], raw_oof[inner_va]
                )
                inner_oof[local_pos[inner_va]] = calibrated

            _, calibrated_va = isotonic_calibrate(
                raw_oof[tr], perf[tr], raw_oof[va]
            )
            inner_calibrated.append(inner_oof)
            outer_val_calibrated.append(calibrated_va)

        if len(inner_calibrated) == 1:
            weights = np.array([1.0])
        else:
            weights, _ = tune_weights(
                inner_calibrated, perf[tr], cost[tr], cost_const, denom,
                step=weight_step,
            )
        train_blend = weighted_average(inner_calibrated, weights)
        val_blend = weighted_average(outer_val_calibrated, weights)
        selected = select_policy(
            train_blend, perf[tr], cost[tr], cost_const, denom, k_idx,
            margins=margins,
        )
        policy = selected["best_policy"]
        margin = selected["best_margin"]

        if policy == "always_K":
            fold_pred = np.full(len(va), k_idx, dtype=np.int64)
        elif policy == "k_margin":
            fold_pred = route_k_anchored(
                val_blend, cost_const, denom, k_idx, margin
            )
        else:
            fold_pred = route(val_blend, cost_const, denom)
        pred_idx[va] = fold_pred

        details.append({
            "fold": outer_fold,
            "weights": weights.tolist(),
            "policy": policy,
            "margin": margin,
            "reward": route_reward(fold_pred, perf[va], cost[va], denom),
            "always_K_reward": route_reward(
                np.full(len(va), k_idx), perf[va], cost[va], denom
            ),
            "n_rows": len(va),
            "n_leave_K": int((fold_pred != k_idx).sum()),
        })

    return pred_idx, details


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
