"""Build conservative leaderboard probes from the validated router_a100_full ensemble."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Output" / "router_a100_full"
DATA = ROOT / "dataset"
MODEL_NAMES = np.array([f"Model_{letter}" for letter in "ABCDEFGHIJK"])
MODEL_K_IDX = 10


def load_scores(stem: str, split: str) -> np.ndarray:
    path = OUT / f"{stem}_{split}_predictions.npz"
    with np.load(path) as data:
        return data["scores"]


def apply_margin_fallback(scores: np.ndarray, threshold: float) -> np.ndarray:
    top2 = np.sort(np.partition(scores, -2, axis=1)[:, -2:], axis=1)
    margin = top2[:, 1] - top2[:, 0]
    pred = scores.argmax(axis=1)
    pred[margin <= threshold] = MODEL_K_IDX
    return pred


def write_submission(name: str, ids: np.ndarray, pred_idx: np.ndarray) -> Path:
    path = OUT / name
    pd.DataFrame({
        "ID": ids,
        "pred_model": MODEL_NAMES[pred_idx],
    }).to_csv(path, index=False)
    return path


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    assert test["ID"].tolist() == sample["ID"].tolist()

    oof = load_scores("weighted_ensemble_best", "oof")
    test_scores = load_scores("weighted_ensemble_best", "test")
    top2 = np.sort(np.partition(oof, -2, axis=1)[:, -2:], axis=1)
    oof_margin = top2[:, 1] - top2[:, 0]

    perf_cols = [f"{name}_performance" for name in MODEL_NAMES]
    cost_cols = [f"{name}_cost" for name in MODEL_NAMES]
    perf = train[perf_cols].to_numpy(np.float64)
    cost = train[cost_cols].to_numpy(np.float64)
    denominator = float(cost.max(axis=1).mean())

    def reward(pred_idx: np.ndarray) -> float:
        rows = np.arange(len(pred_idx))
        return float(
            0.85 * perf[rows, pred_idx].mean()
            - 0.15 * cost[rows, pred_idx].mean() / denominator
        )

    baseline_idx = test_scores.argmax(axis=1)
    saved_baseline = pd.read_csv(OUT / "submission.csv")
    assert saved_baseline["pred_model"].tolist() == MODEL_NAMES[baseline_idx].tolist()

    records = []
    for quantile in (0.05, 0.10):
        threshold = float(np.quantile(oof_margin, quantile))
        oof_idx = apply_margin_fallback(oof, threshold)
        test_idx = apply_margin_fallback(test_scores, threshold)
        path = write_submission(
            f"submission_candidate_k_fallback_q{int(quantile * 100):02d}.csv",
            test["ID"].to_numpy(),
            test_idx,
        )
        records.append({
            "candidate": path.name,
            "oof_margin_quantile": quantile,
            "threshold": threshold,
            "oof_reward": reward(oof_idx),
            "test_changes_vs_validated_046853": int((test_idx != baseline_idx).sum()),
            "test_model_K_count": int((test_idx == MODEL_K_IDX).sum()),
            "test_distribution": json.dumps(
                pd.Series(MODEL_NAMES[test_idx]).value_counts().to_dict()
            ),
        })

    all_k = np.full(len(test), MODEL_K_IDX, dtype=np.int64)
    all_k_path = write_submission(
        "submission_diagnostic_all_model_k.csv",
        test["ID"].to_numpy(),
        all_k,
    )
    records.append({
        "candidate": all_k_path.name,
        "oof_margin_quantile": None,
        "threshold": None,
        "oof_reward": reward(np.full(len(train), MODEL_K_IDX, dtype=np.int64)),
        "test_changes_vs_validated_046853": int((all_k != baseline_idx).sum()),
        "test_model_K_count": len(test),
        "test_distribution": json.dumps({"Model_K": len(test)}),
    })

    report = pd.DataFrame(records)
    report.to_csv(OUT / "candidate_probe_report.csv", index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
