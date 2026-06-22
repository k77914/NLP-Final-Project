from __future__ import annotations
import json
import shutil
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .config import CFG, seed_everything
from . import data, features, models_classical as mc
from .metric import MODEL_NAMES, route_reward, cost_denominator

K_IDX = MODEL_NAMES.index("Model_K")
SEGMENT_ORDER = [
    "code", "mcq", "math", "long", "hard_general",
    "short", "easy_general", "general",
]


def _log(msg):
    print(f"[combine] {msg}", flush=True)


def old_folds(n, seed=42, n_splits=5):
    """Reproduce the validated old notebook's CV split so new members stack on aligned OOF."""
    return [(tr, va) for tr, va in KFold(n_splits, shuffle=True, random_state=seed).split(np.arange(n))]


def _old_dir(cfg, marker="weighted_ensemble_best_oof_predictions.npz"):
    """Locate the validated-artifacts directory (its name varies: router_a100_full,
    router_a100_exact_metric_v2, ...). Returns the dir containing the marker npz."""
    out = cfg.root / "Output"
    for name in ("router_a100_full", "router_a100_exact_metric_v2", "router_a100"):
        if (out / name / marker).exists():
            return out / name
    hits = sorted(out.rglob(marker))
    if hits:
        return hits[0].parent
    raise FileNotFoundError(
        f"Validated artifacts ({marker}) not found under {out}. "
        "Copy the old run's *_predictions.npz dir (e.g. router_a100_exact_metric_v2) there.")


def load_old_member(cfg, name="weighted_ensemble_best"):
    d = _old_dir(cfg)
    oof = np.load(d / f"{name}_oof_predictions.npz")["scores"].astype(np.float64)
    test = np.load(d / f"{name}_test_predictions.npz")["scores"].astype(np.float64)
    return oof, test


# ---- utility-space routing (scores already net out cost; do NOT re-subtract it) ----
def route_argmax(util):
    return np.asarray(util).argmax(1).astype(np.int64)


def route_margin_k(util, margin, k_idx=K_IDX):
    """argmax utility, but fall back to Model_K when the top-2 gap is <= margin (low confidence)."""
    util = np.asarray(util)
    idx = util.argmax(1)
    top2 = np.sort(np.partition(util, -2, axis=1)[:, -2:], axis=1)
    gap = top2[:, 1] - top2[:, 0]
    return np.where(gap > margin, idx, k_idx).astype(np.int64)


def blend(mats, w):
    w = np.asarray(w, np.float64)
    w = w / w.sum()
    return sum(wi * np.asarray(m, np.float64) for wi, m in zip(w, mats))


def _weight_grid(n, step=0.5):
    units = int(round(1 / step))

    def gen(pos, rem, acc):
        if pos == n - 1:
            yield acc + [rem]
            return
        for v in range(rem + 1):
            yield from gen(pos + 1, rem - v, acc + [v])

    for combo in gen(0, units, []):
        yield [x / units for x in combo]


def select_weights_margin(member_oofs, perf, cost, denom, margins, step=0.5, k_idx=K_IDX):
    """Grid-search blend weights + K-fallback margin maximizing utility-route reward."""
    best = (-1e9, None, 0.0)
    for w in _weight_grid(len(member_oofs), step):
        if sum(x > 0 for x in w) < 1:
            continue
        bl = blend(member_oofs, w)
        for mg in margins:
            r = route_reward(route_margin_k(bl, mg, k_idx), perf, cost, denom)
            if r > best[0]:
                best = (r, w, float(mg))
    return best  # (reward, weights, margin)


def crossfit_cv(member_oofs, perf, cost, denom, folds, margins, step=0.5, k_idx=K_IDX):
    """Honest estimate: select weights+margin on the other folds, evaluate on each held-out fold."""
    n = len(perf)
    pred = np.empty(n, np.int64)
    rows = []
    for of, (tr, va) in enumerate(folds):
        tr_oofs = [m[tr] for m in member_oofs]
        va_oofs = [m[va] for m in member_oofs]
        _, w, mg = select_weights_margin(tr_oofs, perf[tr], cost[tr], denom, margins, step, k_idx)
        pred[va] = route_margin_k(blend(va_oofs, w), mg, k_idx)
        rows.append({"fold": of, "weights": w, "margin": mg,
                     "reward": route_reward(pred[va], perf[va], cost[va], denom),
                     "always_K": route_reward(np.full(len(va), k_idx), perf[va], cost[va], denom)})
    return route_reward(pred, perf, cost, denom), rows


def _ordered_segments(labels):
    present = set(np.asarray(labels, dtype=object).tolist())
    ordered = [s for s in SEGMENT_ORDER if s in present]
    ordered += sorted(present - set(ordered))
    return ordered


def make_query_segments(df: pd.DataFrame, llm_features: pd.DataFrame | None = None) -> np.ndarray:
    """Assign a cheap, gold-free segment label per query for segment-specific routing."""
    q = df["query"].fillna("").astype(str)
    hc = features.handcrafted_features(q)
    labels = np.full(len(df), "general", dtype=object)

    short = hc["char_len"].to_numpy() < 300
    long = hc["char_len"].to_numpy() > 5000
    mathy = (
        (hc["latex"].to_numpy() > 0)
        | ((hc["digit_ratio"].to_numpy() > 0.08) & (hc["word_count"].to_numpy() > 20))
        | q.str.contains(
            r"\b(?:prove|solve|equation|integer|probability|geometry|integral|derivative|matrix)\b",
            case=False, regex=True,
        ).to_numpy()
    )
    mcq = (hc["is_mcq"].to_numpy() > 0) | (hc["n_choices"].to_numpy() >= 2)
    code = (
        (hc["code"].to_numpy() > 0)
        | q.str.contains(
            r"```|\b(?:def|class|function|return|import)\b|#include|\bSELECT\b.+\bFROM\b",
            case=False, regex=True,
        ).to_numpy()
    )

    labels[short] = "short"
    labels[long] = "long"
    labels[mathy] = "math"
    labels[mcq] = "mcq"
    labels[code] = "code"

    if isinstance(llm_features, pd.DataFrame) and len(llm_features) == len(df):
        general = labels == "general"
        diff = llm_features.get("judge_difficulty")
        psolv = llm_features.get("judge_p_solvable")
        agree = llm_features.get("sc_agreement")
        if diff is not None or psolv is not None or agree is not None:
            hard = np.zeros(len(labels), dtype=bool)
            easy = np.zeros(len(labels), dtype=bool)
            if diff is not None:
                d = diff.to_numpy(np.float32)
                hard |= d >= 8.0
                easy |= d <= 3.0
            if psolv is not None:
                p = psolv.to_numpy(np.float32)
                hard |= p <= 0.25
                easy &= p >= 0.75
            if agree is not None:
                a = agree.to_numpy(np.float32)
                hard |= a <= 0.25
                easy &= a >= 0.75
            labels[general & easy] = "easy_general"
            labels[general & hard] = "hard_general"
    return labels


def crossfit_segmented_cv(member_oofs, perf, cost, denom, folds, segments, margins,
                          step=0.5, min_segment_rows=200, k_idx=K_IDX):
    """Cross-fit estimate with segment-specific weights/margins.

    For each outer fold, segment policies are selected only from that fold's training
    rows. Small segments fall back to the fold's global policy to avoid tiny-bucket
    overfit.
    """
    n = len(perf)
    segments = np.asarray(segments, dtype=object)
    pred = np.empty(n, np.int64)
    rows = []
    for of, (tr, va) in enumerate(folds):
        tr = np.asarray(tr, np.int64)
        va = np.asarray(va, np.int64)
        global_reward, global_w, global_mg = select_weights_margin(
            [m[tr] for m in member_oofs], perf[tr], cost[tr], denom, margins, step, k_idx
        )
        for seg in _ordered_segments(segments[va]):
            va_seg = va[segments[va] == seg]
            tr_seg = tr[segments[tr] == seg]
            source = "global"
            fit_reward, w, mg = global_reward, global_w, global_mg
            if len(tr_seg) >= min_segment_rows:
                fit_reward, w, mg = select_weights_margin(
                    [m[tr_seg] for m in member_oofs],
                    perf[tr_seg], cost[tr_seg], denom, margins, step, k_idx,
                )
                source = "segment"
            fold_pred = route_margin_k(blend([m[va_seg] for m in member_oofs], w), mg, k_idx)
            pred[va_seg] = fold_pred
            rows.append({
                "fold": int(of),
                "segment": seg,
                "source": source,
                "n_train": int(len(tr_seg)),
                "n_val": int(len(va_seg)),
                "fit_reward": float(fit_reward),
                "reward": route_reward(fold_pred, perf[va_seg], cost[va_seg], denom),
                "always_K": route_reward(np.full(len(va_seg), k_idx), perf[va_seg], cost[va_seg], denom),
                "weights": [float(x) for x in w],
                "margin": float(mg),
                "n_leave_K": int((fold_pred != k_idx).sum()),
            })
    return route_reward(pred, perf, cost, denom), rows


def route_segmented_test(member_oofs, members_test, perf, cost, denom, train_segments,
                         test_segments, margins, step=0.5, min_segment_rows=200,
                         k_idx=K_IDX):
    """Fit segment policies on all train rows and route the test rows."""
    train_segments = np.asarray(train_segments, dtype=object)
    test_segments = np.asarray(test_segments, dtype=object)
    pred = np.empty(len(test_segments), np.int64)
    rows = []
    global_reward, global_w, global_mg = select_weights_margin(
        member_oofs, perf, cost, denom, margins, step, k_idx
    )
    for seg in _ordered_segments(test_segments):
        te = np.where(test_segments == seg)[0]
        tr = np.where(train_segments == seg)[0]
        source = "global"
        fit_reward, w, mg = global_reward, global_w, global_mg
        if len(tr) >= min_segment_rows:
            fit_reward, w, mg = select_weights_margin(
                [m[tr] for m in member_oofs], perf[tr], cost[tr], denom, margins, step, k_idx
            )
            source = "segment"
        pred[te] = route_margin_k(blend([m[te] for m in members_test], w), mg, k_idx)
        rows.append({
            "segment": seg,
            "source": source,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "fit_reward": float(fit_reward),
            "weights": [float(x) for x in w],
            "margin": float(mg),
            "n_leave_K": int((pred[te] != k_idx).sum()),
        })
    return pred, rows


def write_submission(path, ids, pred_idx, sample):
    sub = pd.DataFrame({"ID": np.asarray(ids), "pred_model": [MODEL_NAMES[i] for i in pred_idx]})
    assert sub["ID"].tolist() == sample["ID"].tolist()
    assert sub["pred_model"].isin(MODEL_NAMES).all()
    sub.to_csv(path, index=False)
    return path


def run_combined(cfg: CFG, gate_margin=0.002, segment_gate_margin=0.001,
                 segment_min_rows=200):
    """Stack the validated old ensemble + utility encoder + LLM-GBM; gate the combined
    candidate against the old-ensemble-only honest CV before recommending submission."""
    seed_everything(cfg.seed)
    train, test, sample = data.load_data(cfg)
    perf, cost = data.build_targets(train)
    denom = cost_denominator(cost)
    Y = (0.85 * perf - 0.15 * (cost / denom)).astype(np.float32)
    folds = old_folds(len(train), cfg.seed, cfg.n_splits)
    margins = np.linspace(0.0, 0.03, 31)

    members_oof, members_test, names = [], [], []
    o, t = load_old_member(cfg, "weighted_ensemble_best")
    members_oof.append(o); members_test.append(t); names.append("old_ensemble")

    # LLM difficulty features (optional shared input)
    feats_tr = feats_te = None
    ftr_df = fte_df = None
    ftr_p = cfg.cache_dir / f"llm_feats_train_{len(train)}.parquet"
    fte_p = cfg.cache_dir / f"llm_feats_test_{len(test)}.parquet"
    if ftr_p.exists() and fte_p.exists():
        ftr_df = pd.read_parquet(ftr_p)
        fte_df = pd.read_parquet(fte_p)
        ftr = ftr_df.to_numpy(np.float32)
        fte = fte_df.to_numpy(np.float32)
        sc = StandardScaler().fit(ftr)
        feats_tr = sc.transform(ftr).astype(np.float32)
        feats_te = sc.transform(fte).astype(np.float32)
        _log(f"LLM features: {feats_tr.shape[1]} dims")
    else:
        _log("LLM features not found -> encoder text-only, LLM-GBM skipped")

    # Encoder member (utility, optionally + LLM feats)
    from . import models_encoder as me
    # NOTE: compute the final-cache tag WITHOUT setting _encoder_uses_feats, so it stays
    # "util_..." (matches predictions saved by earlier runs); fold caches use "utilf_..." internally.
    etag = me.cache_tag(cfg, len(train))
    eo_p, et_p = cfg.cache_dir / f"enc_oof_{etag}.npy", cfg.cache_dir / f"enc_test_{etag}.npy"
    dc = getattr(cfg, "drive_cache", None)
    if (not cfg.smoke) and eo_p.exists() and et_p.exists():
        _log("encoder predictions loaded from cache")
        enc_oof, enc_test = np.load(eo_p), np.load(et_p)
    else:
        _log("training utility encoder (5 folds + full-fit)...")
        ts = time.time()
        enc_oof, enc_test = me.encoder_oof_and_test(cfg, train, test, Y, folds, feats_tr,
                                                    feats_te, drive_cache=dc)
        if not cfg.smoke:
            np.save(eo_p, enc_oof); np.save(et_p, enc_test)
            if dc:
                Path(dc).mkdir(parents=True, exist_ok=True)
                shutil.copy(eo_p, Path(dc) / eo_p.name); shutil.copy(et_p, Path(dc) / et_p.name)
                _log("mirrored encoder predictions to Drive cache")
        _log(f"encoder done ({time.time() - ts:.0f}s)")
    members_oof.append(enc_oof.astype(np.float64)); members_test.append(enc_test.astype(np.float64))
    names.append("encoder")

    # LLM-GBM member (utility on embeddings + handcrafted + LLM feats)
    if feats_tr is not None:
        _log("LLM-GBM member...")
        emb_tr = features.load_or_compute_embeddings(cfg, train, "train")
        emb_te = features.load_or_compute_embeddings(cfg, test, "test")
        hc_tr = features.handcrafted_matrix(train["query"])
        hc_te = features.handcrafted_matrix(test["query"])
        Xg_tr = np.hstack([emb_tr, hc_tr, feats_tr]).astype(np.float32)
        Xg_te = np.hstack([emb_te, hc_te, feats_te]).astype(np.float32)
        g_oof = mc.lgbm_oof(Xg_tr, Y, folds, cfg.lgbm_estimators, cfg.lgbm_lr,
                            cfg.lgbm_leaves, cfg.lgbm_min_child, cfg.seed, clip=False)
        g_test = mc.lgbm_full(Xg_tr, Y, Xg_te, cfg.lgbm_estimators, cfg.lgbm_lr,
                              cfg.lgbm_leaves, cfg.lgbm_min_child, cfg.seed, clip=False)
        members_oof.append(g_oof.astype(np.float64)); members_test.append(g_test.astype(np.float64))
        names.append("llm_gbm")

    # Honest CV: old-only vs all members
    cv_old, _ = crossfit_cv(members_oof[:1], perf, cost, denom, folds, margins)
    cv_all, fold_rows = crossfit_cv(members_oof, perf, cost, denom, folds, margins)
    train_segments = make_query_segments(train, ftr_df)
    test_segments = make_query_segments(test, fte_df)
    cv_seg, seg_fold_rows = crossfit_segmented_cv(
        members_oof, perf, cost, denom, folds, train_segments, margins,
        min_segment_rows=segment_min_rows,
    )
    new_idx = list(range(1, len(members_oof)))
    stable = {names[i]: sum(r["weights"][i] > 0 for r in fold_rows) for i in new_idx}
    seg_member_coverage = {
        names[i]: sum(r["n_val"] for r in seg_fold_rows if r["weights"][i] > 0) / len(train)
        for i in new_idx
    }
    seg_policy_coverage = sum(r["n_val"] for r in seg_fold_rows if r["source"] == "segment") / len(train)
    _log(f"members={names}")
    _log(f"honest CV: old_only={cv_old:.5f}  combined={cv_all:.5f}  (gate needs +{gate_margin:.3f})")
    _log(f"segmented CV: segmented={cv_seg:.5f}  (needs combined +{segment_gate_margin:.3f})")
    _log(f"new-member fold-stability (folds with positive weight, /{len(folds)}): {stable}")
    _log(f"segmented trained-row coverage={seg_policy_coverage:.1%}; new-member row coverage={seg_member_coverage}")

    passed = (cv_all >= cv_old + gate_margin) and all(v >= 3 for v in stable.values())
    seg_passed = (
        cv_seg >= cv_old + gate_margin
        and cv_seg >= cv_all + segment_gate_margin
        and seg_policy_coverage >= 0.20
    )

    # Final full-data policy + submission for the combined candidate
    _, w_full, mg_full = select_weights_margin(members_oof, perf, cost, denom, margins)
    test_idx = route_margin_k(blend(members_test, w_full), mg_full)
    out = cfg.out_dir / "submission_combined.csv"
    write_submission(out, test["ID"].to_numpy(), test_idx, sample)
    dist = pd.Series([MODEL_NAMES[i] for i in test_idx]).value_counts()

    seg_test_idx, seg_policy_rows = route_segmented_test(
        members_oof, members_test, perf, cost, denom, train_segments, test_segments, margins,
        min_segment_rows=segment_min_rows,
    )
    seg_out = cfg.out_dir / "submission_segmented.csv"
    write_submission(seg_out, test["ID"].to_numpy(), seg_test_idx, sample)
    seg_dist = pd.Series([MODEL_NAMES[i] for i in seg_test_idx]).value_counts()

    pd.DataFrame(fold_rows).to_csv(cfg.out_dir / "combined_crossfit_folds.csv", index=False)
    pd.DataFrame(seg_fold_rows).to_csv(cfg.out_dir / "segmented_crossfit_folds.csv", index=False)
    pd.DataFrame(seg_policy_rows).to_csv(cfg.out_dir / "segmented_policy.csv", index=False)
    pd.DataFrame([{"members": "+".join(names), "cv_old_only": cv_old, "cv_combined": cv_all,
                   "cv_segmented": cv_seg, "gate_passed": passed, "segmented_gate_passed": seg_passed,
                   "stability": json.dumps(stable),
                   "segmented_new_member_coverage": json.dumps({k: round(v, 4) for k, v in seg_member_coverage.items()}),
                   "segmented_policy_coverage": seg_policy_coverage,
                   "final_weights": json.dumps(dict(zip(names, [round(x, 3) for x in w_full]))),
                   "final_margin": mg_full}]).to_csv(cfg.out_dir / "combined_report.csv", index=False)

    print(f"\nGATE {'PASSED' if passed else 'NOT passed'}: "
          f"combined honest CV {cv_all:.5f} vs old-only {cv_old:.5f}")
    print(f"SEGMENTED GATE {'PASSED' if seg_passed else 'NOT passed'}: "
          f"segmented honest CV {cv_seg:.5f} vs combined {cv_all:.5f}")
    print(f"final weights={dict(zip(names, [round(x, 3) for x in w_full]))} margin={mg_full:.4f}")
    print("combined submission distribution:\n", dist.to_string())
    print("segmented submission distribution:\n", seg_dist.to_string())
    if seg_passed:
        print(f"\n-> SUBMIT {seg_out} (segmented cleared the stricter gate).")
    elif passed:
        print(f"\n-> SUBMIT {out} (cleared the gate).")
    else:
        print("\n-> DO NOT submit combined; keep the validated 0.470 fallback "
              "(Output/router_a100_full/submission_candidate_k_fallback_q05.csv).")
    return {"cv_old": cv_old, "cv_combined": cv_all, "passed": passed,
            "cv_segmented": cv_seg, "segmented_passed": seg_passed,
            "weights": w_full, "margin": mg_full, "names": names}
