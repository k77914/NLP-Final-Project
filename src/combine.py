from __future__ import annotations
import json
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


def _log(msg):
    print(f"[combine] {msg}", flush=True)


def old_folds(n, seed=42, n_splits=5):
    """Reproduce the validated old notebook's CV split so new members stack on aligned OOF."""
    return [(tr, va) for tr, va in KFold(n_splits, shuffle=True, random_state=seed).split(np.arange(n))]


def load_old_member(cfg, name="weighted_ensemble_best"):
    d = cfg.root / "Output" / "router_a100_full"
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


def write_submission(path, ids, pred_idx, sample):
    sub = pd.DataFrame({"ID": np.asarray(ids), "pred_model": [MODEL_NAMES[i] for i in pred_idx]})
    assert sub["ID"].tolist() == sample["ID"].tolist()
    assert sub["pred_model"].isin(MODEL_NAMES).all()
    sub.to_csv(path, index=False)
    return path


def run_combined(cfg: CFG, gate_margin=0.002):
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
    ftr_p = cfg.cache_dir / f"llm_feats_train_{len(train)}.parquet"
    fte_p = cfg.cache_dir / f"llm_feats_test_{len(test)}.parquet"
    if ftr_p.exists() and fte_p.exists():
        ftr = pd.read_parquet(ftr_p).to_numpy(np.float32)
        fte = pd.read_parquet(fte_p).to_numpy(np.float32)
        sc = StandardScaler().fit(ftr)
        feats_tr = sc.transform(ftr).astype(np.float32)
        feats_te = sc.transform(fte).astype(np.float32)
        _log(f"LLM features: {feats_tr.shape[1]} dims")
    else:
        _log("LLM features not found -> encoder text-only, LLM-GBM skipped")

    # Encoder member (utility, optionally + LLM feats)
    from . import models_encoder as me
    etag = me.cache_tag(cfg, len(train))
    eo_p, et_p = cfg.cache_dir / f"enc_oof_{etag}.npy", cfg.cache_dir / f"enc_test_{etag}.npy"
    if (not cfg.smoke) and eo_p.exists() and et_p.exists():
        _log("encoder predictions loaded from cache")
        enc_oof, enc_test = np.load(eo_p), np.load(et_p)
    else:
        _log("training utility encoder (5 folds + full-fit)...")
        ts = time.time()
        enc_oof, enc_test = me.encoder_oof_and_test(cfg, train, test, Y, folds, feats_tr, feats_te)
        if not cfg.smoke:
            np.save(eo_p, enc_oof); np.save(et_p, enc_test)
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
    new_idx = list(range(1, len(members_oof)))
    stable = {names[i]: sum(r["weights"][i] > 0 for r in fold_rows) for i in new_idx}
    _log(f"members={names}")
    _log(f"honest CV: old_only={cv_old:.5f}  combined={cv_all:.5f}  (gate needs +{gate_margin:.3f})")
    _log(f"new-member fold-stability (folds with positive weight, /{len(folds)}): {stable}")

    passed = (cv_all >= cv_old + gate_margin) and all(v >= 3 for v in stable.values())

    # Final full-data policy + submission for the combined candidate
    _, w_full, mg_full = select_weights_margin(members_oof, perf, cost, denom, margins)
    test_idx = route_margin_k(blend(members_test, w_full), mg_full)
    out = cfg.out_dir / "submission_combined.csv"
    write_submission(out, test["ID"].to_numpy(), test_idx, sample)
    dist = pd.Series([MODEL_NAMES[i] for i in test_idx]).value_counts()

    pd.DataFrame(fold_rows).to_csv(cfg.out_dir / "combined_crossfit_folds.csv", index=False)
    pd.DataFrame([{"members": "+".join(names), "cv_old_only": cv_old, "cv_combined": cv_all,
                   "gate_passed": passed, "stability": json.dumps(stable),
                   "final_weights": json.dumps(dict(zip(names, [round(x, 3) for x in w_full]))),
                   "final_margin": mg_full}]).to_csv(cfg.out_dir / "combined_report.csv", index=False)

    print(f"\nGATE {'PASSED' if passed else 'NOT passed'}: "
          f"combined honest CV {cv_all:.5f} vs old-only {cv_old:.5f}")
    print(f"final weights={dict(zip(names, [round(x, 3) for x in w_full]))} margin={mg_full:.4f}")
    print("combined submission distribution:\n", dist.to_string())
    if passed:
        print(f"\n-> SUBMIT {out} (cleared the gate).")
    else:
        print("\n-> DO NOT submit combined; keep the validated 0.470 fallback "
              "(Output/router_a100_full/submission_candidate_k_fallback_q05.csv).")
    return {"cv_old": cv_old, "cv_combined": cv_all, "passed": passed,
            "weights": w_full, "margin": mg_full, "names": names}
