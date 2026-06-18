from __future__ import annotations
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from .config import CFG, seed_everything
from . import data, features, cv, models_classical as mc, ensemble_routing as er
from .metric import MODEL_NAMES, route_reward, cost_denominator


def _log(msg):
    print(f"[router] {msg}", flush=True)


def _cache_np(path, fn, use_cache=True):
    path = Path(path)
    if use_cache and path.exists():
        _log(f"cache hit: {path.name}")
        return np.load(path)
    arr = fn()
    if use_cache:
        np.save(path, arr)
    return arr


def build_feature_blocks(cfg, train, test, extra_tr=None, extra_te=None):
    """Return dense X (for LGBM) and sparse X (for linear), train+test.
    extra_* (e.g. LLM difficulty features) are appended to the dense block only."""
    t = time.time()
    hc_tr = features.handcrafted_matrix(train["query"])
    hc_te = features.handcrafted_matrix(test["query"])
    sc = StandardScaler().fit(hc_tr)
    hc_tr_s = sc.transform(hc_tr).astype(np.float32)
    hc_te_s = sc.transform(hc_te).astype(np.float32)
    _log(f"handcrafted: {hc_tr_s.shape[1]} dims ({time.time() - t:.0f}s)")

    t = time.time()
    tfidf = features.build_tfidf(cfg)
    tr_txt = [features.cap_text(s, cfg.tfidf_max_chars) for s in train["query"]]
    te_txt = [features.cap_text(s, cfg.tfidf_max_chars) for s in test["query"]]
    Xtf_tr = tfidf.fit_transform(tr_txt)
    Xtf_te = tfidf.transform(te_txt)
    _log(f"TF-IDF: {Xtf_tr.shape[1]} features ({time.time() - t:.0f}s)")

    t = time.time()
    n_comp = max(2, min(cfg.svd_components, Xtf_tr.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=cfg.seed).fit(Xtf_tr)
    svd_tr = svd.transform(Xtf_tr).astype(np.float32)
    svd_te = svd.transform(Xtf_te).astype(np.float32)
    _log(f"SVD -> {n_comp} dims ({time.time() - t:.0f}s)")

    try:
        emb_tr = features.load_or_compute_embeddings(cfg, train, "train")
        emb_te = features.load_or_compute_embeddings(cfg, test, "test")
        _log(f"embeddings: {emb_tr.shape[1]} dims")
    except Exception as e:
        print("embeddings unavailable, proceeding without:", repr(e), flush=True)
        emb_tr = np.zeros((len(train), 0), np.float32)
        emb_te = np.zeros((len(test), 0), np.float32)

    blocks_tr = [svd_tr, emb_tr, hc_tr_s]
    blocks_te = [svd_te, emb_te, hc_te_s]
    if extra_tr is not None:
        blocks_tr.append(np.asarray(extra_tr, np.float32))
        blocks_te.append(np.asarray(extra_te, np.float32))
    dense_tr = np.hstack(blocks_tr).astype(np.float32)
    dense_te = np.hstack(blocks_te).astype(np.float32)
    sparse_tr = hstack([Xtf_tr, csr_matrix(hc_tr_s)]).tocsr()
    sparse_te = hstack([Xtf_te, csr_matrix(hc_te_s)]).tocsr()
    _log(f"feature blocks ready: dense={dense_tr.shape}, sparse={sparse_tr.shape}")
    return dense_tr, dense_te, sparse_tr, sparse_te


def classical_learners(cfg, train, test, perf, folds, extra_tr=None, extra_te=None):
    """RAW (uncalibrated) linear (Ridge) + LightGBM base learners. Calibration is done
    later (honestly) in route_and_submit. Reuses cache when present, else computes."""
    uc = not cfg.smoke
    tag = len(train)
    paths = [cfg.cache_dir / f"{p}_{tag}.npy" for p in ("lin_oof", "lin_test", "lgbm_oof", "lgbm_test")]
    if uc and all(p.exists() for p in paths):
        _log("all classical caches present; skipping feature build")
        dense_tr = dense_te = sparse_tr = sparse_te = None
    else:
        dense_tr, dense_te, sparse_tr, sparse_te = build_feature_blocks(cfg, train, test, extra_tr, extra_te)

    oof_list, test_list, names = [], [], []
    _log("linear (Ridge) base learner...")
    t = time.time()
    lin_oof = _cache_np(cfg.cache_dir / f"lin_oof_{tag}.npy",
                        lambda: mc.linear_oof(sparse_tr, perf, folds, cfg.seed, verbose=True), uc)
    lin_test = _cache_np(cfg.cache_dir / f"lin_test_{tag}.npy",
                         lambda: mc.linear_full(sparse_tr, perf, sparse_te, cfg.seed, verbose=True), uc)
    _log(f"linear ready ({time.time() - t:.0f}s)")
    oof_list.append(lin_oof); test_list.append(lin_test); names.append("linear")

    try:
        import lightgbm  # noqa: F401
        have_lgbm = True
    except Exception as e:
        print("LightGBM unavailable -> linear-only:", repr(e), flush=True)
        have_lgbm = False
    if have_lgbm:
        _log("LightGBM base learner (66 fits)...")
        t = time.time()
        g_oof = _cache_np(cfg.cache_dir / f"lgbm_oof_{tag}.npy",
            lambda: mc.lgbm_oof(dense_tr, perf, folds, cfg.lgbm_estimators, cfg.lgbm_lr,
                                cfg.lgbm_leaves, cfg.lgbm_min_child, cfg.seed, verbose=True), uc)
        g_test = _cache_np(cfg.cache_dir / f"lgbm_test_{tag}.npy",
            lambda: mc.lgbm_full(dense_tr, perf, dense_te, cfg.lgbm_estimators, cfg.lgbm_lr,
                                 cfg.lgbm_leaves, cfg.lgbm_min_child, cfg.seed, verbose=True), uc)
        _log(f"lgbm ready ({time.time() - t:.0f}s)")
        oof_list.append(g_oof); test_list.append(g_test); names.append("lgbm")
    return oof_list, test_list, names


def route_and_submit(cfg, oof_list, test_list, perf, cost, cost_const, denom, folds,
                     test, sample, names):
    """Honest calibration + conservative, K-anchored policy selection with an always-K floor.
    The reported reward is an honest (nested-calibration) estimate meant to track the LB."""
    k_idx = MODEL_NAMES.index("Model_K")

    # Per-model calibration: NESTED for honest evaluation; full-OOF fit applied to test.
    cal_oof_list, cal_test_list = [], []
    for oof, te in zip(oof_list, test_list):
        cal_oof_list.append(er.nested_calibrate(oof, perf, folds))
        _, full_cal_test = er.isotonic_calibrate(oof, perf, te)
        cal_test_list.append(full_cal_test)

    # Coarse, low-overfit weight selection on the honestly-calibrated OOF.
    if len(cal_oof_list) == 1:
        w = np.array([1.0])
    else:
        w, _ = er.tune_weights(cal_oof_list, perf, cost, cost_const, denom, step=0.5)
    blend_oof = er.weighted_average(cal_oof_list, w)
    blend_test = er.weighted_average(cal_test_list, w)

    # Select the final full-data policy, then estimate it by re-selecting calibration,
    # weights, and routing policy without each evaluated outer fold.
    sel = er.select_policy(blend_oof, perf, cost, cost_const, denom, k_idx)
    policy, margin = sel["best_policy"], sel["best_margin"]
    honest_idx, fold_details = er.crossfit_policy_predictions(
        oof_list, perf, cost, cost_const, denom, folds, k_idx, weight_step=0.5
    )
    honest = route_reward(honest_idx, perf, cost, denom)
    _log(f"weights={list(np.round(w, 2))} | honest: always_K={sel['always_K']:.5f} "
         f"argmax={sel['argmax']:.5f} k_margin={sel['k_margin']:.5f}(m={margin:.3f}) -> {policy}")
    _log(f"policy-crossfit CV reward={honest:.5f}")

    rows = [{"method": "always_K", "oof_reward": sel["always_K"]}]
    for nm, co in zip(names, cal_oof_list):
        rows.append({"method": f"{nm}_solo",
                     "oof_reward": route_reward(er.route(co, cost_const, denom), perf, cost, denom)})
    rows.append({"method": f"blend_argmax_w={list(np.round(w, 2))}", "oof_reward": sel["argmax"]})
    rows.append({"method": "policy_crossfit_estimate", "oof_reward": honest})
    rows.append({"method": f"selection_fit_k_margin(m={margin:.3f})", "oof_reward": sel["k_margin"]})
    comp = pd.DataFrame(rows).sort_values("oof_reward", ascending=False)
    comp.to_csv(cfg.out_dir / "model_comparison.csv", index=False)
    pd.DataFrame(fold_details).to_csv(
        cfg.out_dir / "policy_crossfit_folds.csv", index=False
    )
    print(comp.to_string(index=False))

    if policy == "always_K":
        test_idx = np.full(len(test), k_idx, dtype=np.int64)
    elif policy == "k_margin":
        test_idx = er.route_k_anchored(blend_test, cost_const, denom, k_idx, margin)
    else:
        test_idx = er.route(blend_test, cost_const, denom)
    er.write_submission(cfg.out_dir / "submission.csv", test["ID"].to_numpy(), test_idx, sample)

    dist = pd.Series([MODEL_NAMES[i] for i in test_idx]).value_counts()
    print(f"\nPolicy-crossfit CV reward: {honest:.5f}  "
          f"[always_K {sel['always_K']:.5f}]")
    print(f"Final full-data policy: {policy}, margin={margin:.3f}, "
          f"weights={list(np.round(w, 2))}")
    print("submission model distribution:\n", dist.to_string())
    return {"honest_reward": honest, "policy": policy, "margin": margin, "weights": w.tolist(),
            "blend_oof": blend_oof, "blend_test": blend_test, "k_idx": k_idx}


def _prep(cfg):
    _log("loading data...")
    train, test, sample = data.load_data(cfg)
    _log(f"train={train.shape} test={test.shape}")
    perf, cost = data.build_targets(train)
    cost_const = data.cost_constants(cost)
    denom = cost_denominator(cost)
    folds = cv.get_folds(cfg, perf, train["query"])
    return train, test, sample, perf, cost, cost_const, denom, folds


def run_phase0(cfg: CFG):
    seed_everything(cfg.seed)
    train, test, sample, perf, cost, cost_const, denom, folds = _prep(cfg)
    oof_list, test_list, names = classical_learners(cfg, train, test, perf, folds)
    return route_and_submit(cfg, oof_list, test_list, perf, cost, cost_const, denom,
                            folds, test, sample, names)


def run_phase1(cfg: CFG):
    """Phase 1: add the fine-tuned ModernBERT encoder, ensembled with the classical learners."""
    from . import models_encoder as me
    seed_everything(cfg.seed)
    train, test, sample, perf, cost, cost_const, denom, folds = _prep(cfg)
    oof_list, test_list, names = classical_learners(cfg, train, test, perf, folds)

    uc = not cfg.smoke
    etag = me.cache_tag(cfg, len(train))
    enc_oof_p = cfg.cache_dir / f"enc_oof_{etag}.npy"
    enc_test_p = cfg.cache_dir / f"enc_test_{etag}.npy"
    if uc and enc_oof_p.exists() and enc_test_p.exists():
        _log("loaded cached encoder predictions")
        enc_oof, enc_test = np.load(enc_oof_p), np.load(enc_test_p)
    else:
        _log("training ModernBERT encoder (5 folds + full-fit)...")
        t = time.time()
        enc_oof, enc_test = me.encoder_oof_and_test(cfg, train, test, perf, folds)
        if uc:
            np.save(enc_oof_p, enc_oof); np.save(enc_test_p, enc_test)
        _log(f"encoder done ({time.time() - t:.0f}s)")
    oof_list.append(enc_oof); test_list.append(enc_test); names.append("encoder")

    return route_and_submit(cfg, oof_list, test_list, perf, cost, cost_const, denom,
                            folds, test, sample, names)


if __name__ == "__main__":
    run_phase0(CFG())
