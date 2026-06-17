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


def _add_calibrated(oof_list, test_list, names, oof, test, perf, name):
    co, ct = er.isotonic_calibrate(oof, perf, test)
    oof_list.append(co); test_list.append(ct); names.append(name)


def classical_learners(cfg, train, test, perf, folds, extra_tr=None, extra_te=None):
    """Linear (Ridge) + LightGBM base learners, calibrated. Reuses cache when present,
    else computes (so any phase can call this and get the full classical ensemble)."""
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
    _add_calibrated(oof_list, test_list, names, lin_oof, lin_test, perf, "linear")

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
        _add_calibrated(oof_list, test_list, names, g_oof, g_test, perf, "lgbm")
    return oof_list, test_list, names


def route_and_submit(cfg, oof_list, test_list, perf, cost, cost_const, denom,
                     test, sample, names):
    _log("tuning ensemble weights...")
    w, w_reward = er.tune_weights(oof_list, perf, cost, cost_const, denom, step=0.1)
    oof_blend = er.weighted_average(oof_list, w)
    test_blend = er.weighted_average(test_list, w)
    _log(f"weights={list(np.round(w, 2))} (oof {w_reward:.5f}); tuning per-model bias...")
    bias = er.tune_bias(oof_blend, perf, cost, cost_const, denom)
    oof_reward = route_reward(er.route(oof_blend, cost_const, denom, bias), perf, cost, denom)

    rows = []
    for nm, oof in zip(names, oof_list):
        rows.append({"method": nm,
                     "oof_reward": route_reward(er.route(oof, cost_const, denom), perf, cost, denom)})
    rows.append({"method": f"ensemble({'+'.join(names)})_w={list(np.round(w, 2))}", "oof_reward": w_reward})
    rows.append({"method": "ensemble+bias", "oof_reward": oof_reward})
    k = MODEL_NAMES.index("Model_K")
    rows.append({"method": "always_K", "oof_reward": route_reward(np.full(len(perf), k), perf, cost, denom)})
    comp = pd.DataFrame(rows).sort_values("oof_reward", ascending=False)
    comp.to_csv(cfg.out_dir / "model_comparison.csv", index=False)
    print(comp.to_string(index=False))

    test_idx = er.route(test_blend, cost_const, denom, bias)
    er.write_submission(cfg.out_dir / "submission.csv", test["ID"].to_numpy(), test_idx, sample)
    dist = pd.Series([MODEL_NAMES[i] for i in test_idx]).value_counts()
    print("\nOOF ensemble+bias reward:", round(oof_reward, 5))
    print("submission model distribution:\n", dist.to_string())
    return {"oof_reward": oof_reward, "weights": w.tolist(), "bias": bias.tolist(),
            "oof_blend": oof_blend, "test_blend": test_blend}


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
                            test, sample, names)


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
    _add_calibrated(oof_list, test_list, names, enc_oof, enc_test, perf, "encoder")
    enc_solo = route_reward(er.route(oof_list[-1], cost_const, denom), perf, cost, denom)
    _log(f"encoder-alone OOF route reward: {enc_solo:.5f}")

    return route_and_submit(cfg, oof_list, test_list, perf, cost, cost_const, denom,
                            test, sample, names)


if __name__ == "__main__":
    run_phase0(CFG())
