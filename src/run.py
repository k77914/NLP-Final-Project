from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from .config import CFG, seed_everything
from . import data, features, cv, models_classical as mc, ensemble_routing as er
from .metric import MODEL_NAMES, route_reward, cost_denominator


def _cache_np(path, fn, use_cache=True):
    path = Path(path)
    if use_cache and path.exists():
        return np.load(path)
    arr = fn()
    if use_cache:
        np.save(path, arr)
    return arr


def build_feature_blocks(cfg, train, test, extra_tr=None, extra_te=None):
    """Return dense X (for LGBM) and sparse X (for linear), train+test.
    extra_* (e.g. LLM difficulty features) are appended to the dense block only."""
    hc_tr = features.handcrafted_matrix(train["query"])
    hc_te = features.handcrafted_matrix(test["query"])
    sc = StandardScaler().fit(hc_tr)
    hc_tr_s = sc.transform(hc_tr).astype(np.float32)
    hc_te_s = sc.transform(hc_te).astype(np.float32)

    tfidf = features.build_tfidf(cfg)
    tr_txt = [features.cap_text(t, cfg.tfidf_max_chars) for t in train["query"]]
    te_txt = [features.cap_text(t, cfg.tfidf_max_chars) for t in test["query"]]
    Xtf_tr = tfidf.fit_transform(tr_txt)
    Xtf_te = tfidf.transform(te_txt)
    n_comp = max(2, min(cfg.svd_components, Xtf_tr.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=cfg.seed).fit(Xtf_tr)
    svd_tr = svd.transform(Xtf_tr).astype(np.float32)
    svd_te = svd.transform(Xtf_te).astype(np.float32)

    try:
        emb_tr = features.load_or_compute_embeddings(cfg, train, "train")
        emb_te = features.load_or_compute_embeddings(cfg, test, "test")
    except Exception as e:
        print("embeddings unavailable, proceeding without:", repr(e))
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
    return dense_tr, dense_te, sparse_tr, sparse_te


def route_and_submit(cfg, oof_list, test_list, perf, cost, cost_const, denom,
                     test, sample, names):
    w, w_reward = er.tune_weights(oof_list, perf, cost, cost_const, denom, step=0.1)
    oof_blend = er.weighted_average(oof_list, w)
    test_blend = er.weighted_average(test_list, w)
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


def _add_calibrated(oof_list, test_list, names, oof, test, perf, name):
    co, ct = er.isotonic_calibrate(oof, perf, test)
    oof_list.append(co); test_list.append(ct); names.append(name)


def run_phase0(cfg: CFG):
    seed_everything(cfg.seed)
    train, test, sample = data.load_data(cfg)
    perf, cost = data.build_targets(train)
    cost_const = data.cost_constants(cost)
    denom = cost_denominator(cost)
    folds = cv.get_folds(cfg, perf, train["query"])
    dense_tr, dense_te, sparse_tr, sparse_te = build_feature_blocks(cfg, train, test)
    uc = not cfg.smoke
    tag = len(train)

    oof_list, test_list, names = [], [], []
    lin_oof = _cache_np(cfg.cache_dir / f"lin_oof_{tag}.npy",
                        lambda: mc.linear_oof(sparse_tr, perf, folds, cfg.seed), uc)
    lin_test = _cache_np(cfg.cache_dir / f"lin_test_{tag}.npy",
                         lambda: mc.linear_full(sparse_tr, perf, sparse_te, cfg.seed), uc)
    _add_calibrated(oof_list, test_list, names, lin_oof, lin_test, perf, "linear")

    try:
        import lightgbm  # noqa: F401
        have_lgbm = True
    except Exception as e:
        print("LightGBM unavailable -> linear-only Phase 0:", repr(e))
        have_lgbm = False
    if have_lgbm:
        g_oof = _cache_np(cfg.cache_dir / f"lgbm_oof_{tag}.npy",
            lambda: mc.lgbm_oof(dense_tr, perf, folds, cfg.lgbm_estimators, cfg.lgbm_lr,
                                cfg.lgbm_leaves, cfg.lgbm_min_child, cfg.seed), uc)
        g_test = _cache_np(cfg.cache_dir / f"lgbm_test_{tag}.npy",
            lambda: mc.lgbm_full(dense_tr, perf, dense_te, cfg.lgbm_estimators, cfg.lgbm_lr,
                                 cfg.lgbm_leaves, cfg.lgbm_min_child, cfg.seed), uc)
        _add_calibrated(oof_list, test_list, names, g_oof, g_test, perf, "lgbm")

    return route_and_submit(cfg, oof_list, test_list, perf, cost, cost_const, denom,
                            test, sample, names)


if __name__ == "__main__":
    run_phase0(CFG())
