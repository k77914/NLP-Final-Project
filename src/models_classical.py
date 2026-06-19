from __future__ import annotations
import warnings
import numpy as np
from sklearn.linear_model import Ridge

# Per-model regressors. `clip=True` bounds predictions to [0,1] (for performance targets);
# pass clip=False when regressing the utility target Y (which is negative for expensive models).
# LightGBM records feature names at fit time; predicting on a nameless ndarray triggers a
# cosmetic sklearn warning. Predictions are positional and correct - silence the noise.
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def _maybe_clip(a, clip):
    a = np.asarray(a, np.float32)
    return np.clip(a, 0.0, 1.0) if clip else a


def _lgbm_factory(n_estimators, lr, leaves, min_child, seed=42):
    from lightgbm import LGBMRegressor
    return lambda: LGBMRegressor(
        objective="regression", n_estimators=n_estimators, learning_rate=lr,
        num_leaves=leaves, min_child_samples=min_child, subsample=0.9,
        colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=0.2,
        random_state=seed, n_jobs=-1, verbosity=-1)


def _oof_multilabel(factory, X, Y, folds, verbose=False, tag="", clip=True):
    oof = np.zeros_like(Y, dtype=np.float32)
    for i, (tr, va) in enumerate(folds, 1):
        for j in range(Y.shape[1]):
            m = factory()
            m.fit(X[tr], Y[tr, j])
            oof[va, j] = _maybe_clip(m.predict(X[va]), clip)
        if verbose:
            print(f"  {tag} oof: fold {i}/{len(folds)} done", flush=True)
    return oof


def _full_multilabel(factory, X, Y, X_test, verbose=False, tag="", clip=True):
    test = np.zeros((X_test.shape[0], Y.shape[1]), dtype=np.float32)
    for j in range(Y.shape[1]):
        m = factory()
        m.fit(X, Y[:, j])
        test[:, j] = _maybe_clip(m.predict(X_test), clip)
    if verbose:
        print(f"  {tag} full-fit done ({Y.shape[1]} models)", flush=True)
    return test


def lgbm_oof(X, Y, folds, n_estimators=1200, lr=0.03, leaves=31, min_child=30, seed=42,
             verbose=False, clip=True):
    return _oof_multilabel(_lgbm_factory(n_estimators, lr, leaves, min_child, seed),
                           X, Y, folds, verbose, "lgbm", clip)


def lgbm_full(X, Y, X_test, n_estimators=1200, lr=0.03, leaves=31, min_child=30, seed=42,
              verbose=False, clip=True):
    return _full_multilabel(_lgbm_factory(n_estimators, lr, leaves, min_child, seed),
                            X, Y, X_test, verbose, "lgbm", clip)


def _ridge_factory(alpha=8.0, seed=42):
    return lambda: Ridge(alpha=alpha, random_state=seed)


def linear_oof(X, Y, folds, seed=42, alpha=8.0, verbose=False, clip=True):
    return _oof_multilabel(_ridge_factory(alpha, seed), X, Y, folds, verbose, "linear", clip)


def linear_full(X, Y, X_test, seed=42, alpha=8.0, verbose=False, clip=True):
    return _full_multilabel(_ridge_factory(alpha, seed), X, Y, X_test, verbose, "linear", clip)
