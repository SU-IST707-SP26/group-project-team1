"""Train, cache, and load both XGBoost models.

The first run trains and pickles to disk (~30-60s).
Subsequent runs load from disk in <1s.

Cache is invalidated by deleting the pickle file. We also write a small
sidecar metadata file so we can detect param drift (e.g. you tweaked the
JSON but forgot to clear the cache).
"""
from __future__ import annotations
import hashlib
import json
import pickle
import time
from pathlib import Path

import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from constants import (
    PRES_PARAMS_PATH, PRED_PARAMS_PATH, PRES_FALLBACK, PRED_FALLBACK,
    CACHE_DIR, CACHE_PATH, TEST_SEASON, RANDOM_STATE,
)


def load_params(path: Path, fallback: dict) -> tuple[dict, bool]:
    """Returns (params, loaded_from_file_flag)."""
    if path.exists():
        with open(path) as f:
            return json.load(f), True
    return fallback, False


def _params_hash(*param_dicts: dict) -> str:
    """Stable hash of all params — used to detect when JSON changes."""
    blob = json.dumps([dict(sorted(p.items())) for p in param_dicts], sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


def _train_xgb(X_train, y_train_enc, X_test, y_test_enc, params: dict, label: str):
    """Fit one classifier, print test accuracy."""
    t0 = time.time()
    model = xgb.XGBClassifier(
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        **params,
    )
    model.fit(X_train, y_train_enc)
    acc = accuracy_score(y_test_enc, model.predict(X_test))
    print(f"  {label}: trained in {time.time()-t0:.1f}s, test acc = {acc:.4f}")
    return model, acc


def train_and_cache(frames: dict) -> dict:
    """Train both models on the prepared frames. Caches result to disk.

    Args:
      frames: output of data_loader.prepare_modeling_frames()

    Returns a dict containing both fitted models, label encoders, accuracies,
    params, and the params-source flag.
    """
    pres_params, pres_from_file = load_params(PRES_PARAMS_PATH, PRES_FALLBACK)
    pred_params, pred_from_file = load_params(PRED_PARAMS_PATH, PRED_FALLBACK)
    h = _params_hash(pres_params, pred_params)

    # Try cache first
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "rb") as f:
                cached = pickle.load(f)
            if cached.get("params_hash") == h:
                print(f"Loaded cached models from {CACHE_PATH.name} (hash {h})")
                return cached
            else:
                print("Param hash mismatch — retraining.")
        except Exception as e:
            print(f"Cache load failed ({e}) — retraining.")

    print("Training prescriptive model…")
    le_pres = LabelEncoder()
    y_pres_enc = le_pres.fit_transform(frames["y_pres_raw"])
    season_p = frames["season_pres"]
    tr_p = season_p[season_p < TEST_SEASON].index
    te_p = season_p[season_p == TEST_SEASON].index
    X_p = frames["X_pres"]
    pres_model, pres_acc = _train_xgb(
        X_p.loc[tr_p], y_pres_enc[X_p.index.get_indexer(tr_p)],
        X_p.loc[te_p], y_pres_enc[X_p.index.get_indexer(te_p)],
        pres_params, "Prescriptive",
    )

    print("Training predictive model…")
    le_pred = LabelEncoder()
    y_pred_enc = le_pred.fit_transform(frames["y_pred_raw"])
    season_d = frames["season_pred"]
    tr_d = season_d[season_d < TEST_SEASON].index
    te_d = season_d[season_d == TEST_SEASON].index
    X_d = frames["X_pred"]
    pred_model, pred_acc = _train_xgb(
        X_d.loc[tr_d], y_pred_enc[X_d.index.get_indexer(tr_d)],
        X_d.loc[te_d], y_pred_enc[X_d.index.get_indexer(te_d)],
        pred_params, "Predictive",
    )

    bundle = {
        "pres_model":       pres_model,
        "pred_model":       pred_model,
        "le_pres":          le_pres,
        "le_pred":          le_pred,
        "pres_acc":         pres_acc,
        "pred_acc":         pred_acc,
        "pres_params":      pres_params,
        "pred_params":      pred_params,
        "params_from_file": pres_from_file and pred_from_file,
        "params_hash":      h,
    }

    CACHE_DIR.mkdir(exist_ok=True, parents=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Cached models to {CACHE_PATH.name}")

    return bundle
