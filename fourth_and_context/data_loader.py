"""Data loading and feature engineering for 4th & Context.

Two top-level entry points:
- load_clean_plays()  -> cleaned dataframe with 'decision' column
- prepare_modeling_frames(df_clean) -> everything needed to train both models

Keeping these as pure functions (no module-level state) so they can be
unit-tested and reused outside the Shiny app (e.g. in notebooks).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from constants import (
    DATA_PATH, PRE_SNAP_FEATURES, PREGAME_EPA_FEATURES, GROUP_COLS,
    YDSTOGO_BINS, YDSTOGO_LABELS, YARDLINE_BINS, YARDLINE_LABELS,
    SCORE_DIFF_BINS, SCORE_DIFF_LABELS, DYNAMIC_ERA_FIRST_SEASON,
)


def _decide_vectorized(df: pd.DataFrame) -> pd.Series:
    """Reconstruct decision label from one-hot encoded play type columns.

    Vectorized — no per-row apply. ~50x faster on 38k rows.
    """
    decision = pd.Series([np.nan] * len(df), index=df.index, dtype=object)
    is_run_pass = (df.get("play_type_pass", 0) == 1) | (df.get("play_type_run", 0) == 1)
    is_punt     = df.get("play_type_punt", 0) == 1
    is_fg       = df.get("field_goal_attempt", 0) == 1
    decision[is_run_pass] = "go"
    decision[is_punt]     = "punt"
    decision[is_fg]       = "field_goal"
    return decision


def load_clean_plays() -> pd.DataFrame:
    """Load CSV, reconstruct decision label, drop rows without a clear call.

    Returns a dataframe with all original columns plus a 'decision' column
    in {'go', 'punt', 'field_goal'}.
    """
    df = pd.read_csv(DATA_PATH, compression="gzip", low_memory=False)
    df = df.copy()  # defragment after read

    # Some encoded variants of the dataset have a literal play_type column,
    # most don't. Handle both.
    if "play_type" in df.columns:
        play_type_map = {"run": "go", "pass": "go",
                         "punt": "punt", "field_goal": "field_goal"}
        df["decision"] = df["play_type"].map(play_type_map)
    else:
        df["decision"] = _decide_vectorized(df)

    df = df.dropna(subset=["decision"]).copy()
    df = df[df["decision"].isin(["go", "punt", "field_goal"])].reset_index(drop=True)
    return df


def add_situation_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Add the four bucket columns used to compute the prescriptive optimum."""
    df = df.copy()
    df["ydstogo_bin"]    = pd.cut(df["ydstogo"],
                                  bins=YDSTOGO_BINS, labels=YDSTOGO_LABELS)
    df["yardline_bin"]   = pd.cut(df["yardline_100"],
                                  bins=YARDLINE_BINS, labels=YARDLINE_LABELS)
    df["score_diff_bin"] = pd.cut(df["score_differential"],
                                  bins=SCORE_DIFF_BINS, labels=SCORE_DIFF_LABELS)
    df["kickoff_era"] = df["season"].apply(
        lambda s: "dynamic" if s >= DYNAMIC_ERA_FIRST_SEASON else "traditional")
    return df


def compute_optimal_decisions(df: pd.DataFrame) -> pd.DataFrame:
    """Within each situation bucket, find the decision with the highest mean EPA.

    Returns a pivot table with columns: GROUP_COLS + go + punt + field_goal +
    optimal_decision. Buckets that have no observations of a decision get NaN
    for that decision; idxmax handles NaN by ignoring it.
    """
    pivot = (
        df.groupby(GROUP_COLS + ["decision"], observed=False)["epa"]
          .mean()
          .reset_index()
          .rename(columns={"epa": "mean_epa"})
          .pivot_table(index=GROUP_COLS, columns="decision",
                       values="mean_epa", observed=False)
          .reset_index()
    )
    pivot.columns.name = None
    for col in ["go", "punt", "field_goal"]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot["optimal_decision"] = pivot[["go", "punt", "field_goal"]].idxmax(axis=1)
    return pivot


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Add era flag + season dummies and return the feature list.

    The returned feature list is the same regardless of whether the dataframe
    is the prescriptive subset or the full clean set — matters because both
    models share the same input space.
    """
    df = df.copy()
    df["is_dynamic_era"] = (df["season"] >= DYNAMIC_ERA_FIRST_SEASON).astype(int)

    season_dummies = pd.get_dummies(df["season"], prefix="season", drop_first=True)
    df = pd.concat([df, season_dummies], axis=1)

    pc_cols = [c for c in df.columns if c.startswith("playcaller_")]
    feats = (PRE_SNAP_FEATURES + PREGAME_EPA_FEATURES + pc_cols
             + ["is_dynamic_era"] + list(season_dummies.columns))
    feats = [f for f in feats if f in df.columns]

    return df, feats, pc_cols


def build_xy(df: pd.DataFrame, features: list[str], target_col: str):
    """Build X (with bool->int conversion) and y, dropping any row with NaN features.

    Returns (X, y, season_series) all aligned by index.
    """
    X = df[features].copy()
    bool_cols = X.select_dtypes("bool").columns
    if len(bool_cols):
        X[bool_cols] = X[bool_cols].astype(int)

    mask = X.notna().all(axis=1)
    X = X[mask]
    y = df.loc[mask, target_col]
    season = df.loc[mask, "season"]
    return X, y, season


def prepare_modeling_frames(df_clean: pd.DataFrame) -> dict:
    """Build everything both models need to train, in one pass.

    Returns a dict with:
      X_pres, y_pres_raw, season_pres : prescriptive training frame
      X_pred, y_pred_raw, season_pred : predictive training frame
      features                        : the shared feature list
      playcaller_cols                 : list of playcaller_* column names
      epa_pivot                       : DataFrame of bucket -> optimal decision
      df_pres                         : full prescriptive frame (for analytics)
    """
    # Prescriptive: only rows with EPA, get bucket-mean optimal label
    df_pres = df_clean.dropna(subset=["epa"]).copy()
    df_pres = add_situation_bins(df_pres)
    epa_pivot = compute_optimal_decisions(df_pres)
    df_pres = df_pres.merge(
        epa_pivot[GROUP_COLS + ["optimal_decision"]],
        on=GROUP_COLS, how="left",
    )
    df_pres = df_pres.dropna(subset=["optimal_decision"])

    # Predictive: all clean rows, target is the actual decision
    df_pred = df_clean.copy()

    # Engineer shared features on each frame independently. The feature list
    # they produce should be identical (same playcaller cols, same seasons).
    df_pres, feats_pres, pc_cols = engineer_features(df_pres)
    df_pred, feats_pred, _       = engineer_features(df_pred)

    # Sanity check — if these ever drift, that's a real bug we want to surface.
    assert feats_pres == feats_pred, (
        "Prescriptive and predictive feature lists drifted. "
        f"pres has {len(feats_pres)}, pred has {len(feats_pred)}."
    )
    features = feats_pres

    X_pres, y_pres, season_pres = build_xy(df_pres, features, "optimal_decision")
    X_pred, y_pred, season_pred = build_xy(df_pred, features, "decision")

    return {
        "X_pres":          X_pres,
        "y_pres_raw":      y_pres,
        "season_pres":     season_pres,
        "X_pred":          X_pred,
        "y_pred_raw":      y_pred,
        "season_pred":     season_pred,
        "features":        features,
        "playcaller_cols": pc_cols,
        "epa_pivot":       epa_pivot,
        "df_pres":         df_pres,
    }


def make_situation_row(features: list[str],
                       *, yardline: int, ydstogo: int, score_diff: int,
                       qtr: int, game_secs: int, half_secs: int,
                       pos_to: int, def_to: int,
                       goal_to_go: bool, shotgun: bool, no_huddle: bool,
                       home: bool, season: int = 2024,
                       playcaller: str | None = None) -> pd.DataFrame:
    """Build a single-row feature DataFrame for prediction.

    Uses league-average pregame EPA values as defaults — these are
    pre-snap probabilities the model expects but the user shouldn't have
    to enter manually.
    """
    base = dict.fromkeys(features, 0.0)
    base.update({
        "yardline_100":               yardline,
        "ydstogo":                    ydstogo,
        "score_differential":         score_diff,
        "qtr":                        qtr,
        "game_seconds_remaining":     game_secs,
        "half_seconds_remaining":     half_secs,
        "posteam_timeouts_remaining": pos_to,
        "defteam_timeouts_remaining": def_to,
        "goal_to_go":                 int(goal_to_go),
        "shotgun":                    int(shotgun),
        "no_huddle":                  int(no_huddle),
        "home_is_posteam":            int(home),
        "is_dynamic_era":             int(season >= DYNAMIC_ERA_FIRST_SEASON),
        # League-average pregame probabilities
        "no_score_prob":              0.17,
        "opp_fg_prob":                0.05,
        "opp_td_prob":                0.19,
        "fg_prob":                    0.06,
        "td_prob":                    0.22,
    })
    season_key = f"season_{season}"
    if season_key in base:
        base[season_key] = 1
    # Set the playcaller one-hot. If unknown/None, leave all playcaller_*
    # columns at 0 — model behavior in that case is undefined and the UI
    # should warn the user.
    if playcaller:
        pc_key = f"playcaller_{playcaller}"
        if pc_key in base:
            base[pc_key] = 1
    return pd.DataFrame([base])
