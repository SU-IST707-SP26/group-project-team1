"""Derived analytics computed once after models are trained.

Everything in here is deterministic given the trained models + the prescriptive
frame, so it goes into the same cached bundle.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from constants import TEST_SEASON, DYNAMIC_ERA_FIRST_SEASON
from data_loader import make_situation_row


def assign_playcaller(df: pd.DataFrame, pc_cols: list[str]) -> pd.Series:
    """Recover the playcaller name from one-hot columns.

    Each row should have at most one playcaller_* flag set to 1. If none are
    set, return 'Unknown'. Vectorized via idxmax on the playcaller subset.
    """
    if not pc_cols:
        return pd.Series(["Unknown"] * len(df), index=df.index)

    pc_block = df[pc_cols]
    has_pc = pc_block.sum(axis=1) > 0
    names = pc_block.idxmax(axis=1).str.replace("playcaller_", "", regex=False)
    names[~has_pc] = "Unknown"
    return names


def compute_playcaller_stats(df_pres: pd.DataFrame, pres_features: list[str],
                             pres_model, pred_model, le_pres, le_pred,
                             X_pres: pd.DataFrame, pc_cols: list[str],
                             min_plays: int = 20) -> tuple[pd.DataFrame, float]:
    """For each playcaller: % of plays where their actual call agrees with the
    EPA-optimal call, and % where the predictive model's predicted call agrees.

    The 'predicted agreement' tells us how often the model thinks each coach
    will make the optimal call — useful for spotting coaches the model
    misunderstands.

    Args:
      df_pres: prescriptive dataframe (must contain optimal_decision, decision,
               and playcaller_* columns)
      X_pres: feature matrix used to fit the prescriptive model — we reuse it
              to score with the predictive model since both use the same features
      le_pres, le_pred: label encoders
      min_plays: drop playcallers with fewer than this many 4th-down plays

    Returns: (per-playcaller stats DataFrame, league agreement rate scalar)
    """
    # Score every play through the predictive model
    df = df_pres.loc[X_pres.index].copy()
    df["predicted_decision"] = le_pred.inverse_transform(pred_model.predict(X_pres))

    df["actual_matches_optimal"]    = (df["decision"]           == df["optimal_decision"]).astype(int)
    df["predicted_matches_optimal"] = (df["predicted_decision"] == df["optimal_decision"]).astype(int)
    df["playcaller"] = assign_playcaller(df, pc_cols)

    league_agree = df["actual_matches_optimal"].mean()

    stats = (
        df.groupby("playcaller")
          .agg(n_plays              =("decision",                 "count"),
               actual_agree_rate    =("actual_matches_optimal",   "mean"),
               predicted_agree_rate =("predicted_matches_optimal","mean"))
          .query(f"n_plays >= {min_plays}")
          .sort_values("actual_agree_rate", ascending=False)
          .round(3)
    )

    return stats, df, league_agree


def compute_audit_cases(df_scored: pd.DataFrame) -> pd.DataFrame:
    """The "model predicts punt inside opponent's 40" cases from the
    predictive notebook. Flags suspicious model predictions — when the
    coach-behavior model says a coach will punt deep in opponent territory,
    is the model right or wrong?

    Returns one row per such case, with the situation context and whether
    the predicted punt actually happened.
    """
    mask = (
        (df_scored["predicted_decision"] == "punt")
        & (df_scored["yardline_100"] <= 40)
    )
    cases = df_scored[mask].copy()
    cases["model_was_right"] = (
        cases["predicted_decision"] == cases["decision"]
    ).astype(int)
    return cases[[
        "season", "playcaller", "yardline_100", "ydstogo",
        "score_differential", "qtr", "game_seconds_remaining",
        "decision", "predicted_decision", "optimal_decision",
        "model_was_right",
    ]].sort_values(["season", "yardline_100"])


def compute_yearly_agreement(df_scored: pd.DataFrame) -> pd.DataFrame:
    """Agreement rate by season. Powers the kickoff-rule narrative — does
    coach behavior shift across the 2023 boundary?
    """
    return (
        df_scored.groupby("season")
                 .agg(actual_agree_rate    =("actual_matches_optimal",    "mean"),
                      predicted_agree_rate =("predicted_matches_optimal", "mean"),
                      n_plays              =("decision",                  "count"))
                 .round(3)
    )


def compute_heatmap_grid(features: list[str], pres_model, le_pres, *,
                         season: int = 2024,
                         score_diff: int = 0, qtr: int = 3,
                         game_secs: int = 1800, half_secs: int = 900,
                         pos_to: int = 2, def_to: int = 2,
                         shotgun: int = 0, no_huddle: int = 0,
                         home: int = 0) -> pd.DataFrame:
    """Build a yardline x ydstogo grid of EPA-optimal recommendations.

    Phase 2 change vs original app.py: this now accepts the contextual params
    instead of hardcoding them, so the heatmap can respond to sidebar inputs.
    """
    coords, rows = [], []
    for yl in range(5, 100, 5):
        for dist in [1, 2, 3, 4, 5, 6, 7, 8, 10, 15]:
            row = make_situation_row(
                features,
                yardline=yl, ydstogo=dist, score_diff=score_diff, qtr=qtr,
                game_secs=game_secs, half_secs=half_secs,
                pos_to=pos_to, def_to=def_to,
                goal_to_go=(yl <= dist), shotgun=bool(shotgun),
                no_huddle=bool(no_huddle), home=bool(home),
                season=season,
            ).iloc[0]
            rows.append(row)
            coords.append((yl, dist))

    X = pd.DataFrame(rows)
    labels = le_pres.inverse_transform(pres_model.predict(X))
    grid = pd.DataFrame(coords, columns=["yardline_100", "ydstogo"])
    grid["optimal"] = labels
    return grid


def predict_one(features: list[str], pres_model, pred_model, le_pres, le_pred,
                **situation) -> dict:
    """Run both models on a single situation, return labels + probabilities."""
    X = make_situation_row(features, **situation)

    pres_proba = dict(zip(le_pres.classes_, pres_model.predict_proba(X)[0]))
    pred_proba = dict(zip(le_pred.classes_, pred_model.predict_proba(X)[0]))
    return {
        "pres_proba": pres_proba,
        "pred_proba": pred_proba,
        "optimal":    max(pres_proba, key=pres_proba.get),
        "predicted":  max(pred_proba, key=pred_proba.get),
    }
