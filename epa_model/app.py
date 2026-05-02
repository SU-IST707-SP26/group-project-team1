# =============================================================================
# 4th & Context — Decision Support Dashboard
# IST 707 | Gullo, Weber, Stein
#
# Requirements:
#   pip install shiny xgboost scikit-learn pandas numpy plotly
#
# Usage:
#   shiny run app.py
#
# Model params:
#   Place best_xgb_params.json (prescriptive) and
#   best_predictive_xgb_params.json (predictive) in the same directory.
#   If not present, hardcoded best params from your notebook runs are used.
#
# Data:
#   Place encoded_fourth_downs.csv.gz one directory up (../encoded_fourth_downs.csv.gz)
#   or update DATA_PATH below.
# =============================================================================

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb
from shiny import App, reactive, render, ui
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from scipy.stats import randint, uniform, loguniform

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
DATA_PATH = HERE.parent / "encoded_fourth_downs.csv.gz"
PRES_PARAMS_PATH = HERE / "best_xgb_params.json"
PRED_PARAMS_PATH = HERE / "best_predictive_xgb_params.json"

# ---------------------------------------------------------------------------
# Fallback params (from your notebook tuning runs)
# ---------------------------------------------------------------------------
PRES_FALLBACK_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.08,
    "subsample": 0.85,
    "colsample_bytree": 0.80,
    "min_child_weight": 5,
    "gamma": 1.2,
    "reg_alpha": 0.05,
    "reg_lambda": 1.5,
}
PRED_FALLBACK_PARAMS = {
    "n_estimators": 350,
    "max_depth": 7,
    "learning_rate": 0.10,
    "subsample": 0.80,
    "colsample_bytree": 0.75,
    "min_child_weight": 3,
    "gamma": 0.8,
    "reg_alpha": 0.02,
    "reg_lambda": 1.0,
}

# ---------------------------------------------------------------------------
# Feature lists (must match your notebook exactly)
# ---------------------------------------------------------------------------
PRE_SNAP_FEATURES = [
    "yardline_100", "ydstogo", "score_differential",
    "game_seconds_remaining", "half_seconds_remaining", "qtr",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "goal_to_go", "shotgun", "no_huddle", "home_is_posteam",
]
PREGAME_EPA_FEATURES = [
    "no_score_prob", "opp_fg_prob", "opp_td_prob", "fg_prob", "td_prob",
]
GROUP_COLS = ["ydstogo_bin", "yardline_bin", "score_diff_bin", "kickoff_era"]
TEST_SEASON = 2024

# ---------------------------------------------------------------------------
# Data + model loading (cached so it only runs once)
# ---------------------------------------------------------------------------
_cache: dict = {}


def load_params(path: Path, fallback: dict) -> dict:
    if path.exists():
        with open(path) as f:
            params = json.load(f)
        print(f"Loaded params from {path.name}")
        return params
    print(f"{path.name} not found — using fallback params")
    return fallback


def build_features(df: pd.DataFrame):
    """Replicates the feature engineering from both notebooks."""
    playcaller_cols = [c for c in df.columns if c.startswith("playcaller_")]
    df = df.copy()
    df["is_dynamic_era"] = (df["season"] >= 2023).astype(int)
    season_dummies = pd.get_dummies(df["season"], prefix="season", drop_first=True)
    df = pd.concat([df, season_dummies], axis=1)
    season_feat_cols = list(season_dummies.columns)

    all_features = (
        PRE_SNAP_FEATURES + PREGAME_EPA_FEATURES
        + playcaller_cols + ["is_dynamic_era"] + season_feat_cols
    )
    all_features = [f for f in all_features if f in df.columns]
    return df, all_features, playcaller_cols


def build_prescriptive_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket-mean EPA → optimal decision label (mirrors prescriptive notebook)."""
    df = df.copy()
    df["ydstogo_bin"] = pd.cut(df["ydstogo"], bins=[0, 1, 3, 6, 10, 99],
                                labels=["1", "2-3", "4-6", "7-10", "10+"])
    df["yardline_bin"] = pd.cut(df["yardline_100"], bins=[0, 20, 40, 60, 80, 100],
                                 labels=["opp_red_zone", "opp_40", "midfield", "own_40", "own_end"])
    df["score_diff_bin"] = pd.cut(df["score_differential"],
                                   bins=[-100, -14, -7, -3, 3, 7, 14, 100],
                                   labels=["down_14+", "down_8-14", "down_1-7",
                                           "close", "up_1-7", "up_8-14", "up_14+"])
    df["kickoff_era"] = df["season"].apply(lambda s: "dynamic" if s >= 2023 else "traditional")

    epa_pivot = (
        df.groupby(GROUP_COLS + ["decision"])["epa"]
        .mean().reset_index().rename(columns={"epa": "mean_epa"})
        .pivot_table(index=GROUP_COLS, columns="decision", values="mean_epa")
        .reset_index()
    )
    epa_pivot.columns.name = None
    for col in ["go", "punt", "field_goal"]:
        if col not in epa_pivot.columns:
            epa_pivot[col] = np.nan
    epa_pivot["optimal_decision"] = epa_pivot[["go", "punt", "field_goal"]].idxmax(axis=1)

    df = df.merge(epa_pivot[GROUP_COLS + ["optimal_decision"]], on=GROUP_COLS, how="left")
    df = df.dropna(subset=["optimal_decision"])
    return df, epa_pivot


def get_models():
    if "models" in _cache:
        return _cache["models"]

    if not DATA_PATH.exists():
        _cache["models"] = None
        return None

    print("Loading data…")
    raw = pd.read_csv(DATA_PATH, compression="gzip", low_memory=False)

    # Reconstruct decision column
    play_type_map = {"run": "go", "pass": "go", "punt": "punt", "field_goal": "field_goal"}
    if "play_type" in raw.columns:
        raw["decision"] = raw["play_type"].map(play_type_map)
    else:
        def get_decision(row):
            if row.get("play_type_pass", 0) == 1 or row.get("play_type_run", 0) == 1:
                return "go"
            elif row.get("play_type_punt", 0) == 1:
                return "punt"
            elif row.get("field_goal_attempt", 0) == 1:
                return "field_goal"
            return np.nan
        raw["decision"] = raw.apply(get_decision, axis=1)

    df_clean = raw.dropna(subset=["decision"]).copy()
    df_clean = df_clean[df_clean["decision"].isin(["go", "punt", "field_goal"])]

    # ---- Prescriptive model ----
    df_pres = df_clean.dropna(subset=["epa"]).copy()
    df_pres, epa_pivot = build_prescriptive_labels(df_pres)
    df_pres, all_features, playcaller_cols = build_features(df_pres)

    X_pres = df_pres[all_features].copy()
    y_pres = df_pres["optimal_decision"].copy()
    X_pres[X_pres.select_dtypes("bool").columns] = X_pres.select_dtypes("bool").astype(int)
    mask_pres = X_pres.notna().all(axis=1)
    X_pres, y_pres = X_pres[mask_pres], y_pres[mask_pres]
    season_col_pres = df_pres.loc[mask_pres, "season"]

    train_idx_p = season_col_pres[season_col_pres < TEST_SEASON].index
    test_idx_p  = season_col_pres[season_col_pres == TEST_SEASON].index
    X_train_p, X_test_p = X_pres.loc[train_idx_p], X_pres.loc[test_idx_p]
    y_train_p, y_test_p = y_pres.loc[train_idx_p], y_pres.loc[test_idx_p]

    le_pres = LabelEncoder()
    y_train_penc = le_pres.fit_transform(y_train_p)
    y_test_penc  = le_pres.transform(y_test_p)

    pres_params = load_params(PRES_PARAMS_PATH, PRES_FALLBACK_PARAMS)
    pres_model = xgb.XGBClassifier(
        eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0, **pres_params
    )
    pres_model.fit(X_train_p, y_train_penc)
    pres_acc = accuracy_score(y_test_penc, pres_model.predict(X_test_p))
    print(f"Prescriptive model test accuracy: {pres_acc:.4f}")

    # ---- Predictive model ----
    df_pred = df_clean.copy()
    df_pred, all_features_pred, _ = build_features(df_pred)

    X_pred = df_pred[all_features_pred].copy()
    y_pred = df_pred["decision"].copy()
    X_pred[X_pred.select_dtypes("bool").columns] = X_pred.select_dtypes("bool").astype(int)
    mask_pred = X_pred.notna().all(axis=1)
    X_pred, y_pred = X_pred[mask_pred], y_pred[mask_pred]
    season_col_pred = df_pred.loc[mask_pred, "season"]

    train_idx_d = season_col_pred[season_col_pred < TEST_SEASON].index
    test_idx_d  = season_col_pred[season_col_pred == TEST_SEASON].index
    X_train_d, X_test_d = X_pred.loc[train_idx_d], X_pred.loc[test_idx_d]
    y_train_d, y_test_d = y_pred.loc[train_idx_d], y_pred.loc[test_idx_d]

    le_pred = LabelEncoder()
    y_train_denc = le_pred.fit_transform(y_train_d)
    y_test_denc  = le_pred.transform(y_test_d)

    pred_params = load_params(PRED_PARAMS_PATH, PRED_FALLBACK_PARAMS)
    pred_model = xgb.XGBClassifier(
        eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0, **pred_params
    )
    pred_model.fit(X_train_d, y_train_denc)
    pred_acc = accuracy_score(y_test_denc, pred_model.predict(X_test_d))
    print(f"Predictive model test accuracy: {pred_acc:.4f}")

    # ---- Playcaller agreement stats ----
    df_scored = df_pres[mask_pres].copy()
    df_scored["predicted_decision"] = le_pred.inverse_transform(
        pred_model.predict(X_pres[mask_pres])
    )
    pc_cols_present = [c for c in playcaller_cols if c in df_scored.columns]
    df_scored["playcaller"] = df_scored[pc_cols_present].apply(
        lambda row: next(
            (col.replace("playcaller_", "") for col in pc_cols_present if row[col] == 1),
            "Unknown"
        ), axis=1
    )
    df_scored["actual_matches_optimal"]    = (df_scored["decision"]           == df_scored["optimal_decision"]).astype(int)
    df_scored["predicted_matches_optimal"] = (df_scored["predicted_decision"] == df_scored["optimal_decision"]).astype(int)

    playcaller_stats = (
        df_scored.groupby("playcaller")
        .agg(n_plays=("decision", "count"),
             actual_agree_rate=("actual_matches_optimal", "mean"),
             predicted_agree_rate=("predicted_matches_optimal", "mean"))
        .query("n_plays >= 20")
        .sort_values("actual_agree_rate", ascending=False)
        .round(3)
    )

    _cache["models"] = {
        "pres_model": pres_model, "le_pres": le_pres,
        "pred_model": pred_model, "le_pred": le_pred,
        "all_features_pres": all_features, "all_features_pred": all_features_pred,
        "epa_pivot": epa_pivot, "playcaller_stats": playcaller_stats,
        "playcaller_cols": playcaller_cols,
        "pres_acc": pres_acc, "pred_acc": pred_acc,
        "league_agree": df_scored["actual_matches_optimal"].mean(),
        "n_divergent": int((1 - df_scored.loc[season_col_pres == TEST_SEASON, "actual_matches_optimal"]).sum()),
    }
    return _cache["models"]


# ---------------------------------------------------------------------------
# Helper: build a single-row feature vector for the models
# ---------------------------------------------------------------------------
def build_single_row(models, yardline, ydstogo, score_diff, qtr, game_secs,
                     half_secs, pos_to, def_to, goal_to_go, shotgun,
                     no_huddle, home, season=2024, playcaller=None):
    base = {f: 0.0 for f in models["all_features_pres"]}
    # Situational
    base["yardline_100"]              = yardline
    base["ydstogo"]                   = ydstogo
    base["score_differential"]        = score_diff
    base["qtr"]                       = qtr
    base["game_seconds_remaining"]    = game_secs
    base["half_seconds_remaining"]    = half_secs
    base["posteam_timeouts_remaining"]= pos_to
    base["defteam_timeouts_remaining"]= def_to
    base["goal_to_go"]                = int(goal_to_go)
    base["shotgun"]                   = int(shotgun)
    base["no_huddle"]                 = int(no_huddle)
    base["home_is_posteam"]           = int(home)
    base["is_dynamic_era"]            = int(season >= 2023)

    # Pregame EPA defaults (league average mid-game values)
    base["no_score_prob"]  = 0.17
    base["opp_fg_prob"]    = 0.05
    base["opp_td_prob"]    = 0.19
    base["fg_prob"]        = 0.06
    base["td_prob"]        = 0.22

    # Season dummy
    season_key = f"season_{season}"
    if season_key in base:
        base[season_key] = 1

    # Playcaller dummy
    if playcaller:
        pc_key = f"playcaller_{playcaller}"
        if pc_key in base:
            base[pc_key] = 1

    return pd.DataFrame([base])


# ---------------------------------------------------------------------------
# Decision heatmap grid (cached per era)
# ---------------------------------------------------------------------------
def get_heatmap_data(models, era_season=2024):
    cache_key = f"heatmap_{era_season}"
    if cache_key in _cache:
        return _cache[cache_key]

    yardlines = list(range(5, 100, 5))
    distances  = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15]
    rows = []
    for yl in yardlines:
        for dist in distances:
            row_df = build_single_row(
                models, yl, dist, 0, 3, 1800, 900, 2, 2,
                int(yl <= dist), 0, 0, 0, season=era_season
            )
            rows.append({"yardline_100": yl, "ydstogo": dist, "row": row_df})

    pres_features = models["all_features_pres"]
    preds = []
    for r in rows:
        x = r["row"][pres_features].fillna(0)
        enc = models["pres_model"].predict(x)[0]
        preds.append(models["le_pres"].inverse_transform([enc])[0])

    result = pd.DataFrame([
        {"yardline_100": r["yardline_100"], "ydstogo": r["ydstogo"], "optimal": p}
        for r, p in zip(rows, preds)
    ])
    _cache[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
DECISION_COLORS = {"go": "#639922", "punt": "#185FA5", "field_goal": "#BA7517"}

app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style("""
            body { background: #f8f9fa; font-family: 'Segoe UI', system-ui, sans-serif; }
            .card { background: white; border-radius: 10px; border: 1px solid #e5e7eb;
                    padding: 1.25rem; margin-bottom: 1rem; }
            .card-title { font-size: 11px; font-weight: 600; text-transform: uppercase;
                          letter-spacing: 0.07em; color: #6b7280; margin-bottom: .75rem; }
            .badge { display: inline-block; padding: 5px 14px; border-radius: 6px;
                     font-size: 13px; font-weight: 600; margin-right: 6px; }
            .badge-go   { background: #EAF3DE; color: #3B6D11; border: 1.5px solid #639922; }
            .badge-punt { background: #E6F1FB; color: #0C447C; border: 1.5px solid #378ADD; }
            .badge-fg   { background: #FAEEDA; color: #854F0B; border: 1.5px solid #BA7517; }
            .metric-card { background: #f3f4f6; border-radius: 8px; padding: .9rem 1rem;
                           text-align: center; }
            .metric-val  { font-size: 26px; font-weight: 600; line-height: 1.1; }
            .metric-lbl  { font-size: 11px; color: #6b7280; margin-top: 4px; }
            .alert-green { background: #f0fdf4; border: 1px solid #86efac;
                           border-radius: 8px; padding: .75rem 1rem; color: #15803d; font-size: 14px; }
            .alert-amber { background: #fffbeb; border: 1px solid #fcd34d;
                           border-radius: 8px; padding: .75rem 1rem; color: #92400e; font-size: 14px; }
            .section-header { font-size: 20px; font-weight: 600; margin-bottom: .25rem; }
            .muted { color: #6b7280; font-size: 13px; }
        """)
    ),

    ui.div(
        ui.h2("4th & Context", class_="section-header"),
        ui.p("NFL 4th Down Decision Support Dashboard — IST 707", class_="muted"),
        style="padding: 1.5rem 1.5rem 0.5rem;"
    ),

    ui.navset_tab(
        # ---- Tab 1: Situation Evaluator ----
        ui.nav_panel("Situation Evaluator",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h6("Game Situation", style="font-weight:600; margin-bottom:.75rem;"),

                    ui.input_slider("yardline", "Yard line (from opp. end zone)", 1, 99, 55, step=1),
                    ui.input_slider("ydstogo",  "Yards to go", 1, 20, 5, step=1),
                    ui.input_slider("score_diff", "Score differential", -28, 28, 0, step=1),
                    ui.input_select("qtr", "Quarter", {"1": "Q1", "2": "Q2", "3": "Q3", "4": "Q4", "5": "OT"}),
                    ui.input_slider("game_secs", "Game seconds remaining", 0, 3600, 1800, step=30),

                    ui.hr(),
                    ui.h6("Context", style="font-weight:600; margin-bottom:.75rem;"),
                    ui.input_slider("pos_to", "Possession team timeouts", 0, 3, 2),
                    ui.input_slider("def_to", "Defense timeouts", 0, 3, 2),
                    ui.input_checkbox("goal_to_go", "Goal to go", False),
                    ui.input_checkbox("shotgun", "Shotgun formation", False),
                    ui.input_checkbox("no_huddle", "No huddle", False),
                    ui.input_checkbox("home", "Home team has ball", True),
                    ui.input_select("season", "Season / Era",
                                   {str(y): str(y) for y in range(2018, 2026)},
                                   selected="2024"),

                    width=300,
                ),
                ui.output_ui("situation_output"),
            )
        ),

        # ---- Tab 2: Decision Heatmap ----
        ui.nav_panel("Decision Heatmap",
            ui.div(
                ui.div(
                    ui.input_select("heatmap_era", "Kickoff era",
                                    {"2024": "Dynamic (2023+)", "2019": "Traditional (pre-2023)"}),
                    style="max-width:240px; margin-bottom:1rem;"
                ),
                ui.output_plot("heatmap_plot", height="480px"),
                style="padding:1rem;"
            )
        ),

        # ---- Tab 3: Playcaller Report ----
        ui.nav_panel("Playcaller Report",
            ui.div(
                ui.output_ui("playcaller_output"),
                style="padding:1rem;"
            )
        ),

        # ---- Tab 4: Model Summary ----
        ui.nav_panel("Model Summary",
            ui.div(
                ui.output_ui("model_summary_output"),
                style="padding:1rem;"
            )
        ),
    )
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
def server(input, output, session):

    @reactive.calc
    def models():
        return get_models()

    # ------------------------------------------------------------------
    # Tab 1 — Situation output
    # ------------------------------------------------------------------
    @output
    @render.ui
    def situation_output():
        m = models()
        if m is None:
            return ui.div(
                ui.div(
                    ui.h5("Data not found"),
                    ui.p(f"Expected: {DATA_PATH}"),
                    ui.p("Place encoded_fourth_downs.csv.gz one directory above this file and restart."),
                    class_="card"
                )
            )

        season = int(input.season())
        half_secs = min(input.game_secs(), 1800)
        row_pres = build_single_row(
            m, input.yardline(), input.ydstogo(), input.score_diff(),
            int(input.qtr()), input.game_secs(), half_secs,
            input.pos_to(), input.def_to(),
            input.goal_to_go(), input.shotgun(), input.no_huddle(),
            input.home(), season=season
        )
        row_pred = build_single_row(
            m, input.yardline(), input.ydstogo(), input.score_diff(),
            int(input.qtr()), input.game_secs(), half_secs,
            input.pos_to(), input.def_to(),
            input.goal_to_go(), input.shotgun(), input.no_huddle(),
            input.home(), season=season
        )
        # Align feature columns
        X_pres = row_pres[m["all_features_pres"]].fillna(0)
        X_pred = row_pred[[f for f in m["all_features_pred"] if f in row_pred.columns] +
                           [f for f in m["all_features_pred"] if f not in row_pred.columns]
                           ].reindex(columns=m["all_features_pred"], fill_value=0)

        # Prescriptive
        pres_proba = m["pres_model"].predict_proba(X_pres)[0]
        pres_classes = m["le_pres"].classes_
        pres_probs = dict(zip(pres_classes, pres_proba))
        optimal = max(pres_probs, key=pres_probs.get)

        # Predictive
        pred_proba = m["pred_model"].predict_proba(X_pred)[0]
        pred_classes = m["le_pred"].classes_
        pred_probs = dict(zip(pred_classes, pred_proba))
        predicted = max(pred_probs, key=pred_probs.get)

        match = optimal == predicted
        label_map = {"go": "Go For It", "punt": "Punt", "field_goal": "Field Goal"}
        badge_cls = {"go": "badge-go", "punt": "badge-punt", "field_goal": "badge-fg"}

        # Field position label
        yl = input.yardline()
        if yl == 50:
            fp_label = "Midfield"
        elif yl < 50:
            fp_label = f"Opp {yl}"
        else:
            fp_label = f"Own {100 - yl}"

        def prob_bars(probs_dict, color_map):
            bars = []
            for d in sorted(probs_dict, key=probs_dict.get, reverse=True):
                pct = probs_dict[d] * 100
                bars.append(
                    ui.div(
                        ui.div(
                            ui.span(label_map[d], style="font-size:13px; color:#374151;"),
                            ui.span(f"{pct:.0f}%", style="font-size:13px; font-weight:600;"),
                            style="display:flex; justify-content:space-between; margin-bottom:4px;"
                        ),
                        ui.div(
                            ui.div(style=f"height:8px; width:{pct:.0f}%; background:{color_map[d]}; border-radius:4px; transition:width .3s;"),
                            style="height:8px; background:#f3f4f6; border-radius:4px; overflow:hidden;"
                        ),
                        style="margin-bottom:10px;"
                    )
                )
            return bars

        alert_cls = "alert-green" if match else "alert-amber"
        alert_text = (
            f"✓ Model and predicted coach behavior agree — {label_map[optimal]}"
            if match else
            f"Divergence: Optimal = {label_map[optimal]}  ·  Coach likely to call = {label_map[predicted]}"
        )

        return ui.div(
            # Metric row
            ui.div(
                ui.div(
                    ui.div(fp_label, class_="metric-val"),
                    ui.div("Field position", class_="metric-lbl"),
                    class_="metric-card"
                ),
                ui.div(
                    ui.div(f"{input.ydstogo()}", class_="metric-val"),
                    ui.div("Yards to go", class_="metric-lbl"),
                    class_="metric-card"
                ),
                ui.div(
                    ui.div(f"{input.score_diff():+d}", class_="metric-val"),
                    ui.div("Score diff", class_="metric-lbl"),
                    class_="metric-card"
                ),
                ui.div(
                    ui.div(f"Q{input.qtr()}", class_="metric-val"),
                    ui.div("Quarter", class_="metric-lbl"),
                    class_="metric-card"
                ),
                style="display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:1rem;"
            ),

            # Two-column model output
            ui.div(
                # Prescriptive
                ui.div(
                    ui.div("Prescriptive model (EPA-optimal)", class_="card-title"),
                    ui.span(label_map[optimal], class_=f"badge {badge_cls[optimal]}",
                            style="font-size:15px; padding:8px 18px; margin-bottom:12px; display:inline-block;"),
                    *prob_bars(pres_probs, DECISION_COLORS),
                    class_="card"
                ),
                # Predictive
                ui.div(
                    ui.div("Predictive model (coach behavior)", class_="card-title"),
                    ui.span(label_map[predicted], class_=f"badge {badge_cls[predicted]}",
                            style="font-size:15px; padding:8px 18px; margin-bottom:12px; display:inline-block;"),
                    *prob_bars(pred_probs, DECISION_COLORS),
                    class_="card"
                ),
                style="display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1rem;"
            ),

            ui.div(alert_text, class_=alert_cls),
        )

    # ------------------------------------------------------------------
    # Tab 2 — Heatmap
    # ------------------------------------------------------------------
    @output
    @render.plot
    def heatmap_plot():
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        m = models()
        if m is None:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Data not loaded", ha="center", va="center", transform=ax.transAxes)
            return fig

        era_season = int(input.heatmap_era())
        heat = get_heatmap_data(m, era_season)

        decision_map = {"go": 0, "field_goal": 1, "punt": 2}
        heat["decision_num"] = heat["optimal"].map(decision_map)
        pivot = heat.pivot(index="ydstogo", columns="yardline_100", values="decision_num")

        fig, ax = plt.subplots(figsize=(13, 5))
        cmap = plt.cm.colors.ListedColormap(["#639922", "#BA7517", "#185FA5"])
        ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=-0.5, vmax=2.5, origin="upper")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(int(v)) for v in pivot.columns], rotation=45, fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_xlabel("Yards from opponent end zone", fontsize=11)
        ax.set_ylabel("Yards to go", fontsize=11)
        era_label = "Dynamic kickoff era (2023+)" if era_season >= 2023 else "Traditional kickoff era (pre-2023)"
        ax.set_title(f"Optimal 4th Down Decision — {era_label}\n(neutral game state: tied, Q3, 2 timeouts each)",
                     fontsize=12, fontweight="bold")

        patches = [
            mpatches.Patch(color="#639922", label="Go for it"),
            mpatches.Patch(color="#BA7517", label="Field goal"),
            mpatches.Patch(color="#185FA5", label="Punt"),
        ]
        ax.legend(handles=patches, loc="upper right", fontsize=10)

        # Add vertical lines at 20 and 50
        for boundary in [3, 6, 9]:  # index positions for yardline 20, 40, 60
            pass
        for xv, lbl in [(3, "Opp 40"), (6, "Midfield"), (9, "Own 40")]:
            ax.axvline(xv - 0.5, color="white", linewidth=0.8, linestyle="--", alpha=0.5)

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Tab 3 — Playcaller report
    # ------------------------------------------------------------------
    @output
    @render.ui
    def playcaller_output():
        m = models()
        if m is None:
            return ui.p("Data not loaded.")

        stats = m["playcaller_stats"].copy()
        league_avg = m["league_agree"]

        # Build bar chart with plotly
        top25 = stats.nlargest(25, "n_plays").sort_values("actual_agree_rate")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top25.index, x=top25["actual_agree_rate"],
            name="Actual coach", orientation="h",
            marker_color="#185FA5", opacity=0.85
        ))
        fig.add_trace(go.Bar(
            y=top25.index, x=top25["predicted_agree_rate"],
            name="Predicted by model", orientation="h",
            marker_color="#f97316", opacity=0.75
        ))
        fig.add_vline(x=league_avg, line_dash="dash", line_color="#185FA5",
                      annotation_text=f"League avg actual {league_avg:.1%}", annotation_position="top right")
        fig.update_layout(
            barmode="group", height=620,
            title="Playcaller Agreement with EPA-Optimal Decision (top 25 by volume)",
            xaxis_title="Agreement rate", xaxis_tickformat=".0%",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=120, r=20, t=60, b=40),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        fig.update_xaxes(range=[0, 1], gridcolor="#f3f4f6")
        fig.update_yaxes(gridcolor="#f3f4f6")

        chart_html = fig.to_html(include_plotlyjs="cdn", full_html=False)

        # Table: top 10 and bottom 10
        top10    = stats.nlargest(10, "actual_agree_rate")[["n_plays", "actual_agree_rate", "predicted_agree_rate"]]
        bottom10 = stats.nsmallest(10, "actual_agree_rate")[["n_plays", "actual_agree_rate", "predicted_agree_rate"]]

        def df_to_html_table(df, title, color):
            rows_html = ""
            for name, row in df.iterrows():
                rows_html += f"<tr><td style='padding:6px 12px;'>{name}</td><td style='padding:6px 12px; text-align:right;'>{int(row['n_plays'])}</td><td style='padding:6px 12px; text-align:right;'>{row['actual_agree_rate']:.1%}</td><td style='padding:6px 12px; text-align:right;'>{row['predicted_agree_rate']:.1%}</td></tr>"
            return f"""
            <div class="card" style="flex:1;">
              <div class="card-title" style="color:{color};">{title}</div>
              <table style="width:100%; border-collapse:collapse; font-size:13px;">
                <thead><tr style="border-bottom:1px solid #e5e7eb;">
                  <th style="text-align:left; padding:6px 12px; font-weight:600;">Playcaller</th>
                  <th style="text-align:right; padding:6px 12px; font-weight:600;">Plays</th>
                  <th style="text-align:right; padding:6px 12px; font-weight:600;">Actual agree</th>
                  <th style="text-align:right; padding:6px 12px; font-weight:600;">Predicted agree</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
              </table>
            </div>"""

        return ui.div(
            ui.HTML(f'<div class="card">{chart_html}</div>'),
            ui.HTML(f'<div style="display:flex; gap:1rem;">'
                    f'{df_to_html_table(top10, "Most Optimal Playcallers", "#15803d")}'
                    f'{df_to_html_table(bottom10, "Least Optimal Playcallers", "#b45309")}'
                    f'</div>'),
        )

    # ------------------------------------------------------------------
    # Tab 4 — Model summary
    # ------------------------------------------------------------------
    @output
    @render.ui
    def model_summary_output():
        m = models()
        if m is None:
            return ui.p("Data not loaded.")

        pres_params = load_params(PRES_PARAMS_PATH, PRES_FALLBACK_PARAMS)
        pred_params = load_params(PRED_PARAMS_PATH, PRED_FALLBACK_PARAMS)

        def param_rows(params):
            return "".join(
                f"<tr><td style='padding:5px 12px; color:#6b7280;'>{k}</td><td style='padding:5px 12px; font-weight:500;'>{v}</td></tr>"
                for k, v in params.items()
            )

        return ui.div(
            ui.div(
                ui.div(
                    ui.div(f"{m['pres_acc']:.1%}", class_="metric-val"),
                    ui.div("Prescriptive accuracy", class_="metric-lbl"),
                    class_="metric-card"
                ),
                ui.div(
                    ui.div(f"{m['pred_acc']:.1%}", class_="metric-val"),
                    ui.div("Predictive accuracy", class_="metric-lbl"),
                    class_="metric-card"
                ),
                ui.div(
                    ui.div(f"{m['league_agree']:.1%}", class_="metric-val"),
                    ui.div("League-wide agree with optimal", class_="metric-lbl"),
                    class_="metric-card"
                ),
                ui.div(
                    ui.div(f"{m['n_divergent']:,}", class_="metric-val"),
                    ui.div("Divergent plays (2024 test)", class_="metric-lbl"),
                    class_="metric-card"
                ),
                style="display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:1rem;"
            ),
            ui.HTML(f"""
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem;">
              <div class="card">
                <div class="card-title">Prescriptive XGBoost params</div>
                <table style="width:100%; border-collapse:collapse; font-size:13px;">
                  {param_rows(pres_params)}
                </table>
              </div>
              <div class="card">
                <div class="card-title">Predictive XGBoost params</div>
                <table style="width:100%; border-collapse:collapse; font-size:13px;">
                  {param_rows(pred_params)}
                </table>
              </div>
            </div>
            """),
            ui.div(
                ui.div("Train seasons: 2018–2023  ·  Test season: 2024  ·  Temporal split prevents leakage",
                       style="font-size:13px; color:#6b7280;"),
                ui.div(f"Params source: {'JSON files' if PRES_PARAMS_PATH.exists() else 'fallback defaults'}",
                       style="font-size:13px; color:#6b7280; margin-top:4px;"),
                class_="card"
            )
        )


app = App(app_ui, server)
