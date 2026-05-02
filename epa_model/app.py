# =============================================================================
# 4th & Context — Decision Support Dashboard
# IST 707 | Gullo, Weber, Stein
#
# Requirements:
#   pip install shiny xgboost scikit-learn pandas numpy plotly matplotlib
#
# Run (from repo root):
#   python -m shiny run epa_model/app.py --reload
#
# File layout expected:
#   group-project-team1/
#   ├── encoded_fourth_downs.csv.gz        <-- one level above epa_model/
#   └── epa_model/
#       ├── app.py
#       ├── best_xgb_params.json           <-- prescriptive model params
#       └── best_predictive_xgb_params.json <-- predictive model params
#
# If JSON param files are missing, hardcoded fallback params are used.
# Models are trained lazily on first tab visit (30-60 s), then cached.
# =============================================================================

from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb
from shiny import App, reactive, render, ui
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE            = Path(__file__).parent
DATA_PATH       = HERE.parent / "encoded_fourth_downs.csv.gz"
PRES_PARAMS_PATH = HERE / "best_xgb_params.json"
PRED_PARAMS_PATH = HERE / "best_predictive_xgb_params.json"

# ---------------------------------------------------------------------------
# Fallback XGBoost params (from notebook tuning runs)
# ---------------------------------------------------------------------------
PRES_FALLBACK = {
    "n_estimators": 400, "max_depth": 6, "learning_rate": 0.08,
    "subsample": 0.85, "colsample_bytree": 0.80, "min_child_weight": 5,
    "gamma": 1.2, "reg_alpha": 0.05, "reg_lambda": 1.5,
}
PRED_FALLBACK = {
    "n_estimators": 350, "max_depth": 7, "learning_rate": 0.10,
    "subsample": 0.80, "colsample_bytree": 0.75, "min_child_weight": 3,
    "gamma": 0.8, "reg_alpha": 0.02, "reg_lambda": 1.0,
}

# ---------------------------------------------------------------------------
# Constants (must match notebook feature engineering exactly)
# ---------------------------------------------------------------------------
PRE_SNAP = [
    "yardline_100", "ydstogo", "score_differential",
    "game_seconds_remaining", "half_seconds_remaining", "qtr",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "goal_to_go", "shotgun", "no_huddle", "home_is_posteam",
]
PREGAME_EPA = ["no_score_prob", "opp_fg_prob", "opp_td_prob", "fg_prob", "td_prob"]
GROUP_COLS  = ["ydstogo_bin", "yardline_bin", "score_diff_bin", "kickoff_era"]
TEST_SEASON = 2024
LABEL_MAP   = {"go": "Go For It", "punt": "Punt", "field_goal": "Field Goal"}
COLORS      = {"go": "#639922",    "punt": "#185FA5", "field_goal": "#BA7517"}
BADGE_CLS   = {"go": "badge-go",   "punt": "badge-punt", "field_goal": "badge-fg"}

# Module-level cache — populated lazily on first use
_cache: dict = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_params(path: Path, fallback: dict) -> dict:
    if path.exists():
        with open(path) as f:
            params = json.load(f)
        print(f"  Loaded {path.name}")
        return params
    print(f"  {path.name} not found — using fallback params")
    return fallback


def engineer_features(df: pd.DataFrame):
    """Adds era flag, season dummies, returns (df, feature_list, playcaller_cols)."""
    pc_cols = [c for c in df.columns if c.startswith("playcaller_")]
    df = df.copy()
    df["is_dynamic_era"] = (df["season"] >= 2023).astype(int)
    sdummies = pd.get_dummies(df["season"], prefix="season", drop_first=True)
    df = pd.concat([df, sdummies], axis=1)
    feats = PRE_SNAP + PREGAME_EPA + pc_cols + ["is_dynamic_era"] + list(sdummies.columns)
    feats = [f for f in feats if f in df.columns]
    return df, feats, pc_cols


def add_prescriptive_labels(df: pd.DataFrame):
    """Bucket-mean EPA → optimal decision label. Returns (df, epa_pivot)."""
    df = df.copy()
    df["ydstogo_bin"]    = pd.cut(df["ydstogo"],            bins=[0,1,3,6,10,99],
                                   labels=["1","2-3","4-6","7-10","10+"])
    df["yardline_bin"]   = pd.cut(df["yardline_100"],       bins=[0,20,40,60,80,100],
                                   labels=["opp_red_zone","opp_40","midfield","own_40","own_end"])
    df["score_diff_bin"] = pd.cut(df["score_differential"], bins=[-100,-14,-7,-3,3,7,14,100],
                                   labels=["down_14+","down_8-14","down_1-7",
                                           "close","up_1-7","up_8-14","up_14+"])
    df["kickoff_era"] = df["season"].apply(lambda s: "dynamic" if s >= 2023 else "traditional")

    pivot = (
        df.groupby(GROUP_COLS + ["decision"])["epa"]
        .mean().reset_index().rename(columns={"epa": "mean_epa"})
        .pivot_table(index=GROUP_COLS, columns="decision", values="mean_epa")
        .reset_index()
    )
    pivot.columns.name = None
    for col in ["go", "punt", "field_goal"]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot["optimal_decision"] = pivot[["go","punt","field_goal"]].idxmax(axis=1)
    df = df.merge(pivot[GROUP_COLS + ["optimal_decision"]], on=GROUP_COLS, how="left")
    df = df.dropna(subset=["optimal_decision"])
    return df, pivot


# ---------------------------------------------------------------------------
# Core model loader — trains once, caches everything
# ---------------------------------------------------------------------------
def get_models() -> dict:
    if "ready" in _cache:
        return _cache

    print("Loading data…")
    if not DATA_PATH.exists():
        _cache["ready"] = False
        _cache["error"] = f"Data file not found:\n{DATA_PATH}"
        return _cache

    raw = pd.read_csv(DATA_PATH, compression="gzip", low_memory=False)
    print(f"  {len(raw):,} rows")

    # Reconstruct decision
    if "play_type" in raw.columns:
        raw["decision"] = raw["play_type"].map(
            {"run":"go","pass":"go","punt":"punt","field_goal":"field_goal"})
    else:
        def _decide(row):
            if row.get("play_type_pass",0)==1 or row.get("play_type_run",0)==1: return "go"
            if row.get("play_type_punt",0)==1:      return "punt"
            if row.get("field_goal_attempt",0)==1:  return "field_goal"
            return np.nan
        raw["decision"] = raw.apply(_decide, axis=1)

    df_clean = raw.dropna(subset=["decision"]).copy()
    df_clean = df_clean[df_clean["decision"].isin(["go","punt","field_goal"])]

    # ---- Prescriptive model ------------------------------------------------
    print("Training prescriptive model…")
    df_p, epa_pivot = add_prescriptive_labels(df_clean.dropna(subset=["epa"]).copy())
    df_p, feat_p, pc_cols = engineer_features(df_p)

    X_p = df_p[feat_p].copy()
    X_p[X_p.select_dtypes("bool").columns] = X_p.select_dtypes("bool").astype(int)
    mask_p   = X_p.notna().all(axis=1)
    X_p      = X_p[mask_p]
    y_p_raw  = df_p.loc[mask_p, "optimal_decision"]
    season_p = df_p.loc[mask_p, "season"]

    le_p    = LabelEncoder()
    y_p_enc = le_p.fit_transform(y_p_raw)

    tr_p = season_p[season_p < TEST_SEASON].index
    te_p = season_p[season_p == TEST_SEASON].index

    pp = load_params(PRES_PARAMS_PATH, PRES_FALLBACK)
    pm = xgb.XGBClassifier(eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0, **pp)
    pm.fit(X_p.loc[tr_p], y_p_enc[X_p.index.get_indexer(tr_p)])
    pres_acc = accuracy_score(y_p_enc[X_p.index.get_indexer(te_p)], pm.predict(X_p.loc[te_p]))
    print(f"  Prescriptive acc: {pres_acc:.4f}")

    # ---- Predictive model --------------------------------------------------
    print("Training predictive model…")
    df_d, feat_d, _ = engineer_features(df_clean.copy())

    X_d = df_d[feat_d].copy()
    X_d[X_d.select_dtypes("bool").columns] = X_d.select_dtypes("bool").astype(int)
    mask_d   = X_d.notna().all(axis=1)
    X_d      = X_d[mask_d]
    y_d_raw  = df_d.loc[mask_d, "decision"]
    season_d = df_d.loc[mask_d, "season"]

    le_d    = LabelEncoder()
    y_d_enc = le_d.fit_transform(y_d_raw)

    tr_d = season_d[season_d < TEST_SEASON].index
    te_d = season_d[season_d == TEST_SEASON].index

    dp = load_params(PRED_PARAMS_PATH, PRED_FALLBACK)
    dm = xgb.XGBClassifier(eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0, **dp)
    dm.fit(X_d.loc[tr_d], y_d_enc[X_d.index.get_indexer(tr_d)])
    pred_acc = accuracy_score(y_d_enc[X_d.index.get_indexer(te_d)], dm.predict(X_d.loc[te_d]))
    print(f"  Predictive acc:   {pred_acc:.4f}")

    # ---- Playcaller agreement stats ----------------------------------------
    print("Computing playcaller agreement…")
    df_sc = df_p[mask_p].copy()
    # Align prescriptive feature rows to predictive feature list for pred model
    X_p_for_pred = X_p.reindex(columns=feat_d, fill_value=0)
    df_sc["predicted_decision"] = le_d.inverse_transform(dm.predict(X_p_for_pred))
    df_sc["actual_matches_optimal"]    = (df_sc["decision"]           == df_sc["optimal_decision"]).astype(int)
    df_sc["predicted_matches_optimal"] = (df_sc["predicted_decision"] == df_sc["optimal_decision"]).astype(int)
    pc_here = [c for c in pc_cols if c in df_sc.columns]
    df_sc["playcaller"] = df_sc[pc_here].apply(
        lambda r: next((c.replace("playcaller_","") for c in pc_here if r[c]==1), "Unknown"), axis=1)

    pc_stats = (
        df_sc.groupby("playcaller")
        .agg(n_plays=("decision","count"),
             actual_agree_rate=("actual_matches_optimal","mean"),
             predicted_agree_rate=("predicted_matches_optimal","mean"))
        .query("n_plays >= 20")
        .sort_values("actual_agree_rate", ascending=False)
        .round(3)
    )

    _cache.update({
        "ready": True,
        "pres_model": pm, "le_pres": le_p, "feat_pres": feat_p,
        "pred_model": dm, "le_pred": le_d, "feat_pred": feat_d,
        "epa_pivot": epa_pivot, "pc_stats": pc_stats,
        "pres_acc": pres_acc, "pred_acc": pred_acc,
        "pres_params": pp, "pred_params": dp,
        "params_from_file": PRES_PARAMS_PATH.exists(),
        "league_agree": df_sc["actual_matches_optimal"].mean(),
        "n_divergent": int((df_sc.loc[season_p == TEST_SEASON, "actual_matches_optimal"]==0).sum()),
    })
    print("Done. Dashboard ready.")
    return _cache


# ---------------------------------------------------------------------------
# Feature row builder
# ---------------------------------------------------------------------------
def make_row(feat_list: list, yardline, ydstogo, score_diff, qtr,
             game_secs, half_secs, pos_to, def_to,
             goal_to_go, shotgun, no_huddle, home, season=2024) -> pd.DataFrame:
    base = dict.fromkeys(feat_list, 0.0)
    base.update({
        "yardline_100": yardline,  "ydstogo": ydstogo,
        "score_differential": score_diff, "qtr": qtr,
        "game_seconds_remaining": game_secs, "half_seconds_remaining": half_secs,
        "posteam_timeouts_remaining": pos_to, "defteam_timeouts_remaining": def_to,
        "goal_to_go": int(goal_to_go), "shotgun": int(shotgun),
        "no_huddle": int(no_huddle), "home_is_posteam": int(home),
        "is_dynamic_era": int(season >= 2023),
        "no_score_prob": 0.17, "opp_fg_prob": 0.05, "opp_td_prob": 0.19,
        "fg_prob": 0.06, "td_prob": 0.22,
    })
    sk = f"season_{season}"
    if sk in base:
        base[sk] = 1
    return pd.DataFrame([base])


# ---------------------------------------------------------------------------
# Heatmap (cached per season)
# ---------------------------------------------------------------------------
def get_heatmap(season: int) -> pd.DataFrame:
    key = f"hm_{season}"
    if key in _cache:
        return _cache[key]
    m = _cache
    coords, rows = [], []
    for yl in range(5, 100, 5):
        for dist in [1, 2, 3, 4, 5, 6, 7, 8, 10, 15]:
            rows.append(make_row(m["feat_pres"], yl, dist, 0, 3, 1800, 900,
                                 2, 2, int(yl<=dist), 0, 0, 0, season).iloc[0])
            coords.append((yl, dist))
    X = pd.DataFrame(rows)
    labels = m["le_pres"].inverse_transform(m["pres_model"].predict(X))
    df = pd.DataFrame(coords, columns=["yardline_100","ydstogo"])
    df["optimal"] = labels
    _cache[key] = df
    return df


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def prob_bars_html(probs: dict) -> str:
    out = ""
    for d in sorted(probs, key=probs.get, reverse=True):
        pct = probs[d] * 100
        out += (
            f'<div style="margin-bottom:10px;">'
            f'<div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px;">'
            f'<span style="color:#6b7280;">{LABEL_MAP[d]}</span>'
            f'<span style="font-weight:600;">{pct:.0f}%</span></div>'
            f'<div style="height:8px;background:#f3f4f6;border-radius:4px;overflow:hidden;">'
            f'<div style="height:100%;width:{pct:.0f}%;background:{COLORS[d]};'
            f'border-radius:4px;transition:width .3s;"></div></div></div>'
        )
    return out


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CSS = """
body { background:#f8f9fa; font-family:'Segoe UI',system-ui,sans-serif; }
.card { background:white; border-radius:10px; border:1px solid #e5e7eb;
        padding:1.25rem; margin-bottom:1rem; }
.card-title { font-size:11px; font-weight:600; text-transform:uppercase;
              letter-spacing:.07em; color:#6b7280; margin-bottom:.75rem; }
.badge { display:inline-block; padding:5px 14px; border-radius:6px;
         font-size:13px; font-weight:600; }
.badge-go   { background:#EAF3DE; color:#3B6D11; border:1.5px solid #639922; }
.badge-punt { background:#E6F1FB; color:#0C447C; border:1.5px solid #378ADD; }
.badge-fg   { background:#FAEEDA; color:#854F0B; border:1.5px solid #BA7517; }
.alert-green { background:#f0fdf4; border:1px solid #86efac; border-radius:8px;
               padding:.75rem 1rem; color:#15803d; font-size:14px; margin-top:.75rem; }
.alert-amber { background:#fffbeb; border:1px solid #fcd34d; border-radius:8px;
               padding:.75rem 1rem; color:#92400e; font-size:14px; margin-top:.75rem; }
.loading-box { background:white; border-radius:10px; border:1px solid #e5e7eb;
               padding:3rem; text-align:center; color:#6b7280; font-size:15px; }
.two-col  { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
.four-col { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-bottom:1rem; }
"""

# ---------------------------------------------------------------------------
# Metric card HTML
# ---------------------------------------------------------------------------
def mcard(value, label):
    return (f'<div style="background:#f3f4f6;border-radius:8px;padding:.9rem;text-align:center;">'
            f'<div style="font-size:24px;font-weight:600;">{value}</div>'
            f'<div style="font-size:11px;color:#6b7280;margin-top:4px;">{label}</div></div>')


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
app_ui = ui.page_fluid(
    ui.tags.head(ui.tags.style(CSS)),
    ui.div(
        ui.h2("4th & Context", style="font-size:22px;font-weight:600;margin:0 0 .2rem;"),
        ui.p("NFL 4th Down Decision Support  ·  IST 707",
             style="color:#6b7280;font-size:13px;margin:0 0 1rem;"),
        style="padding:1.25rem 1.25rem 0;"
    ),
    ui.navset_tab(

        # ---- Tab 1: Situation Evaluator ----
        ui.nav_panel("Situation Evaluator",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h6("Game situation", style="font-weight:600;margin-bottom:.75rem;"),
                    ui.input_slider("yardline",   "Yard line (from opp. end zone)", 1, 99, 55),
                    ui.input_slider("ydstogo",    "Yards to go", 1, 20, 5),
                    ui.input_slider("score_diff", "Score differential (+ = leading)", -28, 28, 0),
                    ui.input_select("qtr", "Quarter",
                                    {"1":"Q1","2":"Q2","3":"Q3","4":"Q4","5":"OT"}, selected="3"),
                    ui.input_slider("game_secs",  "Game seconds remaining", 0, 3600, 1800, step=30),
                    ui.hr(),
                    ui.h6("Context", style="font-weight:600;margin-bottom:.75rem;"),
                    ui.input_slider("pos_to",     "Possession team timeouts", 0, 3, 2),
                    ui.input_slider("def_to",     "Defense timeouts",         0, 3, 2),
                    ui.input_checkbox("goal_to_go","Goal to go",          False),
                    ui.input_checkbox("shotgun",   "Shotgun formation",   False),
                    ui.input_checkbox("no_huddle", "No huddle",           False),
                    ui.input_checkbox("home",      "Possession = home",   True),
                    ui.input_select("season", "Season",
                                    {str(y): str(y) for y in range(2018, 2026)},
                                    selected="2024"),
                    width=290,
                ),
                ui.div(ui.output_ui("situation_ui"), style="padding:.5rem;"),
            )
        ),

        # ---- Tab 2: Decision Heatmap ----
        ui.nav_panel("Decision Heatmap",
            ui.div(
                ui.div(
                    ui.input_select("hm_season", "Kickoff era",
                                    {"2024":"Dynamic kickoff (2023+)",
                                     "2019":"Traditional kickoff (pre-2023)"}),
                    style="max-width:260px;margin-bottom:1rem;"
                ),
                ui.output_plot("heatmap_plot", height="460px"),
                style="padding:1rem;"
            )
        ),

        # ---- Tab 3: Playcaller Report ----
        ui.nav_panel("Playcaller Report",
            ui.div(ui.output_ui("playcaller_ui"), style="padding:1rem;")
        ),

        # ---- Tab 4: Model Summary ----
        ui.nav_panel("Model Summary",
            ui.div(ui.output_ui("summary_ui"), style="padding:1rem;")
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

    @reactive.calc
    def prediction():
        m = models()
        if not m.get("ready"):
            return None
        gs = input.game_secs()
        kw = dict(
            yardline=input.yardline(), ydstogo=input.ydstogo(),
            score_diff=input.score_diff(), qtr=int(input.qtr()),
            game_secs=gs, half_secs=min(gs, 1800),
            pos_to=input.pos_to(), def_to=input.def_to(),
            goal_to_go=input.goal_to_go(), shotgun=input.shotgun(),
            no_huddle=input.no_huddle(), home=input.home(),
            season=int(input.season()),
        )
        Xp = make_row(m["feat_pres"], **kw)
        Xd = make_row(m["feat_pred"], **kw)

        pp = dict(zip(m["le_pres"].classes_, m["pres_model"].predict_proba(Xp)[0]))
        dp = dict(zip(m["le_pred"].classes_, m["pred_model"].predict_proba(Xd)[0]))
        return dict(pres=pp, optimal=max(pp, key=pp.get),
                    pred=dp, predicted=max(dp, key=dp.get))

    # ---- Tab 1 ----
    @output
    @render.ui
    def situation_ui():
        m = models()
        if not m.get("ready"):
            msg = m.get("error", "Models loading — this takes ~60 s on first visit…")
            return ui.HTML(f'<div class="loading-box">{msg}</div>')

        p = prediction()
        yl = input.yardline()
        fp = f"Opp {yl}" if yl < 50 else ("Midfield" if yl == 50 else f"Own {100-yl}")
        match = p["optimal"] == p["predicted"]
        alert = ("alert-green",
                 f"✓ Both models agree — {LABEL_MAP[p['optimal']]}"
                 ) if match else (
                 "alert-amber",
                 f"⚠ Divergence: EPA-optimal = {LABEL_MAP[p['optimal']]}  ·  "
                 f"Coach likely to call = {LABEL_MAP[p['predicted']]}")

        return ui.HTML(f"""
        <div class="four-col">
          {mcard(fp,                    "Field position")}
          {mcard(input.ydstogo(),       "Yards to go")}
          {mcard(f"{input.score_diff():+d}", "Score differential")}
          {mcard(f"Q{input.qtr()}",     "Quarter")}
        </div>
        <div class="two-col">
          <div class="card">
            <div class="card-title">Prescriptive model — EPA-optimal</div>
            <span class="badge {BADGE_CLS[p['optimal']]}"
                  style="font-size:15px;padding:8px 18px;display:inline-block;margin-bottom:12px;">
              {LABEL_MAP[p['optimal']]}
            </span>
            {prob_bars_html(p['pres'])}
          </div>
          <div class="card">
            <div class="card-title">Predictive model — coach behavior</div>
            <span class="badge {BADGE_CLS[p['predicted']]}"
                  style="font-size:15px;padding:8px 18px;display:inline-block;margin-bottom:12px;">
              {LABEL_MAP[p['predicted']]}
            </span>
            {prob_bars_html(p['pred'])}
          </div>
        </div>
        <div class="{alert[0]}">{alert[1]}</div>
        """)

    # ---- Tab 2 ----
    @output
    @render.plot
    def heatmap_plot():
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import ListedColormap

        m = models()
        if not m.get("ready"):
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Data not loaded", ha="center", va="center",
                    transform=ax.transAxes, fontsize=13, color="#6b7280")
            ax.axis("off")
            return fig

        heat = get_heatmap(int(input.hm_season()))
        heat["dn"] = heat["optimal"].map({"go":0,"field_goal":1,"punt":2})
        pivot = heat.pivot(index="ydstogo", columns="yardline_100", values="dn")

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.imshow(pivot.values,
                  cmap=ListedColormap(["#639922","#BA7517","#185FA5"]),
                  aspect="auto", vmin=-0.5, vmax=2.5, origin="upper")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(int(v)) for v in pivot.columns], rotation=45, fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_xlabel("Yards from opponent end zone", fontsize=11)
        ax.set_ylabel("Yards to go", fontsize=11)
        era = ("Dynamic kickoff era (2023+)" if int(input.hm_season()) >= 2023
               else "Traditional kickoff era (pre-2023)")
        ax.set_title(f"Optimal 4th Down Decision — {era}\n"
                     f"(neutral: tied, Q3, 2 timeouts each side)",
                     fontsize=11, fontweight="bold", pad=12)
        ax.legend(handles=[
            mpatches.Patch(color="#639922", label="Go for it"),
            mpatches.Patch(color="#BA7517", label="Field goal"),
            mpatches.Patch(color="#185FA5", label="Punt"),
        ], loc="upper right", fontsize=10, framealpha=0.9)
        ax.tick_params(length=0)
        fig.tight_layout()
        return fig

    # ---- Tab 3 ----
    @output
    @render.ui
    def playcaller_ui():
        m = models()
        if not m.get("ready"):
            return ui.HTML('<div class="loading-box">Data not loaded.</div>')

        stats  = m["pc_stats"]
        lg_avg = m["league_agree"]
        top25  = stats.nlargest(25, "n_plays").sort_values("actual_agree_rate")

        fig = go.Figure([
            go.Bar(y=top25.index, x=top25["actual_agree_rate"],
                   name="Actual coach", orientation="h",
                   marker_color="#185FA5", opacity=0.85),
            go.Bar(y=top25.index, x=top25["predicted_agree_rate"],
                   name="Predicted by model", orientation="h",
                   marker_color="#f97316", opacity=0.75),
        ])
        fig.add_vline(x=lg_avg, line_dash="dash", line_color="#185FA5",
                      annotation_text=f"League avg {lg_avg:.1%}",
                      annotation_position="top right")
        fig.update_layout(
            barmode="group", height=640,
            title="Playcaller Agreement with EPA-Optimal Decision (top 25 by volume)",
            xaxis=dict(title="Agreement rate", tickformat=".0%", range=[0,1],
                       gridcolor="#f3f4f6"),
            yaxis=dict(gridcolor="#f3f4f6"),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
            margin=dict(l=140, r=20, t=60, b=40),
            plot_bgcolor="white", paper_bgcolor="white",
        )

        def tbl(df, title, color):
            rows = "".join(
                f'<tr style="border-bottom:1px solid #f3f4f6;">'
                f'<td style="padding:6px 10px;">{name}</td>'
                f'<td style="padding:6px 10px;text-align:right;">{int(r["n_plays"])}</td>'
                f'<td style="padding:6px 10px;text-align:right;">{r["actual_agree_rate"]:.1%}</td>'
                f'<td style="padding:6px 10px;text-align:right;color:#6b7280;">'
                f'{r["predicted_agree_rate"]:.1%}</td></tr>'
                for name, r in df.iterrows()
            )
            return (f'<div class="card" style="flex:1;min-width:0;">'
                    f'<div class="card-title" style="color:{color};">{title}</div>'
                    f'<table style="width:100%;border-collapse:collapse;font-size:13px;">'
                    f'<thead><tr style="border-bottom:2px solid #e5e7eb;">'
                    f'<th style="text-align:left;padding:6px 10px;">Playcaller</th>'
                    f'<th style="text-align:right;padding:6px 10px;">Plays</th>'
                    f'<th style="text-align:right;padding:6px 10px;">Actual</th>'
                    f'<th style="text-align:right;padding:6px 10px;color:#6b7280;">Predicted</th>'
                    f'</tr></thead><tbody>{rows}</tbody></table></div>')

        return ui.div(
            ui.HTML(f'<div class="card">'
                    f'{fig.to_html(include_plotlyjs="cdn", full_html=False)}</div>'),
            ui.HTML(f'<div style="display:flex;gap:1rem;">'
                    f'{tbl(stats.nlargest(10,"actual_agree_rate"),  "Most optimal",  "#15803d")}'
                    f'{tbl(stats.nsmallest(10,"actual_agree_rate"), "Least optimal", "#b45309")}'
                    f'</div>'),
        )

    # ---- Tab 4 ----
    @output
    @render.ui
    def summary_ui():
        m = models()
        if not m.get("ready"):
            return ui.HTML('<div class="loading-box">Data not loaded.</div>')

        def param_rows(p):
            return "".join(
                f'<tr style="border-bottom:1px solid #f9fafb;">'
                f'<td style="padding:5px 12px;color:#6b7280;">{k}</td>'
                f'<td style="padding:5px 12px;font-weight:500;font-family:monospace;">{v}</td></tr>'
                for k, v in p.items()
            )

        src = "JSON files" if m["params_from_file"] else "hardcoded fallback defaults"
        return ui.HTML(f"""
        <div class="four-col">
          {mcard(f"{m['pres_acc']:.1%}",   "Prescriptive accuracy")}
          {mcard(f"{m['pred_acc']:.1%}",   "Predictive accuracy")}
          {mcard(f"{m['league_agree']:.1%}","League agree w/ optimal")}
          {mcard(f"{m['n_divergent']:,}",   "Divergent plays (2024)")}
        </div>
        <div class="two-col">
          <div class="card">
            <div class="card-title">Prescriptive XGBoost params</div>
            <table style="width:100%;border-collapse:collapse;font-size:13px;">
              {param_rows(m['pres_params'])}
            </table>
          </div>
          <div class="card">
            <div class="card-title">Predictive XGBoost params</div>
            <table style="width:100%;border-collapse:collapse;font-size:13px;">
              {param_rows(m['pred_params'])}
            </table>
          </div>
        </div>
        <div class="card" style="font-size:13px;color:#6b7280;line-height:1.9;">
          Train: 2018–2023 &nbsp;·&nbsp; Test: 2024 &nbsp;·&nbsp;
          Temporal split prevents leakage &nbsp;·&nbsp; Params source: {src}
        </div>
        """)


app = App(app_ui, server)