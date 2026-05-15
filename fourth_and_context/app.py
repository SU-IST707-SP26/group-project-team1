"""4th & Context — NFL 4th Down Decision Support Dashboard

IST 707 | Gullo, Weber, Stein

Run:
    python -m shiny run app.py --reload

First launch trains both XGBoost models (~30-60s) and pickles them to
models_cache/. Subsequent launches load from disk in <1s. To force
retraining (e.g. after editing JSON params), delete models_cache/.

Tabs:
  1. Situation Evaluator — slider-driven recommendation, with optional
                           era comparison side-by-side
  2. Decision Heatmap    — yardline x ydstogo grid colored by optimal call;
                           respects context inputs from the sidebar
  3. Playcaller Report   — coach agreement with EPA-optimal, top 25 by volume
  4. Audit               — drill-down on suspicious model predictions
                           (model-predicted punts inside opponent's 40)
  5. Model Summary       — accuracy, params, training metadata
"""
from __future__ import annotations
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
from shiny import App, reactive, render, ui

from constants import LABEL_MAP, COLORS, BADGE_CLS, DECISIONS, TEST_SEASON
from data_loader import load_clean_plays, prepare_modeling_frames
from models import train_and_cache
from analytics import (
    compute_playcaller_stats, compute_yearly_agreement, compute_audit_cases,
    compute_heatmap_grid, predict_one,
)
from ui_components import (
    CSS, metric_card, decision_badge, prob_bars, alert_box,
    field_position_label,
)

warnings.filterwarnings("ignore")

# Module-level cache: populated lazily by load_everything() on first reactive call.
_state: dict = {}


# ---------------------------------------------------------------------------
# Boot — runs once on first reactive evaluation
# ---------------------------------------------------------------------------
def load_everything() -> dict:
    """Load data, train (or load cached) models, compute analytics. Cached."""
    if "ready" in _state:
        return _state

    print("=== 4th & Context — booting ===")
    try:
        df_clean = load_clean_plays()
        print(f"Loaded {len(df_clean):,} clean 4th-down plays")

        frames = prepare_modeling_frames(df_clean)
        print(f"Prescriptive frame: {frames['X_pres'].shape}")
        print(f"Predictive frame:   {frames['X_pred'].shape}")
        print(f"Features: {len(frames['features'])}")

        bundle = train_and_cache(frames)

        print("Computing playcaller analytics…")
        pc_stats, df_scored, league_agree = compute_playcaller_stats(
            df_pres       = frames["df_pres"],
            pres_features = frames["features"],
            pres_model    = bundle["pres_model"],
            pred_model    = bundle["pred_model"],
            le_pres       = bundle["le_pres"],
            le_pred       = bundle["le_pred"],
            X_pres        = frames["X_pres"],
            pc_cols       = frames["playcaller_cols"],
        )
        yearly = compute_yearly_agreement(df_scored)
        audit  = compute_audit_cases(df_scored)

        n_divergent = int((
            df_scored.loc[df_scored["season"] == TEST_SEASON, "actual_matches_optimal"] == 0
        ).sum())

        _state.update({
            "ready":            True,
            "features":         frames["features"],
            "pres_model":       bundle["pres_model"],
            "pred_model":       bundle["pred_model"],
            "le_pres":          bundle["le_pres"],
            "le_pred":          bundle["le_pred"],
            "pres_acc":         bundle["pres_acc"],
            "pred_acc":         bundle["pred_acc"],
            "pres_params":      bundle["pres_params"],
            "pred_params":      bundle["pred_params"],
            "params_from_file": bundle["params_from_file"],
            "pc_stats":         pc_stats,
            "yearly":           yearly,
            "audit":            audit,
            "league_agree":     league_agree,
            "n_divergent":      n_divergent,
            "heatmap_cache":    {},
        })
        print(f"Ready. League agree w/ optimal: {league_agree:.1%}, "
              f"divergent 2024 plays: {n_divergent:,}")
    except FileNotFoundError as e:
        _state.update({"ready": False, "error": f"Missing file: {e}"})
    except Exception as e:
        _state.update({"ready": False, "error": f"{type(e).__name__}: {e}"})
        import traceback; traceback.print_exc()
    return _state


def get_heatmap(*, season: int, score_diff: int, qtr: int,
                pos_to: int, def_to: int):
    """Cached heatmap grid keyed on the inputs that vary (Phase 2 change:
    no longer hardcoded to neutral context)."""
    key = (season, score_diff, qtr, pos_to, def_to)
    cache = _state.setdefault("heatmap_cache", {})
    if key in cache:
        return cache[key]
    grid = compute_heatmap_grid(
        _state["features"], _state["pres_model"], _state["le_pres"],
        season=season, score_diff=score_diff, qtr=qtr,
        game_secs=1800, half_secs=900,
        pos_to=pos_to, def_to=def_to,
    )
    cache[key] = grid
    return grid


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
SEASON_CHOICES = {str(y): str(y) for y in range(2018, 2026)}

app_ui = ui.page_fluid(
    ui.tags.head(ui.tags.style(CSS)),
    ui.div(
        ui.h2("4th & Context", style="font-size:22px;font-weight:600;margin:0 0 .2rem;"),
        ui.p("NFL 4th Down Decision Support  ·  IST 707",
             style="color:#6b7280;font-size:13px;margin:0 0 1rem;"),
        style="padding:1.25rem 1.25rem 0;",
    ),
    ui.navset_tab(
        # ---- Tab 1: Situation Evaluator ----
        ui.nav_panel(
            "Situation Evaluator",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h6("Game situation", style="font-weight:600;margin-bottom:.75rem;"),
                    ui.input_slider("yardline",   "Yard line (from opp. end zone)", 1, 99, 55),
                    ui.input_slider("ydstogo",    "Yards to go", 1, 20, 5),
                    ui.input_slider("score_diff", "Score differential (+ = leading)", -28, 28, 0),
                    ui.input_select("qtr", "Quarter",
                                    {"1":"Q1","2":"Q2","3":"Q3","4":"Q4","5":"OT"},
                                    selected="3"),
                    ui.input_slider("game_secs",  "Game seconds remaining", 0, 3600, 1800, step=30),
                    ui.hr(),
                    ui.h6("Context", style="font-weight:600;margin-bottom:.75rem;"),
                    ui.input_slider("pos_to",     "Possession team timeouts", 0, 3, 2),
                    ui.input_slider("def_to",     "Defense timeouts",         0, 3, 2),
                    ui.input_checkbox("goal_to_go", "Goal to go",          False),
                    ui.input_checkbox("shotgun",    "Shotgun formation",   False),
                    ui.input_checkbox("no_huddle",  "No huddle",           False),
                    ui.input_checkbox("home",       "Possession = home",   True),
                    ui.input_select("season", "Season",
                                    SEASON_CHOICES, selected="2024"),
                    ui.input_select("playcaller",                                               # NEW
                                    "Playcaller",                                               # NEW
                                    choices={"Andy Reid": "Andy Reid"},  # populated reactively # NEW
                                    selected="Andy Reid",                                       # NEW
                                    ),   
                    ui.hr(),
                    ui.input_checkbox("compare_eras",
                                      "Compare across kickoff eras",
                                      False),
                    width=290,
                ),
                ui.div(ui.output_ui("situation_ui"), style="padding:.5rem;"),
            ),
        ),
        # ---- Tab 2: Decision Heatmap ----
        ui.nav_panel(
            "Decision Heatmap",
            ui.div(
                ui.div(
                    ui.row(
                        ui.column(3, ui.input_select(
                            "hm_season", "Kickoff era",
                            {"2024": "Dynamic kickoff (2023+)",
                             "2019": "Traditional kickoff (pre-2023)"},
                            selected="2024",
                        )),
                        ui.column(3, ui.input_slider(
                            "hm_score_diff", "Score differential",
                            -21, 21, 0, step=1,
                        )),
                        ui.column(2, ui.input_select(
                            "hm_qtr", "Quarter",
                            {"1":"Q1","2":"Q2","3":"Q3","4":"Q4"}, selected="3",
                        )),
                        ui.column(2, ui.input_slider(
                            "hm_pos_to", "Off TOs", 0, 3, 2,
                        )),
                        ui.column(2, ui.input_slider(
                            "hm_def_to", "Def TOs", 0, 3, 2,
                        )),
                    ),
                    style="margin-bottom:1rem;",
                ),
                ui.output_plot("heatmap_plot", height="460px"),
                style="padding:1rem;",
            ),
        ),
        # ---- Tab 3: Playcaller Report ----
        ui.nav_panel(
            "Playcaller Report",
            ui.div(ui.output_ui("playcaller_ui"), style="padding:1rem;"),
        ),
        # ---- Tab 4: Audit ----
        ui.nav_panel(
            "Audit",
            ui.div(ui.output_ui("audit_ui"), style="padding:1rem;"),
        ),
        # ---- Tab 5: Model Summary ----
        ui.nav_panel(
            "Model Summary",
            ui.div(ui.output_ui("summary_ui"), style="padding:1rem;"),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
def server(input, output, session):

    @reactive.calc
    def state():
        return load_everything()
    @reactive.effect                                                # NEW
    def _populate_playcaller_dropdown():                            # NEW
        s = state()                                                 # NEW
        if not s.get("ready"):                                      # NEW
            return                                                  # NEW
        pc_names = sorted(                                          # NEW
            c.replace("playcaller_", "")                            # NEW
            for c in s["playcaller_cols"]                           # NEW
        )                                                           # NEW
        default = "Andy Reid" if "Andy Reid" in pc_names else (     # NEW
            pc_names[0] if pc_names else None                       # NEW
        )                                                           # NEW
        ui.update_select(                                           # NEW
            "playcaller", choices=pc_names, selected=default        # NEW
        )                                                           # NEW

    def _situation_kwargs(season_override: int | None = None) -> dict:
        gs = input.game_secs()
        return dict(
            yardline=input.yardline(), ydstogo=input.ydstogo(),
            score_diff=input.score_diff(), qtr=int(input.qtr()),
            game_secs=gs, half_secs=min(gs, 1800),
            pos_to=input.pos_to(), def_to=input.def_to(),
            goal_to_go=input.goal_to_go(), shotgun=input.shotgun(),
            no_huddle=input.no_huddle(), home=input.home(),
            season=season_override if season_override is not None
                                    else int(input.season()),
            playcaller=input.playcaller(),
        )

    @reactive.calc
    def prediction():
        s = state()
        if not s.get("ready"):
            return None
        return predict_one(
            s["features"], s["pres_model"], s["pred_model"],
            s["le_pres"], s["le_pred"], **_situation_kwargs(),
        )

    @reactive.calc
    def prediction_traditional():
        """Same situation, but in a pre-2023 season (for era comparison)."""
        s = state()
        if not s.get("ready"):
            return None
        return predict_one(
            s["features"], s["pres_model"], s["pred_model"],
            s["le_pres"], s["le_pred"], **_situation_kwargs(season_override=2019),
        )

    @reactive.calc
    def prediction_dynamic():
        s = state()
        if not s.get("ready"):
            return None
        return predict_one(
            s["features"], s["pres_model"], s["pred_model"],
            s["le_pres"], s["le_pred"], **_situation_kwargs(season_override=2024),
        )

    # ---- Tab 1 ----
    @output
    @render.ui
    def situation_ui():
        s = state()
        if not s.get("ready"):
            msg = s.get("error", "Models loading — this takes ~60 s on first visit…")
            return ui.HTML(f'<div class="loading-box">{msg}</div>')

        fp = field_position_label(input.yardline())
        situation_cards = (
            f'<div class="four-col">'
            f'  {metric_card(fp,                          "Field position")}'
            f'  {metric_card(input.ydstogo(),             "Yards to go")}'
            f'  {metric_card(f"{input.score_diff():+d}",  "Score differential")}'
            f'  {metric_card(f"Q{input.qtr()}",           "Quarter")}'
            f'</div>'
        )

        if input.compare_eras():
            # Side-by-side era comparison — the kickoff-rule story
            t = prediction_traditional()
            d = prediction_dynamic()
            note = ""
            if t["optimal"] != d["optimal"]:
                note = (
                    f'<div class="alert-amber" style="margin-top:1rem;">'
                    f'⚠ Optimal call shifts across eras — '
                    f'{LABEL_MAP[t["optimal"]]} (traditional) → '
                    f'{LABEL_MAP[d["optimal"]]} (dynamic)'
                    f'</div>'
                )
            return ui.HTML(f"""
                {situation_cards}
                <div class="era-compare">
                  <div class="card traditional">
                    <div class="card-title">Traditional kickoff era (pre-2023)</div>
                    {decision_badge(t['optimal'], large=True)}
                    {prob_bars(t['pres_proba'])}
                  </div>
                  <div class="card dynamic">
                    <div class="card-title">Dynamic kickoff era (2023+)</div>
                    {decision_badge(d['optimal'], large=True)}
                    {prob_bars(d['pres_proba'])}
                  </div>
                </div>
                {note}
            """)

        # Default view — prescriptive vs predictive for the selected season
        p = prediction()
        match = p["optimal"] == p["predicted"]
        if match:
            box = alert_box("green", f"✓ Both models agree — {LABEL_MAP[p['optimal']]}")
        else:
            box = alert_box("amber",
                f"⚠ Divergence: EPA-optimal = {LABEL_MAP[p['optimal']]}  ·  "
                f"Coach likely to call = {LABEL_MAP[p['predicted']]}")

        return ui.HTML(f"""
            {situation_cards}
            <div class="two-col">
              <div class="card">
                <div class="card-title">Prescriptive — EPA-optimal</div>
                {decision_badge(p['optimal'], large=True)}
                {prob_bars(p['pres_proba'])}
              </div>
              <div class="card">
                <div class="card-title">Predictive — coach behavior</div>
                {decision_badge(p['predicted'], large=True)}
                {prob_bars(p['pred_proba'])}
              </div>
            </div>
            {box}
        """)

    # ---- Tab 2 ----
    @output
    @render.plot
    def heatmap_plot():
        s = state()
        if not s.get("ready"):
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Data not loaded", ha="center", va="center",
                    transform=ax.transAxes, fontsize=13, color="#6b7280")
            ax.axis("off")
            return fig

        season = int(input.hm_season())
        grid = get_heatmap(
            season    = season,
            score_diff= input.hm_score_diff(),
            qtr       = int(input.hm_qtr()),
            pos_to    = input.hm_pos_to(),
            def_to    = input.hm_def_to(),
        )
        # Order: go=0 (green), field_goal=1 (orange), punt=2 (blue)
        grid["dn"] = grid["optimal"].map({"go": 0, "field_goal": 1, "punt": 2})
        pivot = grid.pivot(index="ydstogo", columns="yardline_100", values="dn")

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.imshow(
            pivot.values,
            cmap=ListedColormap([COLORS["go"], COLORS["field_goal"], COLORS["punt"]]),
            aspect="auto", vmin=-0.5, vmax=2.5, origin="upper",
        )
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(int(v)) for v in pivot.columns], rotation=45, fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_xlabel("Yards from opponent end zone", fontsize=11)
        ax.set_ylabel("Yards to go", fontsize=11)

        era = ("Dynamic kickoff era (2023+)" if season >= 2023
               else "Traditional kickoff era (pre-2023)")
        sd  = input.hm_score_diff()
        sd_label = "tied" if sd == 0 else (f"+{sd}" if sd > 0 else str(sd))
        ax.set_title(
            f"Optimal 4th Down Decision — {era}\n"
            f"({sd_label}, Q{input.hm_qtr()}, "
            f"{input.hm_pos_to()}/{input.hm_def_to()} timeouts)",
            fontsize=11, fontweight="bold", pad=12,
        )
        ax.legend(handles=[
            mpatches.Patch(color=COLORS["go"],         label="Go for it"),
            mpatches.Patch(color=COLORS["field_goal"], label="Field goal"),
            mpatches.Patch(color=COLORS["punt"],       label="Punt"),
        ], loc="upper right", fontsize=10, framealpha=0.9)
        ax.tick_params(length=0)
        fig.tight_layout()
        return fig

    # ---- Tab 3 ----
    @output
    @render.ui
    def playcaller_ui():
        s = state()
        if not s.get("ready"):
            return ui.HTML('<div class="loading-box">Data not loaded.</div>')

        stats   = s["pc_stats"]
        lg_avg  = s["league_agree"]
        top25   = stats.nlargest(25, "n_plays").sort_values("actual_agree_rate")

        fig = go.Figure([
            go.Bar(y=top25.index, x=top25["actual_agree_rate"],
                   name="Actual coach", orientation="h",
                   marker_color=COLORS["punt"], opacity=0.85),
            go.Bar(y=top25.index, x=top25["predicted_agree_rate"],
                   name="Predicted by model", orientation="h",
                   marker_color="#f97316", opacity=0.75),
        ])
        fig.add_vline(x=lg_avg, line_dash="dash", line_color=COLORS["punt"],
                      annotation_text=f"League avg {lg_avg:.1%}",
                      annotation_position="top right")
        fig.update_layout(
            barmode="group", height=640,
            title="Playcaller Agreement with EPA-Optimal Decision (top 25 by volume)",
            xaxis=dict(title="Agreement rate", tickformat=".0%", range=[0, 1],
                       gridcolor="#f3f4f6"),
            yaxis=dict(gridcolor="#f3f4f6"),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
            margin=dict(l=140, r=20, t=60, b=40),
            plot_bgcolor="white", paper_bgcolor="white",
        )

        def _table(df, title, color):
            rows = "".join(
                f'<tr style="border-bottom:1px solid #f3f4f6;">'
                f'<td style="padding:6px 10px;">{name}</td>'
                f'<td style="padding:6px 10px;text-align:right;">{int(r["n_plays"])}</td>'
                f'<td style="padding:6px 10px;text-align:right;">{r["actual_agree_rate"]:.1%}</td>'
                f'<td style="padding:6px 10px;text-align:right;color:#6b7280;">'
                f'{r["predicted_agree_rate"]:.1%}</td></tr>'
                for name, r in df.iterrows()
            )
            return (
                f'<div class="card" style="flex:1;min-width:0;">'
                f'  <div class="card-title" style="color:{color};">{title}</div>'
                f'  <table style="width:100%;border-collapse:collapse;font-size:13px;">'
                f'    <thead><tr style="border-bottom:2px solid #e5e7eb;">'
                f'      <th style="text-align:left;padding:6px 10px;">Playcaller</th>'
                f'      <th style="text-align:right;padding:6px 10px;">Plays</th>'
                f'      <th style="text-align:right;padding:6px 10px;">Actual</th>'
                f'      <th style="text-align:right;padding:6px 10px;color:#6b7280;">Predicted</th>'
                f'    </tr></thead><tbody>{rows}</tbody>'
                f'  </table>'
                f'</div>'
            )

        return ui.div(
            ui.HTML(
                f'<div class="card">'
                f'{fig.to_html(include_plotlyjs="cdn", full_html=False)}'
                f'</div>'
            ),
            ui.HTML(
                f'<div style="display:flex;gap:1rem;">'
                f'{_table(stats.nlargest(10, "actual_agree_rate"),  "Most optimal",  "#15803d")}'
                f'{_table(stats.nsmallest(10, "actual_agree_rate"), "Least optimal", "#b45309")}'
                f'</div>'
            ),
        )

    # ---- Tab 4 ----
    @output
    @render.ui
    def audit_ui():
        s = state()
        if not s.get("ready"):
            return ui.HTML('<div class="loading-box">Data not loaded.</div>')

        audit = s["audit"]
        n_total   = len(audit)
        n_correct = int(audit["model_was_right"].sum())
        accuracy  = n_correct / n_total if n_total else 0.0

        # What did coaches actually do when the model said punt and was wrong?
        wrong = audit[audit["model_was_right"] == 0]
        actual_when_wrong = (
            wrong["decision"].value_counts(normalize=True).round(3) * 100
        ).to_dict()
        wrong_summary = ", ".join(
            f"{LABEL_MAP[d]}: {pct:.0f}%"
            for d, pct in sorted(actual_when_wrong.items(), key=lambda x: -x[1])
        ) or "—"

        # Per-playcaller accuracy on these calls
        per_pc = (
            audit.groupby("playcaller")
                 .agg(n=("model_was_right", "size"),
                      correct=("model_was_right", "sum"))
                 .assign(accuracy=lambda d: (d["correct"] / d["n"]).round(3))
                 .query("n >= 5")
                 .sort_values("accuracy", ascending=False)
                 .head(15)
        )
        pc_rows = "".join(
            f'<tr style="border-bottom:1px solid #f3f4f6;">'
            f'<td style="padding:6px 10px;">{name}</td>'
            f'<td style="padding:6px 10px;text-align:right;">{int(r["n"])}</td>'
            f'<td style="padding:6px 10px;text-align:right;">{r["accuracy"]:.1%}</td>'
            f'</tr>'
            for name, r in per_pc.iterrows()
        )

        # Sample plays (most recent season)
        sample = audit[audit["season"] == audit["season"].max()].head(15)
        sample_rows = "".join(
            f'<tr style="border-bottom:1px solid #f9fafb;">'
            f'<td style="padding:5px 10px;">{r["season"]}</td>'
            f'<td style="padding:5px 10px;">{r["playcaller"]}</td>'
            f'<td style="padding:5px 10px;text-align:right;">'
            f'{field_position_label(int(r["yardline_100"]))}</td>'
            f'<td style="padding:5px 10px;text-align:right;">{int(r["ydstogo"])}</td>'
            f'<td style="padding:5px 10px;text-align:right;">{int(r["score_differential"]):+d}</td>'
            f'<td style="padding:5px 10px;">{LABEL_MAP[r["decision"]]}</td>'
            f'<td style="padding:5px 10px;">'
            f'{"✓" if r["model_was_right"] else "✗"}</td>'
            f'</tr>'
            for _, r in sample.iterrows()
        )

        return ui.HTML(f"""
            <div class="card">
              <div class="card-title">Audit — model-predicted punts inside opponent's 40</div>
              <p style="color:#6b7280;font-size:13px;line-height:1.6;">
                When the predictive model thinks a coach will punt deep in opponent
                territory, it's worth asking whether the model is right. These are
                situations where the EPA-optimal call is rarely punt, so a model
                that confidently predicts punt is either capturing real coaching
                conservatism or overfitting noise.
              </p>
            </div>
            <div class="four-col">
              {metric_card(f"{n_total:,}",     "Cases flagged")}
              {metric_card(f"{accuracy:.1%}",  "Model accuracy on flagged")}
              {metric_card(f"{n_correct:,}",   "Times coach actually punted")}
              {metric_card(f"{n_total - n_correct:,}", "Times coach did something else")}
            </div>
            <div class="card">
              <div class="card-title">When the model was wrong, the coach actually:</div>
              <p style="font-size:14px;color:#374151;">{wrong_summary}</p>
            </div>
            <div class="two-col">
              <div class="card">
                <div class="card-title">Per-playcaller accuracy (≥5 flagged)</div>
                <table style="width:100%;border-collapse:collapse;font-size:13px;">
                  <thead><tr style="border-bottom:2px solid #e5e7eb;">
                    <th style="text-align:left;padding:6px 10px;">Playcaller</th>
                    <th style="text-align:right;padding:6px 10px;">N</th>
                    <th style="text-align:right;padding:6px 10px;">Acc</th>
                  </tr></thead>
                  <tbody>{pc_rows}</tbody>
                </table>
              </div>
              <div class="card">
                <div class="card-title">Sample plays — most recent season</div>
                <table style="width:100%;border-collapse:collapse;font-size:12px;">
                  <thead><tr style="border-bottom:2px solid #e5e7eb;">
                    <th style="text-align:left;padding:5px 10px;">Yr</th>
                    <th style="text-align:left;padding:5px 10px;">Playcaller</th>
                    <th style="text-align:right;padding:5px 10px;">Field</th>
                    <th style="text-align:right;padding:5px 10px;">YTG</th>
                    <th style="text-align:right;padding:5px 10px;">Sc</th>
                    <th style="text-align:left;padding:5px 10px;">Coach did</th>
                    <th style="text-align:left;padding:5px 10px;">✓?</th>
                  </tr></thead>
                  <tbody>{sample_rows}</tbody>
                </table>
              </div>
            </div>
        """)

    # ---- Tab 5 ----
    @output
    @render.ui
    def summary_ui():
        s = state()
        if not s.get("ready"):
            return ui.HTML('<div class="loading-box">Data not loaded.</div>')

        def param_rows(p):
            return "".join(
                f'<tr style="border-bottom:1px solid #f9fafb;">'
                f'<td style="padding:5px 12px;color:#6b7280;">{k}</td>'
                f'<td style="padding:5px 12px;font-weight:500;font-family:monospace;">'
                f'{v if not isinstance(v, float) else f"{v:.4g}"}</td>'
                f'</tr>'
                for k, v in p.items()
            )

        src = "JSON files" if s["params_from_file"] else "hardcoded fallback defaults"

        # Yearly trend mini-summary
        yearly = s["yearly"]
        yearly_rows = "".join(
            f'<tr style="border-bottom:1px solid #f9fafb;">'
            f'<td style="padding:5px 12px;">{int(yr)}</td>'
            f'<td style="padding:5px 12px;text-align:right;">{int(r["n_plays"]):,}</td>'
            f'<td style="padding:5px 12px;text-align:right;">{r["actual_agree_rate"]:.1%}</td>'
            f'<td style="padding:5px 12px;text-align:right;color:#6b7280;">'
            f'{r["predicted_agree_rate"]:.1%}</td>'
            f'</tr>'
            for yr, r in yearly.iterrows()
        )

        return ui.HTML(f"""
            <div class="four-col">
              {metric_card(f"{s['pres_acc']:.1%}",     "Prescriptive accuracy")}
              {metric_card(f"{s['pred_acc']:.1%}",     "Predictive accuracy")}
              {metric_card(f"{s['league_agree']:.1%}", "League agree w/ optimal")}
              {metric_card(f"{s['n_divergent']:,}",    "Divergent plays (2024)")}
            </div>
            <div class="two-col">
              <div class="card">
                <div class="card-title">Prescriptive XGBoost params</div>
                <table style="width:100%;border-collapse:collapse;font-size:13px;">
                  {param_rows(s['pres_params'])}
                </table>
              </div>
              <div class="card">
                <div class="card-title">Predictive XGBoost params</div>
                <table style="width:100%;border-collapse:collapse;font-size:13px;">
                  {param_rows(s['pred_params'])}
                </table>
              </div>
            </div>
            <div class="card">
              <div class="card-title">Year-by-year agreement</div>
              <table style="width:100%;border-collapse:collapse;font-size:13px;">
                <thead><tr style="border-bottom:2px solid #e5e7eb;">
                  <th style="text-align:left;padding:5px 12px;">Season</th>
                  <th style="text-align:right;padding:5px 12px;">Plays</th>
                  <th style="text-align:right;padding:5px 12px;">Actual agree</th>
                  <th style="text-align:right;padding:5px 12px;color:#6b7280;">Predicted agree</th>
                </tr></thead>
                <tbody>{yearly_rows}</tbody>
              </table>
            </div>
            <div class="card" style="font-size:13px;color:#6b7280;line-height:1.9;">
              Train: 2018–2023 &nbsp;·&nbsp; Test: 2024 &nbsp;·&nbsp;
              Temporal split prevents leakage &nbsp;·&nbsp; Params source: {src}
            </div>
        """)


app = App(app_ui, server)
