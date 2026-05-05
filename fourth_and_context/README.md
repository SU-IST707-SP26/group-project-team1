# 4th & Context — Decision Support Dashboard

NFL 4th-down decision support, built on top of the prescriptive (EPA-optimal)
and predictive (coach-behavior) XGBoost models from the `epa_*_model` notebooks.

IST 707 · Gullo, Weber, Stein

---

## How to run (in a GitHub Codespace)

Open a Codespace on this repo, then paste this into the terminal:

```bash
cd fourth_and_context && pip install -r requirements.txt && python -m shiny run app.py --host 0.0.0.0 --port 8000
```

That's it. Two things happen:

1. **Dependencies install** (~2 min on first run, instant after that)
2. **The app launches.** First launch trains both models (~30-60 s) and pickles
   them to `models_cache/`. Every subsequent launch loads from disk in <1 s.

When you see `Uvicorn running on http://0.0.0.0:8000`, click the **"Open in
Browser"** popup that appears in the bottom-right. If you miss it, click the
**PORTS** tab next to the terminal, find port 8000, and click the globe icon.

To stop the app: `Ctrl+C` in the terminal.

To force retraining (e.g. after editing the JSON params): delete the
`models_cache/` directory and relaunch.

---

## Tabs

1. **Situation Evaluator** — Slider-driven recommendation for any 4th-down
   situation. Shows prescriptive (EPA-optimal) and predictive (coach behavior)
   models side-by-side with probability bars. Toggle "Compare across kickoff
   eras" to see the same situation evaluated under traditional vs dynamic
   kickoff rules.
2. **Decision Heatmap** — Yardline × yards-to-go grid colored by EPA-optimal
   call. Sidebar inputs (score differential, quarter, timeouts) feed the grid
   instead of being hardcoded to neutral values.
3. **Playcaller Report** — Top-25 playcallers by volume. Bar chart compares
   each coach's actual agreement rate with the EPA-optimal call to the rate
   the predictive model would have produced.
4. **Audit** — Drill-down on cases where the predictive model says a coach
   will punt inside the opponent's 40. Shows model accuracy on these calls,
   per-playcaller breakdown, and what coaches actually did when the model
   was wrong.
5. **Model Summary** — Test accuracies, hyperparameters for both models,
   year-by-year agreement table, training metadata.

---

## File layout

```
group-project-team1/
├── encoded_fourth_downs.csv.gz       ← data lives at the repo root
└── fourth_and_context/
    ├── app.py                         Shiny UI + reactive logic
    ├── constants.py                   Feature lists, bins, colors, paths
    ├── data_loader.py                 CSV load, decision mapping, features
    ├── models.py                      Train + cache both XGBoost models
    ├── analytics.py                   Playcaller stats, heatmaps, audit cases
    ├── ui_components.py               Reusable HTML helpers + CSS
    ├── params/
    │   ├── best_xgb_params.json              prescriptive
    │   └── best_predictive_xgb_params.json   predictive
    ├── models_cache/                  auto-created on first run
    │   └── models.pkl
    ├── requirements.txt
    └── README.md (this file)
```

---

## Notes for the team

- **Train/test split is temporal.** Train on 2018–2023, test on 2024. Random
  splits would leak playcaller info across seasons.
- **Both models share the same 113-feature input space** — just different
  targets (EPA-optimal label vs actual coach decision). The shared feature
  list is asserted at runtime in `data_loader.py`.
- **Hyperparameters live in `params/`.** Edit the JSON files and delete
  `models_cache/` to retrain with new params.
- **Codespace usage:** GitHub gives 60 free hours/month. Codespaces auto-pause
  after 30 min of inactivity, but stop manually at github.com/codespaces when
  you're really done.
