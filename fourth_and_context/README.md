# 4th & Context — Decision Support Dashboard

NFL 4th-down decision support, built on top of the prescriptive (EPA-optimal)
and predictive (coach-behavior) XGBoost models from the `epa_*_model` notebooks.

IST 707 · Gullo, Weber, Stein

## Run

```bash
pip install -r requirements.txt
python -m shiny run app.py --reload
```

Then open http://127.0.0.1:8000 in a browser.

The first launch trains both XGBoost models (~30–60 s) and pickles them to
`models_cache/`. Every subsequent launch loads from disk in <1 s.

To force retraining (e.g. after editing the JSON params), delete the
`models_cache/` directory.

## File layout

```
fourth_and_context/
├── app.py              # Shiny UI + reactive logic
├── constants.py        # Feature lists, bins, colors, paths
├── data_loader.py      # CSV load, decision mapping, feature engineering
├── models.py           # Train + cache both XGBoost models
├── analytics.py        # Playcaller stats, heatmap grid, audit cases
├── ui_components.py    # Reusable HTML helpers + CSS
├── data/
│   └── encoded_fourth_downs.csv.gz
├── params/
│   ├── best_xgb_params.json            # prescriptive
│   └── best_predictive_xgb_params.json
├── models_cache/                        # auto-created on first run
│   └── models.pkl
├── requirements.txt
└── README.md
```

## Tabs

1. **Situation Evaluator** — Slider-driven recommendation for any 4th-down
   situation. Shows prescriptive (EPA-optimal) and predictive (coach behavior)
   models side-by-side with probability bars. Toggle "Compare across kickoff
   eras" to see the same situation evaluated under traditional vs dynamic
   kickoff rules — the kickoff-rule story.
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

## Notes

- **Train/test split is temporal.** Train on 2018–2023, test on 2024. Random
  splits would leak playcaller info across seasons.
- **The two models share a feature space.** Prescriptive and predictive use
  the same 113 features, just different targets (EPA-optimal label vs actual
  coach decision). The shared feature list is asserted in `data_loader`.
- **Hyperparameters live in `params/`.** Edit the JSON files and delete
  `models_cache/` to retrain.
