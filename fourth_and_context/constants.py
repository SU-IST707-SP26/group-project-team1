"""Shared constants for 4th & Context dashboard.

Keeping all magic strings, feature lists, and visual config here avoids
the drift problem that bit us when feature lists were duplicated across
prescriptive and predictive notebooks.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file's location, not CWD)
# ---------------------------------------------------------------------------
ROOT             = Path(__file__).parent
DATA_PATH        = ROOT.parent / "encoded_fourth_downs.csv.gz"
PRES_PARAMS_PATH = ROOT / "params" / "best_xgb_params.json"
PRED_PARAMS_PATH = ROOT / "params" / "best_predictive_xgb_params.json"
CACHE_DIR        = ROOT / "models_cache"
CACHE_PATH       = CACHE_DIR / "models.pkl"

# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
TEST_SEASON = 2024
RANDOM_STATE = 42

PRE_SNAP_FEATURES = [
    "yardline_100", "ydstogo", "score_differential",
    "game_seconds_remaining", "half_seconds_remaining", "qtr",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "goal_to_go", "shotgun", "no_huddle", "home_is_posteam",
]

PREGAME_EPA_FEATURES = [
    "no_score_prob", "opp_fg_prob", "opp_td_prob", "fg_prob", "td_prob",
]

# Bucketing for prescriptive label generation
GROUP_COLS = ["ydstogo_bin", "yardline_bin", "score_diff_bin", "kickoff_era"]

YDSTOGO_BINS    = [0, 1, 3, 6, 10, 99]
YDSTOGO_LABELS  = ["1", "2-3", "4-6", "7-10", "10+"]

YARDLINE_BINS   = [0, 20, 40, 60, 80, 100]
YARDLINE_LABELS = ["opp_red_zone", "opp_40", "midfield", "own_40", "own_end"]

SCORE_DIFF_BINS   = [-100, -14, -7, -3, 3, 7, 14, 100]
SCORE_DIFF_LABELS = ["down_14+", "down_8-14", "down_1-7",
                      "close", "up_1-7", "up_8-14", "up_14+"]

# Fallback XGBoost params (used only if JSON files are missing)
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
# Visual
# ---------------------------------------------------------------------------
DECISIONS = ["go", "punt", "field_goal"]

LABEL_MAP = {
    "go":         "Go For It",
    "punt":       "Punt",
    "field_goal": "Field Goal",
}

COLORS = {
    "go":         "#639922",  # green
    "punt":       "#185FA5",  # blue
    "field_goal": "#BA7517",  # orange
}

BADGE_CLS = {
    "go":         "badge-go",
    "punt":       "badge-punt",
    "field_goal": "badge-fg",
}

# Prescriptive (EPA-optimal) cache key prefix for heatmap
HEATMAP_KEY_PREFIX = "hm"

# Era for the kickoff rule change
DYNAMIC_ERA_FIRST_SEASON = 2023
