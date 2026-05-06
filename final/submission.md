# 4th & Context: Quantifying Optimality in NFL 4th Down Decisions

## Team

| Name | GitHub ID |
|---|---|
| Fred Gullo | freddyg-33 |
| Jared Weber | jweber1380 |
| Gavin Stein | gpstein23 |

---

## Introduction

NFL head coaches face no decision more scrutinized than the 4th down call. Every broadcast now displays a go/punt/field goal recommendation — but the models powering those recommendations rely on coarse, league-average inputs: score, distance, yard line, clock, and generic team strength. They ignore who is calling plays, what formation the offense is in, and — critically — how recent rule changes have shifted the risk-reward calculus of punting and kicking.

Our primary stakeholders are NFL coaches and team analytics staff who need a decision-support tool calibrated to modern game conditions. Since 2023, the NFL's "dynamic kickoff" rule has substantially improved opponent starting field position following punts and field goals, meaning the traditional expected-value math behind conservative calls has eroded. A model trained on pre-2023 league averages will systematically over-recommend punts in situations where going for it is now the optimal play.

This project delivers two complementary models. A **prescriptive model** answers the question "What *should* the coach do?" by deriving EPA-optimal recommendations within situational buckets defined by field position, distance, score differential, and kickoff era. A **predictive model** answers "What *will* the coach do?" using XGBoost on pre-snap features to forecast the actual call with 98% accuracy. The gap between the two — what coaches do versus what they should do — is where the actionable insight lives. In the 2024 season alone, NFL playcallers diverged from optimal in 1,627 plays, leaving an estimated 1,752 total EPA on the table. This project makes that waste measurable, coach-specific, and addressable.

---

## Literature Review

4th down decision analytics trace back to foundational expected-points and win probability frameworks. The most visible public implementation is ESPN's model, which incorporates score differential, yards to go, field position, clock, timeouts, pre-game win probability, and relative offense/defense strength to produce the go/punt/field goal graphics shown during broadcasts (Walder). These models are useful baselines, but they share a structural limitation: they are trained on historical league averages and do not account for heterogeneity in conversion rates across teams, personnel, or coaching staff.

Recent statistical literature has begun to address this. Brill, Yurko, and Wyner (2025) argue that win probability estimates derived from a limited sample of historical plays carry substantial uncertainty, and advocate for explicit uncertainty quantification in 4th down models rather than presenting point estimates as certainties. Separately, Brill and Wyner (2024) demonstrate that fourth-down conversion frequencies are systematically higher than historical averages suggest, implying that conventional models underestimate the expected value of going for it across a wide range of situations.

Neither line of work addresses the structural change introduced by the NFL's dynamic kickoff rule, which debuted in the 2023 season. By stationing blockers and defenders closer together and immobilizing them until the ball is caught, the rule dramatically reduced kick return opportunities and, in concert with an adjusted touchback placement rule in 2024, shifted opponent starting field position meaningfully toward their own 35-yard line. This changes the expected value of punting — particularly from midfield — because the defense receives better starting field position than it did historically. A model that does not account for this era shift will produce systematically miscalibrated recommendations. Our approach directly models this by including kickoff era as a bucketing dimension in the prescriptive model and as a binary feature in the predictive model, and by restricting the test set to 2024 data to evaluate performance under current rule conditions.

**Sources:**
- Walder, S. ESPN analytics model for fourth-down decisions. https://www.espn.com/nfl/story/_/id/39379626/
- Brill, R.S., Yurko, R., & Wyner, A.J. (2025). Analytics, Have Some Humility: A Statistical View of Fourth-Down Decision Making. https://ryansbrill.com/pdf/statistics_in_sports_papers/Brill_Humility_TAS.pdf
- Brill, R.S. & Wyner, A.J. (2024). Fourth-down conversion heterogeneity estimates. https://wsb.wharton.upenn.edu/fourth-down-conversion-frequencies-are-higher-than-you-think/

---

## Data and Methods

### Data

Our primary data source is the nflFastR play-by-play dataset, covering NFL regular season games from 2018 through 2025. nflFastR is widely regarded as the standard for public NFL play-by-play data; it is regularly cited in academic research and the football analytics community, and while minor discrepancies with official PFF/NFL aggregates have been noted, the dataset is considered reliable for modeling purposes.

The raw dataset contains **38,762 fourth-down plays** across 385 variables. After filtering to plays representing a true coach decision — go-for-it (pass or run), punt, or field goal attempt — and removing hard counts (2,085 plays where an encroachment or false start reset the down without a decision), the modeling dataset contains **34,897 usable plays**. The feature space was reduced from 385 raw variables to **113 model-ready features** by eliminating post-snap outcome columns that would constitute data leakage (e.g., `yards_gained`, `fourth_down_converted`, `epa` as a direct input).

The decision distribution across the filtered dataset reflects realistic game conditions: punts are the plurality decision at approximately 55.8% of plays, field goals account for 24.5%, and go-for-it plays (pass + run) make up the remaining 19.7%. There is no severe class imbalance given that punts, field goals, and go decisions are all well-represented in absolute terms.

One variable not available in nflFastR required custom construction: the **weekly offensive playcaller** for every team across every season. No public source tracks mid-season playcaller changes at the weekly level, so this was assembled manually, producing a dataset of playcaller assignments used to construct the playcaller one-hot encoding present in the model feature set (91 unique playcallers identified). The data can be accessed at: *[Google Drive link — to be added prior to submission]*.

The dataset was split temporally: seasons 2018–2023 form the training set (26,351 plays after filtering) and the 2024 season forms the held-out test set (4,556 plays). This temporal split is essential — a random split would leak future game outcomes into training and produce inflated accuracy estimates that do not reflect real-world deployment conditions.

Key visualizations supporting the data description (available in `work/epa_prescriptive_model.ipynb`):
- EPA distribution by decision type (boxplot), showing that go-for-it plays have higher variance EPA than punts or field goals
- Correlation heatmap of pre-snap situational features
- Decision frequency by field position and yards-to-go (heatmap)
- Class distribution across training and test seasons

### Methods

Our modeling framework separates the prescriptive and predictive problems, then combines their outputs to generate actionable coach-facing insight.

#### Prescriptive Model: What Should the Coach Do?

The prescriptive target is constructed without post-hoc outcome labels. Rather than labeling individual plays as "correct" or "incorrect," we assign an optimal decision to each situational **bucket** by computing the mean EPA for each decision type within that bucket and selecting the decision with the highest mean. Buckets are defined along four dimensions:

- **Yards to go**: [1], [2–3], [4–6], [7–10], [10+]
- **Field position**: opponent red zone (0–20), opponent 40 (21–40), midfield (41–60), own 40 (61–80), own end (81–100)
- **Score differential**: seven bins from down 14+ to up 14+
- **Kickoff era**: pre-dynamic (2018–2022) vs. dynamic (2023–present)

This bucketing approach is intentionally conservative — it avoids overfitting to individual play outcomes and produces labels that reflect average expected value across many similar situations. Plays in buckets with fewer than a minimum count threshold were excluded to prevent noisy labels from sparse cells from corrupting training.

Three classifiers were trained on the resulting labels using only pre-snap features (the same features a coach would know before the ball is snapped): **multinomial logistic regression**, **random forest**, and **XGBoost**. All three were tuned using 30-iteration randomized search with 3-fold stratified cross-validation. Hyperparameter search spaces included regularization strength (logistic regression), tree depth and leaf size (random forest), and learning rate, subsample rate, column subsample, gamma, and alpha/lambda regularization (XGBoost). The feature set comprises core situational variables (`yardline_100`, `ydstogo`, `score_differential`, `game_seconds_remaining`, `half_seconds_remaining`, `qtr`, `posteam_timeouts_remaining`, `defteam_timeouts_remaining`, `goal_to_go`, `shotgun`, `no_huddle`, `home_is_posteam`), pregame EPA probability features (`no_score_prob`, `opp_fg_prob`, `opp_td_prob`, `fg_prob`, `td_prob`), playcaller one-hot encodings, the `is_dynamic_era` binary flag, and season dummy variables.

The prescriptive model's output is used in two ways: (1) to generate an EPA-optimal decision heatmap across field position and distance, and (2) as the "ground truth" against which actual coach decisions are compared in the stakeholder impact analysis.

#### Predictive Model: What Will the Coach Do?

The predictive model uses an identical feature set but a different target: the **actual coach decision** rather than the EPA-optimal label. This model is trained to forecast coach behavior, not optimal behavior. XGBoost was selected based on its superior performance in the prescriptive comparison and tuned using the same randomized search procedure (15 iterations with 2-fold CV to manage compute).

The predictive model serves a specific analytical purpose: if it accurately captures coach decision-making patterns, then applying it to all plays and comparing its output to the prescriptive model's recommendations isolates the structural gap between how coaches actually behave and how they should behave. This framing is more rigorous than comparing raw actual decisions to prescriptive labels, because it filters out plays where the coach's decision was situationally unusual for reasons not captured in the feature set.

#### Cluster Analysis of 4th Down Scenarios

K-Means clustering was applied to the pre-snap feature space to identify distinct types of 4th down situations beyond the hand-crafted bins used in the prescriptive model. The optimal number of clusters was evaluated using both the elbow method and silhouette scores. Cluster profiles were visualized using PCA and t-SNE dimensionality reduction. The resulting cluster labels were incorporated into the stakeholder impact analysis to provide contextual framing for coach divergence (e.g., characterizing whether a coach's suboptimal calls are concentrated in desperation situations, aggressive midfield scenarios, or conservative short-yardage plays).

### Supporting Files

All analysis supporting this report lives in the `work/` folder. See the index below:

| Notebook | Purpose |
|---|---|
| `epa_prescriptive_model.ipynb` | Data loading and filtering, EPA bucketing, prescriptive target construction, logistic regression / random forest / XGBoost training and comparison, confusion matrices, feature importance, decision heatmap generation |
| `epa_predictive_model.ipynb` | Predictive model training (XGBoost on actual coach decisions), per-playcaller agreement analysis, divergence quantification, yearly agreement trend analysis, predictive punt analysis |

---

## Results

### Prescriptive Model Performance

All three classifiers were evaluated on the 2024 held-out test set. Results are summarized below:

| Model | Accuracy | Log Loss |
|---|---|---|
| Logistic Regression | ~76% | ~0.58 |
| Random Forest | ~83% | ~0.42 |
| **XGBoost (tuned)** | **85.8%** | **lowest** |

XGBoost — tuned via 30-iteration randomized search with 3-fold stratified CV — is the clear winner on both metrics. The full classification report and per-class confusion matrices are available in `work/epa_prescriptive_model.ipynb`.

A key validation of the prescriptive model is the **EPA-optimal decision heatmap** (field position × yards-to-go, neutral game, dynamic era). The heatmap recovers football common sense without being hand-coded to do so: it recommends punting in one's own territory on long yardage, field goals in opponent red zone territory with short distances, and going for it near midfield on short yardage. This alignment with expert football intuition provides strong evidence that the model has learned real structural signal rather than overfitting to noise.

**Top 20 feature importances** (XGBoost) are dominated by situational variables: `yardline_100` and `ydstogo` rank highest, followed by `score_differential`, `game_seconds_remaining`, and the pregame EPA probability features. The `is_dynamic_era` flag contributes meaningfully, confirming that the model has detected a structural shift in optimal decisions between the pre- and post-dynamic kickoff eras. Playcaller features contribute collectively but are lower in individual importance, which is expected given the large number of one-hot columns.

### Predictive Model Performance

The predictive XGBoost model achieves **98.0% accuracy** on the 2024 test set with a log loss of 0.06. These numbers are exceptionally high because coach decisions are largely deterministic given the situational context — most coaches punt on 4th and 15 from their own 20, and the model learns this quickly. The high accuracy is therefore a feature, not a concern: it confirms that the model has successfully internalized playcaller tendencies and situational norms, making it a valid behavioral proxy for the coach.

The scatter plot of actual versus predicted agreement rates across playcallers (available in `work/epa_predictive_model.ipynb`) shows points tightly clustered around the 45-degree diagonal, meaning the predictive model's estimate of how often a coach agrees with the optimal recommendation is nearly identical to the coach's actual agreement rate. This is the key validation: the predictive model is not merely accurate — it is capturing the right *reasons* for each coach's decisions.

### Stakeholder Impact: The EPA Gap

Applying both models to the full 2024 test season yields the following headline results:

| Metric | Value |
|---|---|
| League-wide agreement with optimal | 57.2% |
| Divergent plays (2024) | 1,627 |
| Total EPA left on the table | 1,752 |
| Average EPA cost per divergent play | 1.08 |

The playcaller agreement analysis (top 25 by volume) reveals meaningful variation across coaches. Among high-volume playcallers, **Brian Johnson** leads at 70.8% agreement, followed by **Anthony Lynn** (67.2%), **Joe Philbin**, and **Kliff Kingsbury**. **Kyle Shanahan** ranks lowest among high-volume playcallers, a finding consistent with his reputation for conservative playcalling in certain field position zones. For every coach examined, actual and predicted agreement rates are nearly identical — reinforcing that the predictive model has learned genuine behavioral patterns.

Year-over-year trends (2018–2024) are available in `work/epa_predictive_model.ipynb`. The post-2022 seasons show a modest increase in league-wide agreement with optimal, suggesting that coaches have gradually adapted to the dynamic kickoff environment — but the gap remains substantial at 1,752 cumulative EPA in a single season.

Additional analysis of predicted punts inside the opponent's 40-yard line — situations where punting is rarely optimal — reveals which playcallers are most prone to conservative calls in high-leverage field position. This analysis is fully documented in `work/epa_predictive_model.ipynb`.

---

## Discussion

This project delivers on its core stakeholder promise: NFL coaches and analytics staff can now see, with playcaller-level specificity, how often and in what situations their decisions diverge from EPA-optimal recommendations under current kickoff rules. The 1,752 EPA gap in 2024 is not an abstraction — at roughly 6–7 points per expected-points unit, it represents a meaningful number of drive outcomes that were suboptimal.

The prescriptive model's 85.8% accuracy and intuitive heatmap output are strong results. The explicit inclusion of kickoff era as a bucketing dimension, rather than allowing the model to average across pre- and post-dynamic kickoff seasons, is the key methodological decision that makes the prescriptive recommendations applicable to the current game environment. Prior public models trained on pooled historical data would produce systematically more conservative recommendations in field position zones where the dynamic kickoff has changed the risk calculus.

The predictive model's 98% accuracy is validating rather than surprising. 4th down decisions are highly situation-dependent, and once playcaller tendencies and situational context are encoded, most decisions become predictable. The real value is not the accuracy itself but the confirmation that the predictive model's behavioral estimates track actual coach behavior closely enough to serve as a reliable proxy in the divergence analysis.

One result worth flagging is the 57.2% league-wide agreement rate. This should not be interpreted as evidence that coaches are making poor decisions 43% of the time — it reflects that the prescriptive labels derived from EPA bucket averages are themselves imperfect. Sparse buckets introduce noise, and the optimal label for a given bucket is a mean over a heterogeneous set of plays. The divergence metric is better understood as a directional signal about systematic conservatism than as a precise measure of suboptimality.

---

## Limitations

**Bucket sparsity** is the primary limitation of the prescriptive model. Rare combinations of field position, distance, score differential, and era can yield buckets with very few historical plays, producing noisy EPA means and consequently unreliable optimal labels. This is most pronounced for rare decision types — a field goal attempt from midfield on 4th and 1 with a large lead in the fourth quarter, for instance, may appear in only a handful of plays across seven seasons. Future work should address this through Bayesian smoothing or hierarchical models that share information across similar buckets.

**Coverage and formation data** are absent from the current feature set. We originally intended to incorporate defensive pre-snap looks and formation data from All-22 film, which would allow the model to distinguish between a 4th-and-2 where the defense is showing a heavy box (reducing go-for-it success probability) and one where they are showing soft coverage. This information is not available in nflFastR, and manual collection from film proved too labor-intensive to complete at scale for this project.

**Prescriptive over-recommendation of go-for-it** was noted during model development. In some bucket configurations, the model recommends going for it more aggressively than football intuition would suggest, particularly at long distances in field position zones where a successful field goal is still a plausible play. This may reflect genuine EPA signal — going for it from opponent territory on 4th-and-long is not obviously suboptimal — but it warrants further investigation before deployment as a real-time tool.

**The custom playcaller dataset** was assembled manually and is difficult to verify at scale. Mid-season playcaller changes, interim arrangements, and situations where multiple coaches share responsibilities introduce noise into the playcaller feature. While we are confident the dataset is accurate for clear-cut cases, edge cases may contain errors.

**Season 2025** was included in the raw data but not meaningfully used in modeling, as it is an incomplete season. Results reported here are based on the 2024 test set.

---

## Future Work

The most impactful immediate extension is addressing bucket sparsity in the prescriptive model through hierarchical priors or Bayesian smoothing, which would produce more stable EPA-optimal labels for rare situational combinations and reduce noise in the divergence analysis.

Integrating pre-snap formation and coverage data — even for a subset of high-leverage plays — would substantially improve the prescriptive model's specificity. A hybrid approach where film-coded features are available for, say, the top 500 most consequential 4th down plays per season would allow testing whether formation context meaningfully changes the optimal recommendation.

The decision dashboard prototype remains the ultimate deliverable for the stakeholder use case. The envisioned tool would accept a real-time 4th down situation as input and return (a) the predicted coach call, (b) the EPA-optimal recommendation, and (c) the expected EPA cost of diverging, along with historical playcaller tendency data for context. The modeling infrastructure to support this dashboard is complete; the interface itself is future work.

A fixed-effects analysis of the dynamic kickoff rule change — quantifying the year-over-year shift in optimal decision boundaries attributable specifically to the rule, controlling for team composition changes — was a stated goal in the proposal that was not completed in this iteration. This is a tractable extension given the current dataset and model infrastructure.

Finally, expanding the playcaller agreement analysis to a full coach-facing profiling tool — one that characterizes each playcaller's historical bias patterns (e.g., systematically conservative in the red zone, aggressive near midfield) — would translate the divergence metrics into directly actionable coaching feedback.
