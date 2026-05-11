The conceptual framing of this project — splitting the problem into a prescriptive model ("what should the coach do?") and a predictive model ("what will the coach do?"), then using the gap between them as the actionable insight — is great. The decision to explicitly bucket on `is_dynamic_era` (pre- vs. post-2023 kickoff rule) is also really thoughtful. You've also done some pretty nice data engineering (the manually-assembled playcaller dataset is non-trivial), were careful with leakage (385 -> 113 features by removing post-snap outcomes), and delivered a dashboard prototype.

That said, several of your highlighted results oversell what the methodology actually supports, as follows.

**The 85.8% prescriptive accuracy is partly circular.** The prescriptive target is a deterministic function of the bucket assignment, and the bucket is defined by `yardline_100`, `ydstogo`, `score_differential`, and `is_dynamic_era` — all of which are also model features. So the classifier is essentially being asked to approximate a discrete lookup table from features that define the lookup keys. The 85.8% accuracy isn't a measure of how well the model "learns football" — it's a measure of how well a classifier can approximate the bucketing function. So, a more realistic framing might be: "we generate optimal labels from bucket EPA means, and a classifier recovers the bucket -> label mapping at 85.8% accuracy." 

**The 98% predictive accuracy is mostly aggregate of easy cases.** Most 4th down decisions are obvious (4th and 12 from your own 20 is always a punt). With 55.8% punts and a large majority of plays being clear-cut, a model that handles all the easy cases correctly and 50/50 on the hard ones still scores in the high 90s in aggregate. The 98% number tells you the model handles the obvious cases, not that it captures decision-making nuance. Your framing — "validates that the model captures the right *reasons* for each coach's decisions" — overstates what aggregate accuracy can tell you. Per-situation accuracy on borderline cases (4th and 2 from midfield, 4th and 4 from the opponent 35, etc.) would have been the informative metric.

**No uncertainty quantification — which is strange given your citations.** You cite Brill, Yurko & Wyner (2025), which is specifically titled "Have Some Humility: A Statistical View of Fourth-Down Decision Making" and argues explicitly for uncertainty quantification in 4th down models. Your headline numbers — 1,752 EPA on the table, 57.2% agreement, 1.08 EPA per divergent play — are reported as point estimates with no confidence intervals, no bootstrap, no sensitivity analysis. The paper you cite would tell you not to do this.

**The 1.08 EPA per divergent play is potentially inflated by selection bias.** Mean EPA per NFL play is approximately 0 by construction. 1.08 per play implies each divergent call costs nearly a quarter of a touchdown on average — possible, but worth scrutinizing. The comparison is presumably "actual outcome EPA" vs. "bucket-mean optimal EPA," which conflates decision quality with outcome luck. Divergent plays may concentrate in high-leverage moments where outcome variance is highest, which inflates the apparent gap. Bootstrap CIs on this number would have shown how stable it is.

**Bucket sparsity is never quantified.** You note that "plays in buckets with fewer than a minimum count threshold were excluded" — but you don't say what the threshold was, how many buckets remain, or what fraction of plays are in sparse vs. dense buckets. The Limitations section flags the issue but doesn't address it empirically. The reliability of the prescriptive labels rests on bucket density, and the reader can't audit it.

**Two playcaller findings warrant scrutiny.** Kyle Shanahan ranks lowest in agreement with optimal, characterized as "consistent with his reputation for conservative playcalling." But Shanahan's league-wide reputation is as one of the *more* aggressive 4th-down playcallers. Brian Johnson ranks highest at 70.8% agreement, but Johnson was Philadelphia's OC in 2023 and was fired largely because of his playcalling. Either your metric is measuring something orthogonal to public/expert perception (which is interesting but should be framed that way), or there's noise in the playcaller-level findings. The Shanahan/Johnson surprise should have triggered more scrutiny of what the metric is actually capturing.

## Strengths

- **The prescriptive/predictive decomposition is great** The gap between them is the actionable insight, and that's a clean way to operationalize "where could coaches improve."
- **Dynamic kickoff rule treated as a domain shift.** Explicit `is_dynamic_era` bucketing in the prescriptive model and as a feature in the predictive model is also insightful feature engineering and a great way to bring domain knowledge to the modeling process.
- **Temporal train/test split is correct and explicitly justified.** "A random split would leak future game outcomes into training" — very good.
- **Data leakage discipline is excellent.** 385 -> 113 features by removing post-snap outcome columns is the right discipline, well-articulated.
- **Custom playcaller dataset.** Manually assembling 91 playcallers x weekly assignments across 7 seasons is great data engineering.
- **Bucketing for prescriptive labels avoids per-play overfitting.** Using EPA bucket means rather than individual play outcomes is the conservative and correct choice.
- **Heatmap recovers football common sense.** Punt deep, FG in red zone, go-for-it near midfield on short — the model learning this without being hand-coded to is a good validation of structural signal.
- **Honest about the headline-rate limitation.** "57.2% agreement should not be interpreted as evidence that coaches are making poor decisions 43% of the time" is the right caveat, even if it doesn't fully propagate to the other numbers.

## Weaknesses

- **The 85.8% prescriptive accuracy is partly tautological** (see above).
- **The 98% predictive accuracy is mostly easy-case aggregate** (see above).
- **No uncertainty quantification on any of the headline numbers**, despite citing a paper specifically about uncertainty in 4th down models.
- **The 1.08 EPA per divergent play is potentially inflated** by conflating decision quality with outcome luck.
- **Bucket sparsity is acknowledged but never quantified** — what fraction of plays are in marginal buckets?
- **Playcaller-level findings (Shanahan, Johnson) warrant more scrutiny.** Surprises that contradict public perception should trigger investigation, not pass through as confirmation.
- **Cluster analysis is mentioned in Methods but doesn't appear in Results.** Generally, if you're going to discuss a method, it should show up in results.  What happened to this analysis?
- **EPA-optimal labels are themselves model outputs.** nflFastR's EPA model is the foundation; if it has biases, they propagate. Worth acknowledging that your "optimal" is "optimal under nflFastR's EPA model."
- **No baseline comparisons.** What's the accuracy of "always punt" on the prescriptive task? Of "follow ESPN's broadcast recommendation"? Natural baselines exist and would have anchored the 85.8% number.
- **The Google Drive link is "to be added prior to submission"** — a small thing, but indicates the report wasn't fully proofed.

## Closing thought

The conceptual decomposition of the problem is excellent and the dynamic-kickoff angle is sophisticated. The dashboard prototype is a nice addition. But I think you've presented your headline numbers — particularly the prescriptive accuracy, predictive accuracy, and the EPA gap — with more confidence than the methodology supports. Adding bootstrap CIs and reporting per-situation (not just aggregate) accuracy would have addressed most of this. Investigating the Shanahan/Johnson surprises rather than accepting them would have made the playcaller-level claims more durable. Strong project overall.

**Score 28/30**


---

## Final Project Grade
**Note**: Midterm report was submitted late. The 5/10 midterm score reflects the late submission.
| Assessment Item | Fred Gullo | Jared Weber | Gavin Stein |
|---|---|---|---|
| **Proposal (5 pts)** | 5 | 5 | 5 |
| **Midterm Report (10 pts)** | 5 | 5 | 5 |
| **Final Presentation (5 pts)** | 5 | 5 | 5 |
| **Final Report (30 pts)** | 28 | 28 | 28 |
| **Weekly Updates (30 pts)** | 30 | 28 | 28 |
| **Total (80 pts)** | **73** | **71** | **71** |
