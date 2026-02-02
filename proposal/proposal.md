**IST 707 Group Project Proposal:** *Team1*

*Fred Gullo, Jared Weber, Gavin Stein*

**Topic:** *NFL 4th Down Decision Making // Fixed-Effects Analysis of Dynamic Kickoff Rule Change*

**Title:** *Fourth & Context (still working on it, want to pick something catchier)*


**Introduction**

A big question for NFL coaches is “When do I go for it on 4th down?” Analytics pros working for teams and television networks have created algorithms answering this question in various scenarios. Coaches have a section of their playcall sheet telling them the recommendation in different scenarios. Every broadcast displays their suggestion on 4th downs where going for it is on the table. Currently, the models aren’t very specific to individual scenarios ESPN listed their predictors as: “the score of the game, distance to gain, yard line, clock, timeouts, pregame win probability and the relative strength of the offense and defense on the field” (Walder). We believe that these predictions can be more granular and situation-specific. We want to build a new predictive model that takes many more variables into account, such as pre-snap formations for the offense and defense, coaching tendencies, and in-game effects. 

An additional consideration for us is the year-to-year changes in kickoff rules. The past two seasons have seen the “dynamic kickoff” implemented in all NFL games. The rule change features all blockers and defenders closer together than they traditionally were and stationary until the kick is caught, with the intent of reducing the occurrence of injuries. This initially led to far fewer kick returns happening and the NFL changed the touchback rules (what happens if the ball hits the ground or the returner ‘fair catches’ and gives themselves up) to incentivize returns this past season. A consequence of this is now offenses start with far better field position than in previous seasons, and that changes the risk/reward payoff of going for 4th downs, since getting a field goal and giving the other team the ball at the 35 yard line may not be an optimal outcome of a drive. 

Our predictive model for 4th down scenarios can be utilized by NFL coaches for data-driven decision making that takes more contextual information into account than the models currently available. With the NFL’s kickoff rule changing likely to continue, having a model that can take these changes into account with and accurately suggest 4th down decisions is invaluable to head coaches. 

**Lit Review**

NFL 4th down decision analytical models are built to maximize expected win probability or expected points, lacking the granularity required for a thorough analysis of each unique 4th down situation. Publicly available analytics tools, including those used by broadcast networks, rely on basic variables such as score, distance, yard line, clock, timeouts, pre-game win probability, and relative offense/defense strength to inform 4th down decisions. ESPN’s win probability model incorporates these predictors into its weekly “go/punt/field goal” guidance graphics shown during games (Walder). Recent statistical literature has highlighted the inherent uncertainty in win probability estimates derived from a limited sample of historical games and advocates for uncertainty quantification in future models (Brill, Yurko & Wyner). 

Traditional models based on league averages and basic situation variables may also overlook heterogeneity in conversion success tied to team strength, possessions, and score context (Brill & Wyner). Incorporating additional predictors will be valuable in attempting to build a more robust approach to these 4th down decisions.

Sources:
- Seth Walder, ESPN analytics model for fourth-down decisions
  https://www.espn.com/nfl/story/_/id/39379626/nfl-analytics-models-fourth-graphics-method-decisions-punt-field-goal-go-it?
- Brill, R. S., Yurkko, R., & Wyner, A. J. (2025) — Analytics, Have Some Humility: A Statistical View of Fourth-Down Decision Making
  https://ryansbrill.com/pdf/statistics_in_sports_papers/Brill_Humility_TAS.pdf
- Brill, R. S. & Wyner, A. J. (2024) — Fourth-down conversion heterogeneity estimates
  https://wsb.wharton.upenn.edu/fourth-down-conversion-frequencies-are-higher-than-you-think/
**Data**

We are going to utilize the nflFastR dataset to accomplish our goals. This publicly available dataset contains information at the play level in a very granular fashion. It contains written descriptions of each play, as well as numerous measurements and characteristics. These include the offensive and defensive formations, players involved in the play, and much more. We also plan on purchasing a subscription to the NFL's All 22 film, which will give us access to videos of every play in our dataset (from multiple angles). This will help validate our feature engineering steps and add dynamic elements to our final paper/presentation. 

We will also gather information for playcallers, which will require some Google searching. We'll create a dataset with each team's offensive/defensive/special teams play caller for every week (this sounds daunting, but playcallers rarely change midseason). We need to be sure of which coach is calling plays in every week because midseason changes are often accompanied with a change in tendency and effectiveness of results. 

nflFastR is seen as very reliable by the football analytics community. It is frequently cited in reseach and exploratory projects. Errors have been noted, as the aggregated game statistics occasionally vary slightly compared to the official statistics from PFF/NFL, but the consensus is that nflFastR is a reputable, reliable source of play-by-play data. Upgrading to a paid program would be extremely costly, and we have full faith that nflFastR provides more than enough value to justify using a free option instead.

**Methods**

We construct a supervised machine learning framework to model 4th down decision quality as a function of both traditional game situation variables and richer contextual information. Play-by-play data are assembled at the 4th down level across multiple NFL seasons, with the response variable defined as the ex-ante optimal action (hard count, go for it, punt, or attempt a field goal) based on expected points and win probability outcomes. Baseline features include score differential, yards to go, field position, time remaining, timeouts, and pregame team strength metrics, mirroring variables used in existing public models. We then extend this feature set to incorporate granular situational covariates, such as offensive and defensive pre-snap formations, personnel groupings, coaching tendencies estimated from historical decision frequencies, and in game momentum indicators. Models considered include regularized multinomial logistic regression and tree-based methods (random forests and gradient boosting), allowing us to balance interpretability with the ability to capture nonlinear interactions. Model performance is evaluated using out of sample cross-validation, with accuracy, log loss, and calibration of predicted decision probabilities as primary metrics.

To account for structural changes induced by recent NFL kickoff rule modifications, we explicitly model season specific field position dynamics. Variables capturing expected starting field position following kickoffs are interacted with 4th down outcomes, ensuring that the altered risk/reward tradeoff of punts and field goals is reflected in the learning process. We also include season fixed effects and rule era indicators to prevent pooling assumptions that would mask systematic differences between pre- and post-“dynamic kickoff” environments. This design allows the model to adapt as kickoff incentives evolve, rather than relying on historically averaged payoffs that may no longer apply. By integrating rule changes directly into the feature space and validation process, the resulting model provides decision recommendations that are both context aware and robust to ongoing shifts in league policy, making it suitable for real time, coach-facing decision support.

**Project Plan**

| Period     | Activity                                                                 | Dependencies          | Milestone                                                                 |
|------------|--------------------------------------------------------------------------|-----------------------|---------------------------------------------------------------------------|
| Week 4-5     | Data acquisition and preprocessing: pull nflFastR play-by-play data (2022–2024), compile manual offensive/defensive playcaller dataset, implement data joining pipeline, and clean data for relevant 4th down and kickoff contexts. | None                  | Project proposal finalized and submitted. Core datasets acquired, merged, and cleaned for analysis. |
| Week 6-7     | Feature engineering and dimensionality reduction: create field position metrics under new kickoff rules, encode categorical variables, perform PCA on game-situation variables, and generate correlation heatmaps for 4th down success factors. | Week 4 data           | Feature set prepared for modeling. Key variance drivers and relationships identified. Weekly update submitted. |
| Weeks 8–9  | Clustering analysis of 4th down scenarios: implement K-Means clustering, evaluate optimal number of clusters using Elbow and Silhouette methods, visualize clusters via PCA/t-SNE, and profile distinct scenario types. | Week 5 features       | Meaningful 4th down scenario clusters defined and interpreted. Weekly updates for Weeks 8 and 9 submitted. |
| Week 10-11     | Initial modeling and reporting: train baseline logistic regression model for 4th down success, tune model hyperparameters, analyze dynamic kickoff impacts, and draft methodology and preliminary results sections. | Weeks 6–7 clusters    | baseline modeling results and initial findings completed. |
| Week 12-13     | Advanced modeling and validation: implement random forests/gradient boosting, incorporate fixed effects for kickoff rules, perform cross-validation, and refine based on metrics (accuracy >80%, log loss <0.5). | Week 8 baseline       | Optimized model ready; rule change impacts quantified. Weekly update submitted. |
| Week 14-15    | Final integration, visualization, and delivery: add film validation for key features, create decision dashboards, finalize paper/presentation, and conduct internal review. | Week 9 model          | Full project delivered: model, report, presentation. Success metrics met (e.g., model outperforms baselines by 10% in simulation). |



**Risks**

Collecting data for coverage looks could prove to be the most difficult aspect of the project. Publicly available data is very sparse for defensive scheme play calls and we may have to resort to manually inputting coverage looks by reviewing the film of select 4th down plays. While tedious, we believe we can build an efficient enough process for the 3 of us to collect this data in a timely manner.

There is also a chance that some 4th down decision variables we try to include, such as encroachment possibility, may be such a rare and/or random occurrence that it doesn’t end up being useful to our analysis. In this case, we would make sure to further refine the robustness of our findings that do carry significance.

