**IST 707 Group Project Proposal:** *Team1*

*Fred Gullo, Jared Weber, Gavin Stein*

**Topic:** *NFL 4th Down Decision Making // Fixed-Effects Analysis of Dynamic Kickoff Rule Change*

**Title:** *Fourth & Context (still working on it, want to pick something catchier)*


**Introduction**

A big question for NFL coaches is “When do I go for it on 4th down?” Analytics pros working for teams and television networks have created algorithms answering this question in various scenarios. Coaches have a section of their playcall sheet telling them the recommendation in different scenarios. Every broadcast displays their suggestion on 4th downs where going for it is on the table. Currently, the models aren’t very specific to individual scenarios ESPN listed their predictors as: “the score of the game, distance to gain, yard line, clock, timeouts, pregame win probability and the relative strength of the offense and defense on the field” (Walder). We believe that these predictions can be more granular and situation-specific. We want to build a new predictive model that takes many more variables into account, such as pre-snap formations for the offense and defense, coaching tendencies, and in-game effects. 

An additional consideration for us is the year-to-year changes in kickoff rules. The past two seasons have seen the “dynamic kickoff” implemented in all NFL games. The rule change features all blockers and defenders closer together than they traditionally were and stationary until the kick is caught, with the intent of reducing the occurrence of injuries. This initially led to far fewer kick returns happening and the NFL changed the touchback rules (what happens if the ball hits the ground or the returner ‘fair catches’ and gives themselves up) to incentivize returns this past season. A consequence of this is now offenses start with far better field position than in previous seasons, and that changes the risk/reward payoff of going for 4th downs, since getting a field goal and giving the other team the ball at the 35 yard line may not be an optimal outcome of a drive. 

Our predictive model for 4th down scenarios can be utilized by NFL coaches for data-driven decision making that takes more contextual information into account than the models currently available. With the NFL’s kickoff rule changing likely to continue, having a model that can take these changes into account with and accurately suggest 4th down decisions is invaluable to head coaches. 

**Lit Review**


**Data**

We are going to utilize the nflFastR dataset to accomplish our goals. This publicly available dataset contains information at the play level in a very granular fashion. It contains written descriptions of each play, as well as numerous measurements and characteristics. These include the offensive and defensive formations, players involved in the play, and much more. We also plan on purchasing a subscription to the NFL's All 22 film, which will give us access to videos of every play in our dataset (from multiple angles). This will help validate our feature engineering steps and add dynamic elements to our final paper/presentation. 

We will also gather information for playcallers, which will require some Google searching. We'll create a dataset with each team's offensive/defensive/special teams play caller for every week (this sounds daunting, but playcallers rarely change midseason). We need to be sure of which coach is calling plays in every week because midseason changes are often accompanied with a change in tendency and effectiveness of results. 

nflFastR is seen as very reliable by the football analytics community. It is frequently cited in reseach and exploratory projects. Errors have been noted, as the aggregated game statistics occasionally vary slightly compared to the official statistics from PFF/NFL, but the consensus is that nflFastR is a reputable, reliable source of play-by-play data. Upgrading to a paid program would be extremely costly, and we have full faith that nflFastR provides more than enough value to justify using a free option instead.

**Methods**





