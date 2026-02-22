Template:

## YYYY-MM-DD - [Brief Title] (responsible person)

**Context**: Why was this work needed?

**Problem Identified** (if applicable): What issue prompted this work?

**Solution Implemented**:
- Key changes made
- Files modified
- New functionality added

**Impact**: How does this affect the pipeline/project?

**Next Steps** (optional): What should happen next?

-------------------------------------------------------------------------------------

## 2026-02-07 - [Manual Playcaller Data Collection] (FG)

**Context**: One of our goals is to associate coaching tendencies with our findings. Do certain coaches have a significant impact on the number of penalties their teams draw/commit on fourth down? Do they bring ST tendencies with them? Now we can look at year over year as well as week by week trends, especially if there are in-season coaching changes

**Problem Identified** (if applicable): What issue prompted this work?
- there is no universal data source with this information readily available
- many, many Google searches were required, would take hours to write a scraper with the necessary guardrails to get the right answer every time


**Solution Implemented**:
- taught a gemini instance to gather the information and format properly, then validated information by randomly drawing teams/seasons and manually searching information
- added playcallers_by_week.xlsx file to repo

**Impact**: How does this affect the pipeline/project?
- adds layer to our analysis (coaching impact)

**Next Steps** (optional): What should happen next?
- data engineering: merge with play-by-play data, encode coach variables

-------------------------------------------------------------------------------------

## 2026-02-07 - [Fourth Down Data Collection] (JW)

**Context**:  
A central objective of the project is to understand decision-making and outcomes in high-leverage situations. Fourth downs represent some of the most strategically important moments in a game, and isolating these plays allows us to study how context (field position, score, time, personnel, coaching tendencies) influences behavior and success rates.

**Problem Identified** (if applicable):  
- Fourth down plays are relatively sparse compared to total play volume, making them easy to lose in broader play-by-play datasets  
- Raw play-by-play data contains many edge cases (timeouts, penalties, no-plays) that complicate identifying true fourth down attempts  
- Relevant situational variables are spread across multiple columns and require careful filtering and standardization  

**Solution Implemented**:
- Pulled full play-by-play data using `nflfastR` for the target seasons  
- Filtered the dataset to isolate true fourth down plays, excluding no-plays, dead-ball penalties, and irrelevant stoppages  
- Engineered key situational features (e.g., distance to go, field position, score differential, time remaining, quarter) to support downstream analysis  
- Created a clean, fourth-down–only dataset suitable for modeling and exploratory analysis  

**Impact**:  
- Establishes a reliable foundation for analyzing fourth down behavior and outcomes  
- Enables consistent comparisons across teams, seasons, and coaching staffs  
- Serves as a core input for later modeling, dimensionality reduction, and coaching impact analysis  

**Next Steps** (optional):  
- Merge fourth down dataset with manually collected coaching/playcaller data  
- Conduct exploratory analysis to identify key drivers of fourth down decisions and success  
- Integrate engineered features into downstream predictive models

  -------------------------------------------------------------------------------------

## 2026-02-08 - [Full Data Join] (GS)

**Context**:  
Have all of our data in 6 different sets, need to join into a single csv file

**Problem Identified** (if applicable):  


**Solution Implemented**:
- Created pipeline to join all data together including, all 4th downs, coaching impact, and hard count attempts

**Impact**:  
- Gives us a dataset to build off of for the rest of the project

  -------------------------------------------------------------------------------------

## 2026-02-14 - [Drive Field Position Start] (FG)

**Context**: Field position was stored as "BUF 30" previously. Needed to be converted into a format where a model could interpret bigger numbers as being further from scoring. It isn't going to get the fact that the offensive team's abbrev means the team is further from scoring and the defensive team's abbrev is closer to scoring. Now it is universally formated no matter which two teams are matched up

**Problem Identified** (if applicable): Consistent formatting is super important for ML models

**Solution Implemented**:
- Wrote feature engineering process as a function so it can be universally applied to all of our csv files
- started a codespace
- function can be tweaked later on to account for new scenarios/issues

**Impact**: will help us with every step of modeling

**Next Steps** (optional): What should happen next?

-------------------------------------------------------------------------------------


## 2026-02-15 - [Correlation Heatmap] (FG)

**Context**: Observe relationships between predictors/features

**Problem Identified** (if applicable): What issue prompted this work?

**Solution Implemented**:
- Started by removing unnecessary columns such as play id, game id, play desc, etc
- Then removed player identifiers like jersey number and name, since this was just trying to uncover numeric relationships

**Impact**: Ran into issues. There are so many predictors to sift through, I think this will be more valuable to complete later on once we've decided exactly what variables we will use to predict success/playcall

**Next Steps** (optional): Reduce dimensions of data

-------------------------------------------------------------------------------------

## 2026-02-15 - [Feature Reduction for Fourth Down Modeling] (JW)

**Context**: The combined dataset contained 372 variables, many of which were irrelevant, redundant, outcome-based, or too sparse to be useful for modeling fourth down decisions. Reducing dimensionality was necessary to focus on pre-snap situational factors and make modeling feasible.

**Problem Identified** (if applicable): High-dimensional data increases computational cost, complicates interpretation, and raises overfitting risk. Numerous variables represented player information, identifiers, post-play outcomes, or contained almost no variation (mostly zeros or missing values).

**Solution Implemented**:
- Reduced the dataset from 372 variables to 139 variables
- Removed columns matching patterns associated with outcomes, identifiers, or irrelevant contexts (e.g., probability metrics, EPA/WP, player fields, IDs, kickoff-related variables)
- Eliminated columns with ≥99% zeros or missing values to remove near-constant predictors
- Created a reproducible filtering pipeline using pattern-based exclusion and sparsity thresholds
- Exported lists of removed and retained variables (`removed_variables.txt`, `kept_variables.txt`) for documentation and transparency
  
**Impact**: Produces a cleaner, model-ready dataset focused on variables plausibly influencing fourth down decision-making, reduces noise and computational burden, and improves interpretability for downstream analysis and modeling
  
**Next Steps** (optional): Validate that key situational variables were retained, perform additional feature selection or dimensionality reduction if needed, and integrate the filtered dataset into the predictive modeling pipeline

-------------------------------------------------------------------------------------

## 2026-02-15 - [Encoding Categorical Variables] (GS)

**Context**: Large dataset with many numeric and categorical variables

**Problem Identified** (if applicable): Determining which categorical variables are meaningful and should be encoded versus removed, dealing with data before full dimensionality reduction.

**Solution Implemented**:
- Wrote an encoding script for future use once full dimensionality reduction happens and meaningful categorical variables are determined
- Pipelines for ordinal variables to be onehotencoded and others to be label encoded
  
**Impact**: While encoding for categorical variables isn't completely finished, it will be easily done through saved script once initial steps are made
  
**Next Steps** (optional): Fully Reduce dimensions of data

--------------------------------------------------------------------------------------
## 2026-02-22 - [Encoding Categorical Variables Continued] (GS)

**Context**: Large dataset with many numeric and categorical variables

**Problem Identified** (if applicable): Determining which categorical variables are meaningful and should be encoded versus removed, dealing with data before full dimensionality reduction. Getting everything to numerical form for PCA purposes, handling data that inherently has many nulls.

**Solution Implemented**:
- Wrote an encoding script for future use once full dimensionality reduction happens and meaningful categorical variables are determined
- Pipelines for ordinal variables to be onehotencoded and others to be label encoded
- Replaced nulls with 0s and removed variables with far too many nulls
  
**Impact**: Dataset is prepared for dimensionality reduction.
  
**Next Steps** (optional): Fully Reduce dimensions of data




