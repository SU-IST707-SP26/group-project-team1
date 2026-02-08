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
