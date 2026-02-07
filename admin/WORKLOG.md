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
