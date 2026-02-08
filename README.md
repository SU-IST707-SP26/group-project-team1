# Overview

Welcome to your group project repository!  Please follow these instructions:
- Please use the following folders as instructed; failure to do so may result in lost points on your assignments:
   - **`admin` directory** - All project management related files belong here.  Please create your own WORKPLAN and VISION files rather than editing the "_sample" files.
   - **`work` directory** - All supporting work will go here; feel free to create subdirectories here to keep things organized.
   - **`data` directory** - You may keep _small_ (<100M) data files here.  However, be aware that GitHub will not allow you to upload large data files directly to your repository.  If you do so accidentally, it will result in some not fun issues that I will have to help you correct. So, please don't do this.
   - **`proposal`,`checkpoint`,`presentation`,`final`** - Please place your submissions in the relevant folders when they are due.  The rubric in each folder explains what is expected.  Follow the rubrics.

- There will be new commits to the project every week henceforth.  Lack of regular activity will reduce your final grade, even if your project is wonderful in the end! Individual contributions will be tracked and used to assign your final project grade.

- See the note about data above.  Do not try to save *any* files > 100M, or else you will encounter problems committing your work and will likely need my help fixing it.

The following project management guide explains how we will track and manage our group project throughout the semester. Our approach is intentionally lightweight — the goal is to form strong project management habits without adding unnecessary tool overhead.

---

# Project Management

## Why We Do This

* **Keep the team aligned** — Everyone should know the project vision, the current plan, and recent progress at all times.
* **Support AI-assisted work** — Modern AI tools are powerful, but their memory is limited. Having structured, up-to-date documents gives AI agents the context they need to work effectively with you, even after restarts or when using multiple agents.
* **Integrate with git** — By keeping planning artifacts in the repository, you can link tasks in the plan directly to your commits, making it easy to correlate planning with execution.

The key is not to have a “perfect” system — it’s to **use a system reliably**.



All documents live at the top level of your project repository.


* **Purpose:** Defines the project's "North Star" — what you are trying to accomplish and why.
* **Stability:** Should remain unchanged unless there's a major pivot.
* **Version History:** If the vision changes, archive the old vision at the bottom with the date and reason for the change.

### `WORKPLAN.md` - Active Work Tracking

### Purpose
The WORKPLAN.md tracks current and upcoming work in a hierarchical task list. It answers: "What are we working on and what's next?"

### Structure
```markdown
# Work Plan

## Active Tasks

### Milestone 1: [Name]
- ✅ Completed task
- ⏳ Current task (marked with hourglass)
- [ ] Future task 1
  - [ ] Subtask if needed
- [ ] Future task 2

### Milestone 2: [Name]
- [ ] Task

## Changelog

### YYYY-MM-DD
- Added milestone X with tasks Y, Z
- Completed task A, removed task B (no longer needed)
- Reprioritized milestone C
```

### Task Granularity
- Each task should reflect something that can be done in 1-3 hours.
- Tasks should be **actionable milestones**, not detailed steps
- If you need more detail, create more detailed plan documents in the admin folder (e.g., "DATA_JOINING_PLAN.md")
- Example of good granularity:
  - ✅ Good: "Implement unified pipeline runner with stage orchestration"
  - ❌ Too detailed: "Add argparse arguments for pipeline runner"
  - ❌ Too vague: "Work on pipeline"

### Marking Progress
- **Responsibility**: Always include the person responsible for any progress on a task using parentheses and initials / username at the end of the task.  E.g., (JEI).
- **Completed**: Add ✅ at the start, keep in place (don't move/delete)
- **Current work**: Add ⏳ at the start of exactly one task
- **Blocked**: Add 🚫 for tasks that cannot proceed due to external blockers
- **Abandoned**: Add ❌ for tasks that are no longer needed
- **Future tasks**: Leave as `- [ ]`

### Changelog
- **Log significant changes only**: New milestones, major scope changes, completed phases
- **Timestamp all entries**: Use YYYY-MM-DD format
- **Keep concise**: One-line summaries of what changed

### Maintenance
- Update WORKPLAN.md at the **start** and **end** of each work session
- Always mark responsibility for tasks
- Check off completed tasks immediately (don't batch)
- Add new tasks as they're discovered
- Log major changes to the changelog

### `WORKLOG.md` - Activity History

### Purpose
The WORKLOG.md serves as a chronological record of all significant work completed on the project. It answers: "What did we do, who did it, and why?"

### When to Update
Update WORKLOG.md immediately after completing significant milestones:
- Implementing new features or fixing critical bugs
- Completing pipeline stages or major refactorings
- Making architectural decisions
- Discovering and resolving important issues

### Entry Format
```markdown
## YYYY-MM-DD - [Brief Title] (responsible person)

**Context**: Why was this work needed?

**Problem Identified** (if applicable): What issue prompted this work?

**Solution Implemented**:
- Key changes made
- Files modified
- New functionality added

**Impact**: How does this affect the pipeline/project?

**Next Steps** (optional): What should happen next?
```

### Guidelines
- **Date entries in reverse chronological order** - Most recent work at the **top**
- **Be specific** - Include file names, line numbers, and key code snippets
- **Explain decisions** - Future readers need context for why changes were made
- **Link to plans** - Reference relevant plan documents (e.g., "See STAGE_08_TRAJECTORY_PLAN.md")
- **Document breaking changes** - Highlight changes that affect other components

---

## How to Maintain These Documents

### VISION.md

1. Write the initial vision at the top of the document in a “Current Vision” section.
    * Less detailed than a project proposal
    * Should contain project title, stakeholder(s), problem statement, and envisioned solution
    * Add as much detail as you think is necessary to communicate your project goals to an AI
2. If a major pivot happens:
   * Move the old vision to the “Version History” section with date and reason.
   * Write the new vision at the top.

### WORKPLAN.md

1. Keep the **Active Plan** current:

   * Add new tasks with a unique identifier (usually something like `M2-T4` - the id should never change once assigned).  
   * Always include the responsible person once known
   * Revise tasks as necessary, and note them with 🔄 in the Changelog.
   * Update statuses in the active plan as work progresses (from ⏳ to ✅); make sure to indicate the responsible parties for the task.
   * Remove tasks that are abandoned and tag them in the **Changelog** with ❌. 

2. Log your work after every work session:

   * Each individual should log their own work
   * If a task has been completed, indicate this and include any observations or summaries as necessary
   * If a task has not been complete, include a note describing what you did and where you are leaving off
   * Always add your initial or username to each entry
   
3. At the end of each week:

   * Update statuses in **Active Plan**.
   * Add any work in progress to the **Changelog** to help you, the team, and your professor keep track of work that doesn't necessarily result in a completed task.
   * Add **Changelog** entries to note changes (decision, pivot, scope change, abandoned, blocker).

4. If the project changes significantly:

   * For a **pivot**: If the entire plan changes, archive the previous plan into Changelog under a "Project Pivot" heading, then start fresh.
   * For **major adaptations**: Note the change in the Changelog, adjust affected tasks in Active Plan, and tag them ❌ or 🔄.
   * For **minor additions**: Just add 🆕 tasks to the Active Plan.

---

## Using These Documents with AI

You can copy relevant sections of these files (or even the whole file) into AI prompts to:

* Summarize current status.
* Generate next-step suggestions.
* Identify blockers and propose solutions.
* Update the plan automatically.

Because these files are plain Markdown, they can be read, edited, and version-controlled by both humans and AI.

---

## Linking to Git Commit History

When you make commits:

* Reference the related WORKPLAN task in the commit message.
* Example:

```
git commit -m "✅ Implement PCA for feature reduction — relates to M2-T3"
```

* This creates a clear link between your commits and your project plan, making it easy to see how plans translate into work.

---

# Worked Examples


## 🔄 Example 1: Major Pivot (Dataset Change)

### **Before Pivot (Week 4)**

**WORKPLAN.md (Active Plan excerpt)**

```
## Active Plan

### Milestone 1: Data Prep
- [✅] M1.T1 — Acquire EHR dataset (Alice)
- [⏳] M1.T2 — Initial EDA (Team)
- [ ]  M1.T3 — Clean data (Bob)
- [ ]  M1.T4 — Feature engineering (Carol)
```

**WORKLOG.md**

```
2025-09-19
- (Team) M1.T2 — Initial EDA

2025-09-12
- (Alice) M1.T1 — Acquire EHR dataset
- (Team) M1.T2 — Initial EDA (started)
```

---

### **Pivot Happens**

EHR dataset is unusable. Switch to predicting educational outcomes using a new dataset.
Cleaning and feature engineering remain relevant but are applied to the new data.

**WORKPLAN.md (Active Plan excerpt, after pivot)**

```
## Active Plan

### Milestone 1: Data Prep
- [✅] M1.T1 Acquire EHR dataset (Alice)
- [ ] M1.T2 — Initial EDA (Team)
- [⏳] M1.T1A — Acquire educational dataset (Alice)
- [ ] M1.T2A — Initial EDA of new data (Team)
- [ ] M1.T3 — Clean data (Bob)   [revised]
- [ ] M1.T4 — Feature engineering (Carol)   [revised]

## Changelog
2025-09-21 - 
- (Bob) M1.T2 — Initial EDA - worked on this, scatterplots suggest this is simulated data? Contacted others...

2025-09-23 - **Major Pivot**
- (Alice) 🆕 M1.T1A — Added new task to acquire educational dataset.
- (Team) 🆕 M1.T2A added to reflect new data set
```

**WORKLOG.md**

```
2025-09-26
- (Alice) M1.T1A — Acquire educational dataset
- (Team) M1.T2A — Initial EDA on new dataset

2025-09-19
- (Team) M1.T2 — Initial EDA; the data was too regular to be real. GPA's were uniformly distributed, rather than normally distributed.

2025-09-12
- (Alice) M1.T1 — Acquire EHR dataset
- (Team) M1.T2 — Initial EDA (started)
```

---

## 🔄 Example 2: Minor Change (New Step Added)

### **Before Change (Week 3)**

**WORKPLAN.md (Active Plan excerpt)**

```
## Active Plan

### Milestone 2: Feature Engineering
- [⏳] M2.T1 — Extract sentiment scores from tweets (Bob)
- [ ]  M2.T2 — Aggregate scores into daily indicators (Alice)

### Milestone 3: Initial linear modeling
- [ ] M3.T1 - Attempt to fit linear regression (Carol)

## Changelog
2025-09-22
- (Bob) M2.T1 — Evaluated VADER; not happy with results. Going to try https://huggingface.co/shhossain/all-MiniLM-L6-v2-sentiment-classifier

```
---

### **Minor Change Happens (Week 4)**

Team decides to add PCA dimensionality reduction.

**WORKPLAN.md (Active Plan excerpt, after change)**

```
## Active Plan

### Milestone 2: Feature Engineering
- [✅] M2.T1 — Extract sentiment scores from tweets (Bob)
- [⏳] M2.T2 — Aggregate scores into daily indicators (Alice)
- [ ]  M2.T3 — Implement PCA for dimensionality reduction (Carol)

### Milestone 3: Initial linear modeling
- [ ] M3.T1 - Attempt to fit linear regression (Carol)


## Changelog
- (Carol) 🆕 M2.T3 — Added PCA for dimensionality reduction.
```

**WORKLOG.md**
```
2025-09-26
- (Carol) Data has too many dimensions; I think we need dimensionality reduction

2025-09-24
- (Alice) Tried aggregating scores, but uncertain how to handle different sentiment types from new model?

2025-09-22
- (Bob) M2.T1 — Evaluated VADER; not happy with results. Going to try https://huggingface.co/shhossain/all-MiniLM-L6-v2-sentiment-classifier
```


---

👉 Notice how this system keeps:

* **Active Plan** as the always-current, canonical view.
* **Changelog** as the history of any major changes to the plan
* **WORKLOG.md** as a trace of decisions, interim summaries, and decision points

---


## Final Notes

* Commit updates to these documents **at least once a week**. This is a large part of your project grade!
* Consistency matters more than perfection — aim for accurate, current, and readable updates.
* Use these documents as living tools, not just grading requirements. If you keep them well-maintained, they will make your work easier and your AI assistance more effective.
