# Architecture Audit & Simplification Plan

Status: **implemented** on branch `simplify` (the previous design is tagged `pre-simplify`). Deviations from the plan below: the loop has 5 nodes (`plan_experiments`, `run_experiments`, `analyze`, `decide`, `finalize`); dataset deletion keeps its 409-unless-cascade guard; the MLP trains on CUDA when available.

**Verdict:** the project is roughly 3× larger than its purpose needs. Most of the complexity is not essential. It is scaffolding built to work around problems that earlier complexity created:

- the anomaly lifecycle exists because the anomaly rules produced false flags;
- the cycle history exists because the loop runs unattended for up to 6 cycles;
- the one-sample t-test exists because the t-test treats seeds as the sample;
- the run-phase, current-node and resume machinery exists because one HTTP request runs the whole investigation.

Remove the root causes and most of the rest falls away.

Rough size today:

- backend: ~8.4k lines of Python;
- frontend: ~3.3k lines of TS/TSX, plus ~1.1k of tests;
- Python tests: ~5.6k lines;
- checkpoint scripts: ~1.3k lines.

A realistic target is about **2–2.5k backend lines and ~1.5k frontend lines**, with **no lost capability** that matters for the project's purpose.

---

## A. Current architecture (what actually happens)

```
POST /datasets ── ingest CSV → profile → fixed 70/15/15 split
POST /sessions ── row in `sessions` (question, dataset, parent_session_id)
POST /sessions/{id}/run-cycle   (ONE blocking HTTP request runs everything)
   executor: read sessions.current_node → LangGraph entry router → node
   ┌─ planning     LLM → full config list → seeds discarded and renumbered 1..3
   │               → sessions.pending_configs, plan_explanation, current_node
   ├─ executing    run each config (tenacity retry) → experiments row (cycle=N)
   │               → shrink pending_configs after each run (crash resume)
   ├─ validating   3 anomaly rules over ALL experiments → reconcile against the
   │               anomalies table (raise / withdraw / re-open) → flip
   │               experiments.status success⇄anomalous
   ├─ analyzing    every PAIR of conditions → t-test (two-sample, one-sample,
   │               or skip with reason) + per-condition summaries
   │               → sessions.latest_analysis (JSON)
   └─ recommending LLM gets every experiment + stats + open/resolved anomalies
                   + cycle budget → conclude | run_more (seeds renumbered again)
                   → current_recommendation, append to cycle_history,
                     pending_configs, cycle_count, termination_reason
                   → loop back to executing (max 6 cycles)
GET /cycles ── rebuilds per-cycle vs cumulative views from 3 sources
Frontend polls session / experiments / cycles / recommendation every 2.5–3 s
```

What each stage really does:

| Stage | Reality |
|---|---|
| **Session** | Six fields track "where is the run": `status`, `current_node`, `run_phase`, `run_error`, `termination_reason`, `cycle_count`. |
| **Planner** | The LLM writes full config dicts. Code overrides `dataset_id`, discards its seeds, and tops each condition up to 3. "One factor at a time" is only *requested* in the prompt, never enforced. `planner._group_by_condition` is a **third definition of "condition"** that omits `dataset_id`, even though `condition.py` says the definitions were unified. |
| **Execution** | Fine. Retries are overkill: transient-error string matching is pointless for CPU training on a small CSV. |
| **Validation** | Mostly inert at n=3 (the outlier rule needs ≥5 replicates) and harmful when it does fire (see the statistics section below). Loss-divergence and validation-collapse flags are per-run and can **never be withdrawn**, so the lifecycle only helps the outlier rule. An open flag blocks concluding, so one jittery seed can force the loop to the 6-cycle cap. |
| **Analysis** | Tests every pair of conditions on seed noise only. Has the special cases (one-sample, skip records) and a CI variance bug (ddof=0, CI ~18% too narrow at n=3). |
| **Recommender** | Most of the prompt is **deterministic rules the LLM is asked to apply**: conclude if p<0.05 and ≥5 replicates and no open anomalies; never add seeds to a deterministic condition; must conclude when `cycles_remaining=0`. Its seeds and `dataset_id` get overwritten. Its only genuine freedom is *which values to try next*. |
| **Conclusion** | Either the LLM says conclude, or the cap fires and the UI has to explain "concluded but not really". |

### Statistical problems in the current design (summary)

- **The unit of analysis is a training run (seed), not a data row.** With 3 seeds per condition, a t-test has 4 degrees of freedom. The split is fixed, so seeds only capture initialisation/shuffle noise. The validation-set sampling error (~±0.75 pts SE on 2,250 rows) is ignored, though it is the dominant source of uncertainty.
- **The deterministic baseline is treated as an exact constant** (one-sample t-test). That makes MLP-vs-baseline comparisons very overconfident.
- **p < 0.05 is used as the stopping rule**, re-tested every cycle (optional stopping), with **no multiple-comparison correction** across all pairs.
- **The CI uses population variance** (`np.var`, ddof=0) while `ttest_ind` uses sample variance, so the CI and p-value can disagree.
- **MLP gets best-epoch selection on the validation set** and is scored on that same set; the baseline gets no such selection.
- **The test split is loaded but never used.**
- **Anomaly rules drop legitimate variance:** outlier removal before testing inflates significance. Loss-divergence has zero tolerance. Validation-collapse uses `1/n_classes` (33%) as chance on a 60/25/15 dataset, so a model that always predicts the majority class (60%) is never flagged.

---

## B. Complexity audit

| Component | What it does | Why it exists | Needed? | Action | Reason |
|---|---|---|---|---|---|
| Anomaly detector (3 rules) + templates | Outlier, loss-divergence and collapse flags | Original spec | **No** | **Delete** | Inert or harmful at this scale. Replace with a 5-line validity check. |
| `anomalies` table, IDs, `detected_cycle` / `resolved_cycle`, reconcile, re-open | Lifecycle for flags | False flags blocked concluding | **No** | **Delete** | Patches the detector's own false positives. Gone with the detector. |
| `experiments.status` = anomalous ⇄ success transitions | Excludes runs from stats | Anomaly subsystem | **No** | **Delete** | Silently dropping valid runs biases stats. Keep only `ok` / `failed`. |
| `pending` / `running` statuses | Never written | Spec | No | Delete | Dead values. |
| `sessions.current_node` + LangGraph entry router + `RESUMABLE_NODES` | Crash resume mid-cycle | One blocking request runs minutes of training | **No** | **Delete** | A failed run on a demo project can simply be re-run. |
| `pending_configs` shrinking after each run | Resume mid-batch | Same | No | Delete | Same. |
| `run_phase`, `run_error`, `status`, `termination_reason`, `cycle_count` | Five overlapping run-state fields | UI needs to tell running / failed / done apart | Partly | **Merge** | One `status` (`pending`/`running`/`done`/`failed`) plus `error`. Round count is derivable. |
| `latest_analysis` (JSON) | Hands stats from one node to the next | Nodes communicate via the DB | **No** | **Delete** | Pure recomputation from experiments. Pass in memory. |
| `current_recommendation` + `cycle_history` + `CycleHistoryEntry` + `SessionCycle` + `GET /cycles` | Per-cycle history: per-cycle vs cumulative scopes, anomaly counts | Unattended multi-cycle loop needs to be visible | Partly | **Replace** | The only non-recomputable thing is the LLM's decision text. Keep a slim `decisions` list. Stats are recomputed. |
| `plan_explanation` column | Planner rationale | Was discarded otherwise | Yes | Merge | Into a single `plan` JSON. |
| `parent_session_id` + FollowUpPanel | Follow-up investigations | Feature creep | **No** | **Delete** | "New session" does the same thing. |
| `AnalysisSnapshot`, `ComparisonSkip`, `SummaryStatistics`, `compute_summary_statistics` | Stats containers | Pairwise design and skip reporting | No | Replace | `SummaryStatistics` is **unused**. Skips disappear with the new method. |
| Two-sample / one-sample / skip t-test branches | Seed-based significance | Treats seeds as the sample | **No** | **Replace** | Paired bootstrap over evaluation rows (§H). One code path, no special cases. |
| `assign_matched_seeds` + top-ups (4, 5, 6) + dedup | Renumber seeds | LLM picked seeds; adaptive replicate top-ups | **No** | Delete | Seeds become `range(n_seeds)` set by code. The LLM never sees seeds. |
| Best-epoch selection (`best_epoch`, `final_*`, `initial_train_loss`, `_loss_window_endpoints`, `epochs_ran`) | Early-stopping metrics, divergence input | Anomaly rule plus fairness worry | No | Delete | The sealed test set handles selection bias. The divergence rule is gone. |
| LangGraph 5-node graph, DB round-trip per node, context bundle, recursion limit, `CycleResult` / `CycleError` | Orchestration | "Agentic" architecture | Partly | **Simplify** | 4 nodes with in-memory state, or a plain loop (§C). |
| Two LLM providers (Groq + Gemini), `LLMClient` Protocol, factory, `health_check`, `close` | Provider abstraction | Hypothetical flexibility | **No** | Delete one | ~640 lines plus 2 test files for a choice nobody needs. |
| `parsing.py` balanced-brace JSON extraction | Parse non-strict LLM output | Earlier local-model era | No | Delete | Strict JSON-schema mode makes it redundant. |
| `_common.request_structured` validation-retry loop | Feeds errors back to the LLM | Business-rule failures | Yes, smaller | Simplify | Keep a ~20-line "validate, retry once with the error" helper. |
| Recommender's rule-heavy prompt (conclude criteria, budget, anomaly lifecycle, vocabulary) | Asks the LLM to apply deterministic rules | Moved logic into prose | **No** | **Replace** | Rules go in code. The LLM only proposes *what to try next* and writes the interpretation. |
| `StateManager` (968 lines, tenacity on every method, ORM⇄Pydantic converters) | DB access | Many tables and fields | Yes, smaller | Shrink | ~150 lines and ~10 functions. No retry decorators on a local DB. |
| `ExperimentRunner` tenacity retry + `_is_transient` + `run_batch` (unused) | Retry training | Requirement 10.x | No | Simplify | One try/except → `failed`. |
| Blocking `POST /run-cycle` | Runs the whole investigation in one request | — | No | Modify | Run it as a background task. The UI polls one endpoint. |
| `GET /recommendation`, `GET /experiments/{id}` | Extra read paths | Spec | No | Delete | Fold into `GET /sessions/{id}`. |
| `errors.py` with 9 exception handlers | Error mapping | Many custom exceptions | Partly | Shrink | 404 / 400 / 500 is enough. |
| GIN index on `config`, 8 other indexes | Query speed | "Phase 3 grouping" | No | Delete | Grouping happens in Python over <100 rows. |
| `scripts/checkpoint_*.py` (1.3k lines) | Phase checkpoints | Development history | No | Delete | Not part of the product. |
| Dataset ingestion / profile / split / preprocessing | Arbitrary CSV → typed dataset | Core | **Yes** | Keep | Real, useful generality. |
| Frontend: CycleHistory, AdaptiveLoopVisualizer, FollowUpPanel, `runState.ts`, EvidenceScope / Termination tests | Visualise cycle and anomaly machinery | Backend complexity | Mostly no | Delete or simplify | They mirror backend state that will no longer exist. |

---

## C. Target architecture

```
User: dataset + question
        │
        ▼
1. PLAN (LLM)  → {factor, levels[], reference_level, fixed_config, rationale}
        │        code expands to configs × seeds (one-factor rule enforced by construction)
        ▼
┌─► 2. RUN       train each new config; store metrics + per-row correctness
│                on val AND test (test stays sealed)
│       ▼
│   3. CHECK     invalid = crashed / NaN / missing metrics → excluded and shown.
│                Nothing else is ever excluded.
│       ▼
│   4. ANALYZE   (val) per condition: mean ± seed spread; each level vs reference:
│                paired-bootstrap CI of the difference. Pure function, not stored.
│       ▼
│   5. DECIDE    round < MAX_ROUNDS?  → LLM: "conclude" or "explore new levels
│                of the SAME factor" (validated by code)
└──── explore    else / conclude ↓
        ▼
6. FINALIZE    unseal test: ONE comparison (best val level vs reference)
               → bootstrap CI on test + LLM-written interpretation of the numbers
        ▼
Report
```

**Orchestration:** I recommend keeping LangGraph, but honestly. Use 4 nodes (plan → run → analyze → decide, plus a conditional edge back to run, then finalize). State lives **in memory** (`results`, `decisions`, `round`). Experiments are written to the DB as they finish, so the UI can show progress. There is no resume and no per-node DB round trip. If you can't explain why the graph beats a 25-line `while` loop, use the loop instead; both are defensible.

### The autonomous loop, explicitly

1. **What the planner decides:** which factor the question is about, which levels to try, and which level is the reference. This is the genuine natural-language-to-design step.
2. **What runs:** each level × `N_SEEDS` (3) for the MLP, and **1 run** for the linear baseline (it's deterministic, so extra seeds are wasted compute). Seeds are `0..N-1`, set by code.
3. **What the analyzer calculates:** per-level validation score; for each level, the difference from the reference with a 95% bootstrap CI; the majority-class rate as a sanity floor.
4. **What the recommender decides:** whether exploring nearby or unexplored levels of the same factor is worth a round (for example, "0.2 beat 0.0 and 0.5; try 0.1 and 0.3"), or whether to stop. Code rejects any proposal that changes a different factor.
5. **What causes another round:** the LLM chose `explore` with valid new levels, **and** `round < MAX_ROUNDS` (3).
6. **What causes termination:** the LLM chose `conclude`, or the round budget is used up. Both go to the same finalize step, so there is no "concluded but not really" state. The report just says how many rounds ran.

There are no replicate top-ups, no p-value stopping and no anomaly gating. Every extra round explores a new part of the question, which is the meaningful adaptive behaviour.

### Simplified data flow

```
Plan {factor, levels, reference, fixed_config}
   ↓ (code expands)
ExperimentConfig[]  ──run──►  ExperimentResult {config, round, status, metrics,
                                                 val_correct[], test_correct[]}
   ↓ (pure functions, recomputed every time)
validity → summaries + comparisons (val)
   ↓
Decision {round, action, new_levels, rationale}   (LLM, persisted)
   ↓ (on conclude / budget)
Report {best_level, test CI vs reference, majority floor, interpretation}
```

What stops moving around: anomalies (open/resolved), cycle numbers on every row, `AnalysisSnapshot` round-tripped through the DB, the cumulative/per-cycle split, experiment status flips, the cycle budget sent to the LLM, and full experiment dumps in every LLM prompt.

---

## D. Delete

**Backend files**

- `backend/tools/anomaly_detector.py`, `backend/tools/anomaly_templates.py`, `backend/models/anomaly.py`
- `backend/models/cycle.py` (replaced by a ~10-line `Decision` model)
- `backend/tools/statistical_analyzer.py` (replaced by `stats.py`, ~80 lines)
- `backend/state_machine/executor.py`, `backend/state_machine/context.py`, `backend/state_machine/state.py`: fold into one `loop.py` / `graph.py`
- `backend/agents/gemini_client.py`, `backend/agents/client_factory.py`, `backend/agents/llm_client.py`, `backend/agents/parsing.py`
- `backend/api/routes/experiments.py`: fold the list into the session detail
- `scripts/checkpoint_*.py`, plus `langgraph_agent_graph.png` (regenerate later if wanted)

**Classes and functions**

- `AnomalyModel`, `AnomalyReport`, `CycleHistoryEntry`, `SessionCycle`, `AnalysisSnapshot`, `ComparisonSkip`, `SummaryStatistics`, `InsufficientDataError`, `InsufficientVarianceError`, `CycleResult`, `CycleError`, `RunCycleResponse`
- `assign_matched_seeds`; planner `_repair_seeds` and `_group_by_condition`; recommender `_repair_seeds`, `_seeds_already_run`, `_as_snapshot`, most of `_build_evidence`
- `AdaptiveLoopNodes._reconcile_anomalies`, `_sync_experiment_statuses`
- `ExperimentRunner.run_batch`, `_is_transient`, `_retry_transient`
- `_loss_window_endpoints` and the best-epoch tracking in `train_mlp`
- `StateManager`: `store_anomaly`, `set_anomaly_resolution`, `query_anomalies`, `update_experiment_status`, `save/load_planned_configs`, `save/load_analysis`, `save_plan_explanation`, `append/get_cycle_history`, `set_run_phase`, `update_session_node`, `_retry_db`
- `ExperimentPlan._sync_total_count`

**Schema**

- Drop the `anomalies` table.
- Drop these `sessions` columns: `parent_session_id`, `current_node`, `run_phase`, `termination_reason`, `current_recommendation`, `pending_configs`, `latest_analysis`, `cycle_history`, `plan_explanation`, `cycle_count`, `updated_at`.
- Drop `experiments.task_type` (derivable from the dataset) and the extra indexes, including GIN.

**API**

- `GET /sessions/{id}/cycles`, `GET /sessions/{id}/recommendation`, `GET /experiments/{id}`

**Frontend**

- `CycleHistory.tsx`, `FollowUpPanel.tsx`, `lib/runState.ts`
- `AdaptiveLoopVisualizer.tsx` (or reduce it to a static 4-step indicator)
- anomaly and parent-link pieces of `ExperimentTable`, `StatisticsPanel` and `ResearchQuestionDisplay`

**Config**

- `GEMINI_*`, `LLM_PROVIDER`
- `MAX_ADAPTIVE_CYCLES` becomes `MAX_ROUNDS=3`; add `N_SEEDS=3`

---

## E. Modify

| Component | Redesign |
|---|---|
| **Planner** | Output schema becomes `{factor, levels, reference_level, fixed_config, rationale}`. Code validates that the factor is a known key (`model_type`, `normalize`, or an MLP hyperparameter), that the values are in range, and that there are ≥2 levels; it then expands to configs. One-factor-at-a-time becomes a **guarantee**, not a prompt request. |
| **Recommender** | Input: question, factor, a compact per-level table (score, seed spread, Δ vs reference, CI), and the rounds left. Output: `{action: conclude\|explore, new_levels[], rationale}`. Code rejects already-run or invalid levels. A second small call (or the same agent at finalize) writes the **interpretation** of the final numbers, with the numbers inserted by code, not by the LLM. |
| **Trainers** | Return metrics plus `val_correct` / `test_correct` per-row arrays (0/1 for classification, squared error for regression). Fixed epochs, no best-epoch logic. Same split and preprocessing for both models. |
| **Runner** | try/except → `failed` with an error message. Nothing else. |
| **Stats** | New `stats.py`: `summarize(results, split)`, `compare(a, b, split)` (paired bootstrap), `majority_rate(split)`. Pure functions. |
| **Persistence** | 3 tables, about 10 repository functions (`create/get/list dataset`, `create/get/list session`, `set_status`, `add_experiment`, `list_experiments`, `save_plan/decision/report`). |
| **API** | `POST /run` starts a BackgroundTask. `GET /sessions/{id}` returns status, plan, decisions, experiments, **live-computed** analysis and the report. |
| **Frontend** | DatasetManager, SessionList/Create, SessionView (plan card, experiments table, results table + CI chart, decision timeline, final report). One polling query. |
| **Errors** | 404 (`KeyError`), 400 (planning or validation), 502 (LLM unreachable), 500 (everything else). |

---

## F. Keep

- Dataset ingestion, profiling, the fixed split and train-only normalisation. These are good engineering and worth mentioning in an interview.
- `TabularMLP`, the `train_mlp` / `train_linear_baseline` core, and the `ExperimentConfiguration` hyperparameter validation.
- `condition_key` / `condition_label` (one definition, used everywhere).
- The Groq client with strict JSON-schema output, plus a single validation-retry.
- The "LLM never computes statistics" boundary. It is the best design idea in the project; make it even stricter.
- FastAPI + Postgres + React/TanStack Query as the stack.
- ExperimentVisualizer (rework it to plot per-level score with CI bars).

### Core vs optional

| MUST KEEP | NICE TO HAVE | DELETE |
|---|---|---|
| Dataset ingestion + fixed split | LangGraph (vs plain loop) | Anomaly subsystem + lifecycle |
| Planner LLM (factor/levels/reference) | Majority-class floor in report | Cycle history / per-cycle vs cumulative views |
| Trainers (MLP + baseline) | Decision timeline in UI | Crash resume (`current_node`, `pending_configs`) |
| Validity check (crash/NaN) | Loop-stage indicator in UI | Seed matching + top-ups |
| Stats: paired bootstrap CI | Postgres (vs SQLite) | One-sample t-test / skip records |
| Recommender LLM (explore/conclude) | | Follow-up sessions |
| Sealed test-set finalize + report | | Second LLM provider + abstraction |
| 3-table persistence + small API + UI | | Checkpoint scripts, unused models/functions |

---

## G. Data and state

**Persisted:**

```
datasets   (id, filename, storage_path, profile JSON, created_at)
sessions   (id, dataset_id, question, status, error,
            plan JSON, decisions JSON[], report JSON, created_at)
experiments(id, session_id, round, config JSON, status ok|failed, error,
            metrics JSON, val_correct JSON, test_correct JSON, created_at)
```

**Computed on read, never stored:** condition groups, summaries, comparisons, round count, which runs are invalid, and the majority-class floor.

**In memory only during a run:** the LangGraph state (`session_id`, `round`, `plan`, `results`, `decisions`).

Test for keeping anything: *"Is it non-recomputable and shown to the user?"* Only the plan, the LLM decisions, the experiments and the report pass that test.

---

## H. Statistical design

**Question:** *does level X perform better than the reference on this dataset?*

| Element | Role |
|---|---|
| **Train** | Fit models. |
| **Validation** | Everything inside the loop: per-level scores, comparisons, the recommender's decisions. Look at it as often as you like; these results are **exploratory** and labelled as such. |
| **Test** | Sealed. Predictions are stored at training time, but only `finalize` reads them, **once**, for **one** pre-determined comparison: the best validation level vs the reference. |
| **Seeds** | Not the unit of analysis. 3 seeds per MLP condition to average out initialisation noise, and the seed spread is reported as a *stability* number. The baseline runs once. |
| **Baseline** | Same split, same preprocessing, same metric, no extra selection advantage for either model. The majority-class rate is shown as a sanity floor; this replaces the validation-collapse rule. |
| **Comparison method** | **Paired bootstrap over evaluation rows.** For each row, average correctness across a condition's seeds, take the per-row difference (level − reference), resample rows 2,000 times, and use the 2.5–97.5 percentiles as the 95% CI of the accuracy difference. |
| **Significance** | "Better" means the CI lower bound is > 0. Report the CI and the difference in points; skip p-values. |
| **Multiple comparisons** | Not needed. The validation comparisons are exploratory; the test result is a single comparison chosen in advance. |
| **Stopping** | Round budget plus the LLM's explore/conclude judgement. **No p-value stopping**; peeking at validation is harmless because the test set is sealed. |

Why this design works:

- It uses the 15k rows as the sample, so it captures the dominant noise source.
- One code path handles a deterministic baseline and regression (use squared error per row).
- The bootstrap is about 15 lines of numpy.
- The ddof bug and the one-sample special case are deleted rather than patched.

Limitation to state in an interview: it conditions on the one training split, so it doesn't capture training-set variability. That would need k-fold CV, which is out of scope on purpose.

---

## I. Tests

**Delete** (they test removed machinery):

- `test_anomaly_detector`, `test_anomaly_lifecycle`
- `test_state_manager_cycles`, `test_api_cycles`, `test_followup_sessions`
- `test_seed_matching`, `test_recommender_context`
- `test_gemini_client`, `test_client_factory`, `test_agent_parsing`
- `test_trainer_epoch_metrics`, `test_analysis_reporting`, `test_state_machine_executor`
- frontend: `EvidenceScope`, `FollowUp`, `TerminationStates`, `AdaptiveLoopVisualizer`

**Rewrite** (smaller):

- `test_statistical_analyzer` → `test_stats`
- `test_planner`, `test_recommender` (fake LLM)
- `test_state_machine_nodes` / `_loop` → one `test_loop`
- `test_state_manager` (around 594 → 150 lines)
- `test_api_sessions`, `test_experiment_runner`
- frontend: `SessionView`, `StatisticsPanel`

**Keep:**

- `test_dataset_ingestion`, `test_dataset_split_and_preprocessing`
- `test_api_datasets`, `test_api_errors`, `test_groq_client`
- `test_connection` if `connection.py` survives

**Add:**

1. Bootstrap tests:
   - identical predictions give a CI of exactly [0, 0];
   - a known synthetic difference is covered by the CI;
   - results are reproducible with a fixed RNG;
   - the deterministic baseline needs no special case.
2. Validity: NaN or a crash is excluded; a weird-but-finite run is **included**.
3. Planner: a two-factor plan or an out-of-range level is rejected; reference and levels expand to the right configs.
4. Recommender: proposals for a different factor or already-run levels are rejected; at the budget it goes to finalize.
5. Loop end-to-end with a fake LLM and a tiny CSV: 2 rounds, then finalize; the test set is read exactly once (spy).

**Also fix:** `tests/` is git-ignored and untracked as of commit `01b952e`. For a resume project, **visible tests are part of the pitch**. Track them again once they've been slimmed.

---

## J. Implementation order

Each step leaves the app runnable.

0. `git tag pre-simplify` and work on a branch.
1. **Stats first, alongside the old code.** Make trainers also emit `val_correct` / `test_correct`. Add `stats.py` with tests. Nothing else changes yet.
2. **Neutralise anomalies.** The validation node calls only the validity check and never flips statuses. Stop excluding anomalous runs. Delete the detector, templates, anomaly model, table and lifecycle code, plus their tests.
3. **New planner contract** (factor / levels / reference) with code-side expansion. Delete the seed-matching machinery.
4. **New recommender contract** (conclude | explore same-factor levels) and deterministic budget handling. Add `finalize` (sealed test plus interpretation).
5. **Collapse orchestration.** Build the in-memory 4-node graph (or loop) with experiments persisted as they run. Delete executor, context, state constants, resume and the per-node DB scratch columns.
6. **Schema reset.** Create the new 3-table `schema.sql` and shrink `StateManager` to the repository. There's no Alembic, so drop and recreate the dev DB (as the schema comment already prescribes).
7. **API shrink.** Background run; a single rich `GET /sessions/{id}`; delete `/cycles`, `/recommendation`, `/experiments/{id}` and the follow-up feature; trim `errors.py`.
8. **Frontend shrink.** Remove the cycle, anomaly and follow-up components; rebuild SessionView around plan → runs → results → decisions → report.
9. **Remove leftovers:** the Gemini path, provider abstraction, `parsing.py`, checkpoint scripts, unused config.
10. **Docs last.** Rewrite README and architecture docs, and regenerate the graph image.

---

## K. Interview explanation (about 2½ minutes)

> "It's an autonomous experiment-analysis agent for tabular ML. You upload a CSV and ask a question like *'does dropout help on this dataset?'* or *'is an MLP actually better than logistic regression here?'*. The system designs, runs and statistically evaluates the experiments to answer it.
>
> There are two kinds of components, and the split is deliberate. **LLMs do the parts that need language understanding; deterministic code does everything that needs to be correct.**
>
> First, a **planner LLM** turns the question into a controlled experiment design: the factor to vary, the levels to try, and the reference to compare against. It doesn't write raw configs. It fills a small schema, and code expands that into runs, so 'change one factor at a time' is guaranteed by construction.
>
> The **runner** trains a small PyTorch MLP or a scikit-learn baseline on a fixed train/validation/test split. Each config runs with a few seeds to average out initialisation noise. Crashed or NaN runs are excluded and reported; everything else counts, because silently dropping unusual but valid results biases the statistics.
>
> The **analyzer** is plain numpy. For each level it computes the accuracy difference from the reference with a paired bootstrap over validation rows. That gives a confidence interval reflecting the real uncertainty, which comes from the finite evaluation data rather than from the random seed.
>
> Then a **recommender LLM** looks at that table and decides whether another round is worth running, for example trying dropout 0.1 and 0.3 because 0.2 looked best. Code validates the proposal and enforces a round budget.
>
> Finally, the system **unseals the test set exactly once** and evaluates a single comparison, the best validation level against the reference. Because the loop only ever looked at validation data, adaptively exploring doesn't inflate the final result: there's no p-hacking by repeated testing. The LLM writes the interpretation, but every number in the report comes from code.
>
> Under the hood: FastAPI, Postgres with three tables, a LangGraph loop of four nodes, and a React UI that shows experiments streaming in and the final result with confidence intervals. The main design choices were keeping the LLM away from the statistics, enforcing experimental-design rules in code instead of prompts, and holding out the test set so an adaptive loop still gives an honest answer."

---

### Notes on confidence

- Everything in the delete lists was checked against real usage. For example, `SummaryStatistics`, `compute_summary_statistics` and `run_batch` are referenced nowhere outside their own definitions.
- `database/connection.py` and `dataset/ingestion.py` were not read line by line, so treat "simplify connection" as tentative.
- The size targets are estimates.
