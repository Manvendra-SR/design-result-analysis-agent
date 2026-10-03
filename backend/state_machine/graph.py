"""
backend/state_machine/graph.py
================================
The investigation loop, as a LangGraph ``StateGraph``:

    plan_experiments ─► run_experiments ─► analyze ─► decide ──refine──► run_experiments
                                                         │                (next round)
                                                         └──conclude──► finalize ─► END

plan_experiments  Planner LLM -> Plan. Effect mode: one factor, its levels, a
                  reference. Selection mode: 2-4 model families at their
                  defaults. Code derives the round-1 candidates and expands
                  them into runs: N_SEEDS per seeded family, one otherwise.
run_experiments   Train this round's runs; store each result as it finishes.
analyze           Validation-split statistics over every result so far (tools/stats.py):
                  vs reference, and in selection mode vs the leader and vs parent,
                  plus the contenders (candidates not clearly worse than the leader).
decide            Code first: the last round concludes (budget); in selection
                  mode, contenders that all belong to one family conclude
                  (settled). Otherwise the Recommender LLM refines one knob of
                  one contender (1-3 values) or concludes; code validates it.
finalize          From validation, choose the winner (and, in selection mode, the
                  runner-up from another family). Then unseal the test split ONCE
                  for those comparisons, at 95% / 97.5% each (Bonferroni).
                  Deterministic headline + LLM-written interpretation -> Report.

Hard limits, all code constants: rounds <= MAX_ROUNDS, new candidates per
round <= MAX_NEW_CANDIDATES, runs per candidate <= N_SEEDS, so an
investigation trains at most (initial candidates + (MAX_ROUNDS - 1) x 3) x
N_SEEDS runs and makes at most MAX_ROUNDS + 1 LLM calls.

Why the test split stays sealed until ``finalize``: the loop looks at
validation results as often as it likes, and adaptively choosing what to try
next based on them would bias any significance claim made on that same data.
The comparisons on the untouched test split are chosen before it is looked at,
so they are an honest answer.

State lives in memory for the duration of one run. Experiments, the plan, each
decision and the report are written to the database as they are produced so
the UI can follow along; a run interrupted by a crash is marked failed and can
simply be started again.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Callable, List, Optional, Sequence, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

import backend.config as config
from backend.agents.llm import LLMError
from backend.database.repository import Repository
from backend.models.dataset import DatasetProfile
from backend.models.experiment import FAMILIES, ExperimentConfiguration, ExperimentResult
from backend.models.investigation import Analysis, Candidate, Comparison, Decision, Plan, Report
from backend.tools import stats
from backend.tools.experiment_runner import run_experiment

logger = logging.getLogger(__name__)

RunFn = Callable[[ExperimentConfiguration, DatasetProfile, str, int], ExperimentResult]


class LoopState(TypedDict, total=False):
    session_id: str
    question: str
    profile: DatasetProfile
    plan: Plan
    round: int
    queue: List[ExperimentConfiguration]
    results: List[ExperimentResult]
    analysis: Analysis
    decisions: List[Decision]
    report: Report


def configs_for(candidates: Sequence[Candidate], dataset_id: str, n_seeds: int) -> List[ExperimentConfiguration]:
    """Seeds 0..n_seeds-1 for each seeded family; one run for a seed-independent one."""
    return [
        c.config(dataset_id, seed)
        for c in candidates
        for seed in (range(n_seeds) if FAMILIES[c.model_type].seeded else [0])
    ]


def _fmt(metric: str):
    accuracy = metric == "accuracy"
    value = (lambda v: f"{v:.1%}") if accuracy else (lambda v: f"{v:.4g}")
    delta = (lambda d: f"{d * 100:+.1f} pts") if accuracy else (lambda d: f"{d:+.4g}")
    return value, delta


def headline(
    report_fields: dict, primary: Comparison, secondary: Optional[Comparison], reference_led: bool = False
) -> str:
    """Deterministic sentences stating the test-split result.

    ``reference_led``: the reference itself was best on validation, so the
    candidate tested is the best *alternative* to it - the headline says so.
    """
    value, delta = _fmt(report_fields["metric"])
    level = f"{report_fields['confidence']:.1%}".replace(".0%", "%")

    def ci(c: Comparison) -> str:
        return f"difference {delta(c.diff)} ({level} CI {delta(c.ci_low)} to {delta(c.ci_high)})"

    winner, reference = report_fields["winner"], report_fields["reference"]
    answer = {
        "better": f"{winner} performs better than {reference}",
        "worse": f"{winner} performs worse than {reference}",
        "inconclusive": f"there is no reliable difference between {winner} and {reference}",
    }[primary.verdict]
    lead = f"{reference} was the best candidate on validation, so its strongest alternative was tested. " \
        if reference_led else ""
    text = lead + (
        f"On the held-out test set ({report_fields['n_test_rows']} rows), {answer}: "
        f"{report_fields['metric']} {value(report_fields['winner_score'].value)} vs "
        f"{value(report_fields['reference_score'].value)}, {ci(primary)}."
    )
    if secondary is not None:
        rival = report_fields["runner_up"]
        versus = {
            "better": f"{winner} is also better than the best other family, {rival}",
            "worse": f"{winner} is worse than {rival}",
            "inconclusive": f"{winner} and {rival} are statistically tied",
        }[secondary.verdict]
        text += f" Against the strongest rival, {versus}: {ci(secondary)}."
    return text


def build_graph(
    repo: Repository,
    planner: Any,
    recommender: Any,
    run: RunFn = run_experiment,
    max_rounds: int = config.MAX_ROUNDS,
    n_seeds: int = config.N_SEEDS,
):
    def plan_node(state: LoopState) -> LoopState:
        plan = planner.plan(state["question"], state["profile"])
        repo.save_plan(state["session_id"], plan)
        queue = configs_for(plan.candidates, state["profile"].dataset_id, n_seeds)
        logger.info("plan: mode=%s candidates=%s -> %d runs", plan.mode, [c.label for c in plan.candidates], len(queue))
        return {"plan": plan, "round": 1, "queue": queue, "results": [], "decisions": []}

    def run_node(state: LoopState) -> LoopState:
        new = []
        for cfg in state["queue"]:
            result = run(cfg, state["profile"], state["session_id"], state["round"])
            repo.add_experiment(result)
            new.append(result)
        logger.info("run: round %d -> %d runs (%d failed)", state["round"], len(new),
                    sum(r.status == "failed" for r in new))
        return {"results": [*state["results"], *new], "queue": []}

    def analyze_node(state: LoopState) -> LoopState:
        return {"analysis": stats.analyze(state["plan"], state["results"], state["profile"], "val")}

    def decide_node(state: LoopState) -> LoopState:
        plan, rnd, analysis = state["plan"], state["round"], state["analysis"]
        if rnd >= max_rounds:
            decision = Decision(
                round=rnd, action="conclude", decided_by="budget",
                rationale=f"Round budget reached ({max_rounds} rounds).",
            )
        elif stats.settled(plan, analysis):
            family = plan.get(analysis.contenders[0]).model_type
            decision = Decision(
                round=rnd, action="conclude", decided_by="settled",
                rationale=f"Every candidate still in contention is a {family}, so further tuning "
                          "cannot change which family is best.",
            )
        else:
            decision = recommender.decide(state["question"], plan, analysis, rnd, max_rounds)
        update: LoopState = {}
        if decision.action == "refine":
            new = plan.refine(decision.parent, decision.knob, decision.values, rnd + 1, analysis.contenders or None)
            decision = decision.model_copy(update={"new_candidates": [c.id for c in new]})
            plan = plan.with_candidates(new)
            repo.save_plan(state["session_id"], plan)
            update.update(plan=plan, round=rnd + 1,
                          queue=configs_for(new, state["profile"].dataset_id, n_seeds))
        repo.append_decision(state["session_id"], decision)
        update["decisions"] = [*state["decisions"], decision]
        logger.info("decide: round %d -> %s %s", rnd, decision.action, decision.new_candidates or "")
        return update

    def finalize_node(state: LoopState) -> LoopState:
        plan, analysis, results = state["plan"], state["analysis"], state["results"]
        # select: decided from validation only
        winner_id, runner_up_id = stats.select_final(plan, analysis)
        if winner_id is None or analysis.leader is None:
            raise RuntimeError(
                "No successful runs for the reference or for any other candidate, so there is nothing to compare."
            )
        # test: the only place test scores are read
        primary, secondary, confidence = stats.final_comparisons(
            plan, results, winner_id, runner_up_id, analysis.higher_is_better
        )
        if primary is None:
            raise RuntimeError("The reference has no successful run, so there is nothing to compare.")
        groups = stats.group_by_candidate(plan, results)
        winner_rows = stats.condition_rows(groups[winner_id], "test")
        reference_rows = stats.condition_rows(groups[plan.reference], "test")
        winner_summary = next(c for c in analysis.conditions if c.id == winner_id)
        sign = 1.0 if analysis.higher_is_better else -1.0
        # report
        fields = dict(
            mode=plan.mode,
            winner=plan.get(winner_id).label,
            winner_id=winner_id,
            reference=plan.get(plan.reference).label,
            runner_up=plan.get(runner_up_id).label if runner_up_id else None,
            metric=analysis.metric,
            higher_is_better=analysis.higher_is_better,
            winner_score=stats.score_ci(winner_rows),
            reference_score=stats.score_ci(reference_rows),
            confidence=confidence,
            winner_val_score=winner_summary.mean,
            val_to_test_drop=sign * (winner_summary.mean - float(np.mean(winner_rows))),
            effort=dict(Counter(c.model_type for c in plan.candidates)),
            candidates_tried=len(plan.candidates),
            majority_rate=analysis.majority_rate,
            n_test_rows=len(winner_rows),
            rounds_run=state["round"],
            stopped_by=state["decisions"][-1].decided_by,
        )
        report = Report(primary=primary, secondary=secondary,
                        headline=headline(fields, primary, secondary, analysis.leader == plan.reference),
                        **fields)
        try:
            report.interpretation = recommender.interpret(state["question"], plan, report)
        except LLMError as exc:  # the numbers stand on their own; the prose is optional
            logger.warning("finalize: interpretation unavailable: %s", exc)
        repo.save_report(state["session_id"], report)
        logger.info("finalize: %s", report.headline)
        return {"report": report}

    graph = StateGraph(LoopState)
    graph.add_node("plan_experiments", plan_node)
    graph.add_node("run_experiments", run_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("decide", decide_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "plan_experiments")
    graph.add_edge("plan_experiments", "run_experiments")
    graph.add_edge("run_experiments", "analyze")
    graph.add_edge("analyze", "decide")
    graph.add_conditional_edges(
        "decide",
        lambda s: "run_experiments" if s["decisions"][-1].action == "refine" else "finalize",
        ["run_experiments", "finalize"],
    )
    graph.add_edge("finalize", END)
    return graph.compile()


def run_investigation(session_id: str, repo: Repository, planner: Any, recommender: Any, **graph_options) -> None:
    """Run one investigation start to finish. Never raises: a failure is
    recorded on the session (status 'failed' + error) for the UI to show."""
    try:
        session = repo.get_session(session_id)
        profile = repo.get_dataset(session.dataset_id)
        repo.set_status(session_id, "running")
        graph = build_graph(repo, planner, recommender, **graph_options)
        rounds = graph_options.get("max_rounds", config.MAX_ROUNDS)
        graph.invoke(
            {"session_id": session_id, "question": session.research_question, "profile": profile},
            config={"recursion_limit": 4 * rounds + 10},
        )
        repo.set_status(session_id, "done")
    except Exception as exc:  # noqa: BLE001 - background task: record, don't raise
        logger.exception("investigation %s failed", session_id)
        repo.set_status(session_id, "failed", error=f"{type(exc).__name__}: {exc}"[:2000])
