"""
backend/state_machine/graph.py
================================
The investigation loop, as a LangGraph ``StateGraph``:

    plan_experiments ─► run_experiments ─► analyze ─► decide ──explore──► run_experiments
                                                         │                (next round)
                                                         └──conclude──► finalize ─► END

plan_experiments  Planner LLM -> Plan (factor, levels, reference). Code expands it
                  into runs: N_SEEDS per MLP level, one per linear_baseline level.
run_experiments   Train this round's runs; store each result as it finishes.
analyze           Validation-split statistics over every result so far (tools/stats.py).
decide            Last round -> conclude (budget). Otherwise the Recommender LLM
                  explores new levels of the same factor, or concludes.
finalize          Unseal the test split ONCE: best validation level vs the reference.
                  Deterministic headline + LLM-written interpretation -> Report.

Why the test split stays sealed until ``finalize``: the loop looks at
validation results as often as it likes, and adaptively choosing what to try
next based on them would bias any significance claim made on that same data.
The one comparison on the untouched test split is chosen before it is looked
at, so it is an honest answer - and, being a single comparison, needs no
multiple-comparison correction.

State lives in memory for the duration of one run. Experiments, the plan, each
decision and the report are written to the database as they are produced so
the UI can follow along; a run interrupted by a crash is marked failed and can
simply be started again.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

import backend.config as config
from backend.agents.llm import LLMError
from backend.database.repository import Repository
from backend.models.dataset import DatasetProfile
from backend.models.experiment import ExperimentConfiguration, ExperimentResult
from backend.models.investigation import Analysis, Comparison, Decision, Level, Plan, Report
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


def configs_for(plan: Plan, levels: List[Level], dataset_id: str, n_seeds: int) -> List[ExperimentConfiguration]:
    """Seeds 0..n_seeds-1 for each MLP level; one run for a (seed-independent) linear_baseline level."""
    configs = []
    for level in levels:
        is_linear = plan.config_for(level, dataset_id).model_type == "linear_baseline"
        for seed in [0] if is_linear else range(n_seeds):
            configs.append(plan.config_for(level, dataset_id, seed))
    return configs


def headline(report_fields: dict, comparison: Comparison) -> str:
    """One deterministic sentence stating the test-split result."""
    accuracy = report_fields["metric"] == "accuracy"

    def value(v: float) -> str:
        return f"{v:.1%}" if accuracy else f"{v:.4g}"

    def delta(d: float) -> str:
        return f"{d * 100:+.1f} pts" if accuracy else f"{d:+.4g}"

    challenger, reference = report_fields["challenger"], report_fields["reference"]
    answer = {
        "better": f"{challenger} performs better than {reference}",
        "worse": f"{challenger} performs worse than {reference}",
        "inconclusive": f"there is no reliable difference between {challenger} and {reference}",
    }[comparison.verdict]
    return (
        f"On the held-out test set ({report_fields['n_test_rows']} rows), {answer}: "
        f"{report_fields['metric']} {value(report_fields['challenger_score'])} vs "
        f"{value(report_fields['reference_score'])}, difference {delta(comparison.diff)} "
        f"(95% CI {delta(comparison.ci_low)} to {delta(comparison.ci_high)})."
    )


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
        queue = configs_for(plan, plan.levels, state["profile"].dataset_id, n_seeds)
        logger.info("plan: factor=%s levels=%s -> %d runs", plan.factor, plan.levels, len(queue))
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
        plan, rnd = state["plan"], state["round"]
        if rnd >= max_rounds:
            decision = Decision(
                round=rnd, action="conclude", decided_by="budget",
                rationale=f"Round budget reached ({max_rounds} rounds).",
            )
        else:
            decision = recommender.decide(state["question"], plan, state["analysis"], rnd, max_rounds)
        repo.append_decision(state["session_id"], decision)
        update: LoopState = {"decisions": [*state["decisions"], decision]}
        if decision.action == "explore":
            plan = Plan.model_validate({**plan.model_dump(), "levels": [*plan.levels, *decision.new_levels]})
            repo.save_plan(state["session_id"], plan)
            update.update(
                plan=plan,
                round=rnd + 1,
                queue=configs_for(plan, decision.new_levels, state["profile"].dataset_id, n_seeds),
            )
        logger.info("decide: round %d -> %s %s", rnd, decision.action, decision.new_levels or "")
        return update

    def finalize_node(state: LoopState) -> LoopState:
        plan, analysis, results = state["plan"], state["analysis"], state["results"]
        challenger = stats.best_challenger(plan, analysis)
        groups = stats.group_by_level(plan, results)
        comparison: Optional[Comparison] = (
            stats.compare(plan, challenger, groups, "test", analysis.higher_is_better)
            if challenger is not None else None
        )
        if comparison is None:
            raise RuntimeError(
                "No successful runs for the reference or for any other level, so there is nothing to compare."
            )
        fields = dict(
            challenger=plan.label(challenger),
            reference=plan.label(plan.reference),
            metric=analysis.metric,
            higher_is_better=analysis.higher_is_better,
            challenger_score=float(stats.condition_rows(groups[plan.label(challenger)], "test").mean()),
            reference_score=float(stats.condition_rows(groups[plan.label(plan.reference)], "test").mean()),
            majority_rate=analysis.majority_rate,
            n_test_rows=len(next(r for r in results if r.status == "ok").test_scores),
            rounds_run=state["round"],
            stopped_by=state["decisions"][-1].decided_by,
        )
        report = Report(comparison=comparison, headline=headline(fields, comparison), **fields)
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
        lambda s: "run_experiments" if s["decisions"][-1].action == "explore" else "finalize",
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
