import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { DecisionLog } from "../DecisionLog";
import { ExperimentTable } from "../ExperimentTable";
import { ReportPanel } from "../ReportPanel";
import { ResultsPanel } from "../ResultsPanel";
import {
  makeAnalysis,
  makeComparison,
  makeCondition,
  makeDecision,
  makeExperiment,
  makeReport,
} from "../../test/fixtures";

describe("ResultsPanel", () => {
  it("marks the reference and shows each comparison's CI and verdict", () => {
    render(<ResultsPanel analysis={makeAnalysis()} />);
    expect(screen.getByText("reference")).toBeInTheDocument();
    expect(screen.getByText("+0.8 pts to +3.4 pts")).toBeInTheDocument();
    expect(screen.getByText("better")).toHaveClass("verdict--better");
  });

  it("reports MSE as lower-is-better numbers", () => {
    render(
      <ResultsPanel
        analysis={makeAnalysis({
          metric: "mse",
          higher_is_better: false,
          majority_rate: null,
          comparisons: [makeComparison({ diff: -0.5, ci_low: -0.8, ci_high: -0.2 })],
        })}
      />,
    );
    expect(screen.getByText(/lower is better/)).toBeInTheDocument();
    expect(screen.getByText("-0.500")).toBeInTheDocument();
  });

  it("in selection mode marks the leader and contenders and shows the change vs the parent", () => {
    render(
      <ResultsPanel
        analysis={makeAnalysis({
          mode: "selection",
          leader: "c3",
          contenders: ["c2", "c3"],
          conditions: [
            makeCondition({ id: "c1", label: "linear_baseline", family: "linear_baseline", change: null }),
            makeCondition({ id: "c2", label: "mlp", is_reference: false, change: null, contender: true }),
            makeCondition({ id: "c3", label: "mlp, hidden_size=128", parent: "c2", change: "hidden_size=128",
                            is_reference: false, contender: true, mean: 0.82 }),
          ],
          comparisons: [
            makeComparison({ a: "c2", b: "c3", anchor: "leader", diff: -0.01, ci_low: -0.03, ci_high: 0.01,
                             verdict: "inconclusive" }),
            makeComparison({ a: "c3", b: "c2", anchor: "parent", diff: 0.01 }),
          ],
        })}
      />,
    );
    expect(screen.getByText("leader")).toBeInTheDocument();
    expect(screen.getByText("contender")).toBeInTheDocument();
    expect(screen.getByText("Δ vs leader")).toBeInTheDocument();
    expect(screen.getByText("+1.0 pts (hidden_size=128)")).toBeInTheDocument();
  });
});

describe("ReportPanel", () => {
  it("shows the headline, both scores and how the run stopped", () => {
    render(<ReportPanel report={makeReport({ stopped_by: "budget", rounds_run: 3 })} />);
    expect(screen.getByText(/performs better than dropout=0/)).toBeInTheDocument();
    expect(screen.getByText("81.2%")).toBeInTheDocument();
    expect(screen.getByText("79.3%")).toBeInTheDocument();
    expect(screen.getByText(/3 rounds · stopped by the round budget/)).toBeInTheDocument();
  });

  it("shows the secondary comparison at the Bonferroni level in selection mode", () => {
    render(
      <ReportPanel
        report={makeReport({
          mode: "selection",
          runner_up: "random_forest",
          confidence: 0.975,
          secondary: makeComparison({ anchor: "runner_up", diff: 0.004, ci_low: -0.01, ci_high: 0.018,
                                      verdict: "inconclusive", confidence: 0.975 }),
          stopped_by: "settled",
        })}
      />,
    );
    expect(screen.getByText("vs random_forest · 97.5% CI")).toBeInTheDocument();
    expect(screen.getByText("vs reference · 97.5% CI")).toBeInTheDocument();
    expect(screen.getByText("inconclusive")).toBeInTheDocument();
    expect(screen.getByText(/every contender was one family/)).toBeInTheDocument();
  });

  it("works without an interpretation", () => {
    render(<ReportPanel report={makeReport({ interpretation: null })} />);
    expect(screen.queryByText("LLM Interpretation")).not.toBeInTheDocument();
  });
});

describe("DecisionLog", () => {
  it("lists refine and budget decisions", () => {
    render(
      <DecisionLog
        decisions={[
          makeDecision({ round: 1, action: "refine", parent: "c1", knob: "dropout", values: [0.1, 0.3],
                         new_candidates: ["c3", "c4"], rationale: "Try between." }),
          makeDecision({ round: 2, decided_by: "budget", rationale: "Round budget reached (2 rounds)." }),
        ]}
      />,
    );
    expect(screen.getByText("refine c1 · dropout = 0.1, 0.3")).toBeInTheDocument();
    expect(screen.getByText("round budget")).toBeInTheDocument();
  });
});

describe("ExperimentTable", () => {
  it("shows ok and failed runs and expands a row to its error", async () => {
    render(
      <ExperimentTable
        experiments={[
          makeExperiment(),
          makeExperiment({ status: "failed", metrics: null, error: "non-finite value" }),
        ]}
      />,
    );
    expect(document.querySelector(".status-pill--ok")).toBeInTheDocument();
    expect(screen.getByText("1 failed")).toBeInTheDocument();

    expect(screen.queryByText(/non-finite value/)).not.toBeInTheDocument();
    await userEvent.click(document.querySelectorAll("tbody tr.row-clickable")[1]);
    expect(screen.getByText(/non-finite value/)).toBeInTheDocument();
  });
});
