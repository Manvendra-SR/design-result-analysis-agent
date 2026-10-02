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
});

describe("ReportPanel", () => {
  it("shows the headline, both scores and how the run stopped", () => {
    render(<ReportPanel report={makeReport({ stopped_by: "budget", rounds_run: 3 })} />);
    expect(screen.getByText(/performs better than dropout=0/)).toBeInTheDocument();
    expect(screen.getByText("81.2%")).toBeInTheDocument();
    expect(screen.getByText("79.3%")).toBeInTheDocument();
    expect(screen.getByText(/3 rounds · stopped by the round budget/)).toBeInTheDocument();
  });

  it("works without an interpretation", () => {
    render(<ReportPanel report={makeReport({ interpretation: null })} />);
    expect(screen.queryByText("LLM Interpretation")).not.toBeInTheDocument();
  });
});

describe("DecisionLog", () => {
  it("lists explore and budget decisions", () => {
    render(
      <DecisionLog
        decisions={[
          makeDecision({ round: 1, action: "explore", new_levels: [0.1, 0.3], rationale: "Try between." }),
          makeDecision({ round: 2, decided_by: "budget", rationale: "Round budget reached (2 rounds)." }),
        ]}
      />,
    );
    expect(screen.getByText("explore 0.1, 0.3")).toBeInTheDocument();
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
