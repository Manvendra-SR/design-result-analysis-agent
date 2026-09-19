import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import {
  makeAnomaly,
  makeComparison,
  makeConditionSummary,
  makeCycle,
  makeExperiment,
  makeRecommendation,
  makeSkip,
} from "../../test/fixtures";
import { CycleHistory } from "../CycleHistory";
import { StatisticsPanel } from "../StatisticsPanel";

/**
 * The statistics panel used to assert a fixed reason for an empty comparison
 * list ("needs at least two conditions with >=2 successful experiments each")
 * which was frequently false - in the run that motivated this work both
 * conditions had 16 and 6 successful replicates and the real blocker was a
 * deterministic baseline. It now renders the backend's recorded reason.
 */
describe("StatisticsPanel reports real reasons, not guesses", () => {
  it("shows the recorded skip reason instead of a hardcoded precondition", () => {
    render(
      <StatisticsPanel
        comparisons={[]}
        skipped={[
          makeSkip({
            reason: "Both conditions are deterministic, so no t-test is defined.",
          }),
        ]}
      />,
    );

    expect(screen.getByText(/pairs that could not be compared/i)).toBeInTheDocument();
    expect(screen.getByText(/no t-test is defined/i)).toBeInTheDocument();
    expect(
      screen.queryByText(/at least two conditions with/i),
    ).not.toBeInTheDocument();
  });

  it("explains a one-sample test rather than hiding it", () => {
    render(
      <StatisticsPanel
        comparisons={[makeComparison({ test_type: "one_sample_t" })]}
      />,
    );

    expect(screen.getByText("1-sample")).toBeInTheDocument();
    expect(screen.getByText(/known constant/i)).toBeInTheDocument();
  });

  it("shows per-condition replicate counts and flags deterministic conditions", () => {
    render(
      <StatisticsPanel
        comparisons={[]}
        conditionSummaries={[
          makeConditionSummary({
            condition_name: "linear_baseline",
            n_successful: 6,
            std: 0,
            deterministic: true,
          }),
          makeConditionSummary({ condition_name: "mlp", n_anomalous: 2 }),
        ]}
      />,
    );

    expect(screen.getByText("linear_baseline")).toBeInTheDocument();
    expect(screen.getAllByText("deterministic").length).toBeGreaterThan(0);
    expect(screen.getByText(/\+2 flagged/)).toBeInTheDocument();
    expect(screen.getByText(/add no information/i)).toBeInTheDocument();
  });

  it("does not claim a blocker before any analysis has run", () => {
    render(<StatisticsPanel comparisons={[]} />);

    expect(screen.getByText(/analysis node runs after the first cycle/i)).toBeInTheDocument();
  });
});

/**
 * Per-cycle counts and cumulative prose previously sat side by side unlabelled,
 * so "Cycle 3 - 3 experiments - 0 anomalies" appeared directly above a
 * recommendation saying "two MLP runs were flagged as anomalous".
 */
describe("CycleHistory separates per-cycle facts from cumulative reasoning", () => {
  it("labels the recommendation and statistics as cumulative", () => {
    render(
      <CycleHistory
        cycles={[
          makeCycle({
            cycle_number: 3,
            experiments: [makeExperiment()],
            cumulative_experiment_count: 12,
            statistical_comparisons: [makeComparison()],
            recommendation: makeRecommendation({
              action: "run_more_experiments",
              explanation: "Two MLP runs were flagged as anomalous in cycle 1.",
            }),
            continued: true,
          }),
        ]}
      />,
    );

    expect(screen.getByText(/1 experiments \(12 total\)/)).toBeInTheDocument();
    expect(screen.getAllByText(/cumulative/i).length).toBeGreaterThanOrEqual(2);
  });

  it("distinguishes flags raised from flags withdrawn in a cycle", () => {
    render(
      <CycleHistory
        cycles={[
          makeCycle({
            cycle_number: 2,
            experiments: [makeExperiment()],
            cumulative_experiment_count: 6,
            anomalies_detected: [makeAnomaly({ explanation: "newly flagged" })],
            anomalies_resolved: [
              makeAnomaly({
                explanation: "cleared later",
                detected_cycle: 1,
                resolved_cycle: 2,
              }),
            ],
            open_anomaly_count: 1,
            recommendation: makeRecommendation(),
          }),
        ]}
      />,
    );

    expect(screen.getByText(/anomalies flagged in this cycle/i)).toBeInTheDocument();
    expect(screen.getByText(/anomalies withdrawn in this cycle/i)).toBeInTheDocument();
    expect(screen.getByText(/ordinary after all/i)).toBeInTheDocument();
    expect(screen.getByText(/1 flagged · 1 cleared · 1 open/)).toBeInTheDocument();
  });

  it("marks the cycle that hit the safety limit as stopped, not concluded", () => {
    render(
      <CycleHistory
        cycles={[
          makeCycle({
            cycle_number: 6,
            experiments: [makeExperiment()],
            cumulative_experiment_count: 24,
            recommendation: makeRecommendation({
              action: "run_more_experiments",
              explanation: "We should re-run with fresh seeds.",
            }),
            termination_reason: "cycle_limit",
          }),
        ]}
      />,
    );

    expect(screen.getByText(/stopped at limit/i)).toBeInTheDocument();
    expect(screen.getByText(/not a conclusion it reached/i)).toBeInTheDocument();
  });
});
