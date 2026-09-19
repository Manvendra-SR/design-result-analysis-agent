import { screen } from "@testing-library/react";
import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { deriveRunState, isFinished } from "../../lib/runState";
import { describeConfig } from "../../lib/format";
import { makeConfig, makeRecommendation, makeSession } from "../../test/fixtures";
import { RecommendationPanel } from "../RecommendationPanel";
import { RunControl } from "../RunControl";

const idleMutation = { isPending: false, isError: false };

/**
 * A run stopped by the safety cap is finished but UNRESOLVED. The backend now
 * stores the agent's recommendation verbatim - so its action is still
 * "run_more_experiments" - and records the stop separately. The UI must never
 * render that combination as a conclusion.
 */
describe("safety-limit termination is distinct from a real conclusion", () => {
  it("derives a distinct run state for a capped run", () => {
    const capped = makeSession({
      status: "concluded",
      current_node: "concluded",
      termination_reason: "cycle_limit",
    });
    const concluded = makeSession({
      status: "concluded",
      current_node: "concluded",
      termination_reason: "agent_concluded",
    });

    expect(deriveRunState(capped, idleMutation)).toBe("stopped_at_limit");
    expect(deriveRunState(concluded, idleMutation)).toBe("concluded");
    expect(isFinished("stopped_at_limit")).toBe(true);
    expect(isFinished("concluded")).toBe(true);
  });

  it("RunControl warns instead of congratulating when the cap stopped the run", () => {
    render(
      <RunControl
        session={makeSession({
          status: "concluded",
          current_node: "concluded",
          termination_reason: "cycle_limit",
          cycle_count: 6,
        })}
        runState="stopped_at_limit"
        error={null}
        onRun={vi.fn()}
      />,
    );

    expect(screen.getByText(/stopped at the 6-cycle safety limit/i)).toBeInTheDocument();
    expect(screen.queryByText(/investigation complete/i)).not.toBeInTheDocument();
    // No run button: the session is terminal either way.
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("RunControl still congratulates a genuine conclusion", () => {
    render(
      <RunControl
        session={makeSession({ status: "concluded", current_node: "concluded" })}
        runState="concluded"
        error={null}
        onRun={vi.fn()}
      />,
    );

    expect(screen.getByText(/investigation complete/i)).toBeInTheDocument();
  });

  it("RecommendationPanel labels a capped run's 'run more' output honestly", () => {
    render(
      <RecommendationPanel
        recommendation={makeRecommendation({
          action: "run_more_experiments",
          recommended_experiments: [makeConfig()],
          explanation: "We should re-run the MLP with fresh seeds.",
        })}
        terminationReason="cycle_limit"
        maxCycles={6}
      />,
    );

    expect(screen.getByText(/not a settled answer/i)).toBeInTheDocument();
    expect(screen.getByText(/stopped at limit/i)).toBeInTheDocument();
    expect(screen.getByText(/never run/i)).toBeInTheDocument();
    // It must NOT claim the recommender judged the evidence sufficient.
    expect(
      screen.queryByText(/judged the evidence sufficient/i),
    ).not.toBeInTheDocument();
  });

  it("RecommendationPanel shows a real conclusion as concluded", () => {
    render(
      <RecommendationPanel
        recommendation={makeRecommendation({ action: "conclude" })}
        terminationReason="agent_concluded"
        maxCycles={6}
      />,
    );

    expect(screen.getByText(/judged the evidence sufficient/i)).toBeInTheDocument();
    expect(screen.queryByText(/not a settled answer/i)).not.toBeInTheDocument();
  });
});

describe("experiment vs epoch presentation", () => {
  it("renders epochs as a quantity, not a bare key=value count", () => {
    const label = describeConfig(
      makeConfig({ hyperparameters: { dropout: 0.1, epochs: 20 } }),
    );

    expect(label).toContain("20 epochs");
    expect(label).not.toContain("epochs=20");
    // seeds stay as key=value - they identify a replicate, not a quantity
    expect(label).toContain("seed=42");
  });
});
