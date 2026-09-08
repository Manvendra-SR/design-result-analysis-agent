import { screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { SessionView } from "../SessionView";
import { renderWithClient } from "../../test/utils";
import { api } from "../../services/api";
import { makeComparison, makeExperiment, makeSession } from "../../test/fixtures";

vi.mock("../../services/api", () => ({
  apiErrorMessage: (e: unknown) => String(e),
  isNotFound: () => true,
  errorStatus: () => 0,
  api: {
    getSession: vi.fn(),
    getDataset: vi.fn(),
    getExperiments: vi.fn(),
    getCycles: vi.fn(),
    getRecommendation: vi.fn(),
    runCycle: vi.fn(),
  },
}));

const m = api as unknown as Record<string, ReturnType<typeof vi.fn>>;

function stageState(name: string) {
  return document
    .querySelector(`[data-stage="${name}"]`)
    ?.getAttribute("data-state");
}

beforeEach(() => {
  vi.clearAllMocks();
  m.getDataset.mockResolvedValue({
    dataset_id: "ds-1",
    original_filename: "churn.csv",
    task_type: "classification",
    n_classes: 2,
  });
  m.getExperiments.mockResolvedValue([makeExperiment()]);
  m.getCycles.mockResolvedValue([
    {
      cycle_number: 1,
      plan_explanation: "vary dropout",
      experiments: [makeExperiment()],
      anomalies: [],
      statistical_comparisons: [makeComparison()],
      recommendation: null,
      continued: false,
    },
  ]);
  m.getRecommendation.mockRejectedValue(new Error("404"));
});

describe("SessionView — run state is backend-driven", () => {
  it("a new session (run_phase idle) is NOT shown as running", async () => {
    m.getSession.mockResolvedValue(makeSession({ run_phase: "idle" }));

    renderWithClient(<SessionView sessionId="sess-1" onBack={vi.fn()} />);
    await screen.findByRole("button", { name: /run investigation/i });

    expect(screen.queryByText(/investigation running/i)).not.toBeInTheDocument();
    expect(screen.getAllByText(/not running/i).length).toBeGreaterThan(0);
    expect(
      document.querySelector('[data-stage="planning"]')?.className,
    ).not.toContain("loop__stage--pulse");
    expect(document.querySelector(".loop")?.getAttribute("data-run-state")).toBe(
      "idle",
    );
  });

  it("run_phase 'running' pulses the stage named by current_node", async () => {
    m.getSession.mockResolvedValue(
      makeSession({ current_node: "analyzing", run_phase: "running", cycle_count: 1 }),
    );

    renderWithClient(<SessionView sessionId="sess-1" onBack={vi.fn()} />);

    await waitFor(() => expect(stageState("analyzing")).toBe("active"));
    expect(
      document.querySelector('[data-stage="analyzing"]')?.className,
    ).toContain("loop__stage--pulse");
    expect(stageState("planning")).toBe("done");
    expect(stageState("recommending")).toBe("pending");
    expect(screen.getAllByText(/investigation running/i).length).toBeGreaterThan(0);
  });

  it("run_phase 'failed' shows a failed state, not a running one", async () => {
    m.getSession.mockResolvedValue(
      makeSession({
        current_node: "recommending",
        run_phase: "failed",
        run_error: "Recommender LLM did not produce a usable recommendation",
        cycle_count: 2,
      }),
    );

    renderWithClient(<SessionView sessionId="sess-1" onBack={vi.fn()} />);

    await waitFor(() =>
      expect(
        document.querySelector('[data-stage="recommending"]')?.className,
      ).toContain("loop__stage--failed"),
    );
    expect(screen.queryByText(/investigation running/i)).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /resume investigation/i }),
    ).toBeEnabled();
    expect(screen.getAllByText(/did not produce a usable recommendation/i).length)
      .toBeGreaterThan(0);
  });

  it("shows the whole loop complete once the backend reports concluded", async () => {
    m.getSession.mockResolvedValue(
      makeSession({ current_node: "concluded", status: "concluded", cycle_count: 2 }),
    );

    renderWithClient(<SessionView sessionId="sess-1" onBack={vi.fn()} />);

    await waitFor(() => expect(stageState("recommending")).toBe("done"));
    expect(screen.getByText(/no further runs needed/i)).toBeInTheDocument();
    expect(screen.queryByText(/investigation running/i)).not.toBeInTheDocument();
  });
});
