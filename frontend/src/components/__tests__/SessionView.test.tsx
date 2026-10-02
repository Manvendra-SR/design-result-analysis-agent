import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { SessionView } from "../SessionView";
import { renderWithClient } from "../../test/utils";
import { api } from "../../services/api";
import {
  makeAnalysis,
  makeDecision,
  makeExperiment,
  makePlan,
  makeReport,
  makeSession,
} from "../../test/fixtures";

vi.mock("../../services/api", () => ({
  apiErrorMessage: (e: unknown) => String(e),
  api: { getSession: vi.fn(), getDataset: vi.fn(), runSession: vi.fn() },
}));

const m = api as unknown as Record<string, ReturnType<typeof vi.fn>>;

beforeEach(() => {
  vi.clearAllMocks();
  m.getDataset.mockResolvedValue({
    dataset_id: "ds-1",
    original_filename: "diabetes.csv",
    task_type: "classification",
    n_classes: 3,
  });
});

describe("SessionView", () => {
  it("a new investigation offers to run and explains what will appear", async () => {
    m.getSession.mockResolvedValue(makeSession());
    renderWithClient(<SessionView sessionId="sess-1" onBack={vi.fn()} />);

    expect(await screen.findByRole("button", { name: "Run investigation" })).toBeEnabled();
    expect(screen.getByText(/planner designs the experiment/i)).toBeInTheDocument();
    expect(screen.queryByText(/held-out test set/i)).not.toBeInTheDocument();
  });

  it("starting a run calls the API", async () => {
    m.getSession.mockResolvedValue(makeSession());
    m.runSession.mockResolvedValue(makeSession({ status: "running" }));
    renderWithClient(<SessionView sessionId="sess-1" onBack={vi.fn()} />);

    await userEvent.click(await screen.findByRole("button", { name: "Run investigation" }));
    await waitFor(() => expect(m.runSession).toHaveBeenCalledWith("sess-1"));
  });

  it("while running it shows progress and disables the button", async () => {
    m.getSession.mockResolvedValue(
      makeSession({ status: "running", plan: makePlan(), experiments: [makeExperiment(), makeExperiment()] }),
    );
    renderWithClient(<SessionView sessionId="sess-1" onBack={vi.fn()} />);

    expect(await screen.findByRole("button", { name: /running/i })).toBeDisabled();
    expect(screen.getByText("Round 1 of up to 3 · 2 runs finished")).toBeInTheDocument();
  });

  it("a failed run shows the error and offers to start over", async () => {
    m.getSession.mockResolvedValue(makeSession({ status: "failed", error: "LLMError: rate limited" }));
    renderWithClient(<SessionView sessionId="sess-1" onBack={vi.fn()} />);

    expect(await screen.findByText(/rate limited/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start over" })).toBeEnabled();
  });

  it("a finished investigation shows design, validation results, decisions and the test result", async () => {
    m.getSession.mockResolvedValue(
      makeSession({
        status: "done",
        plan: makePlan(),
        analysis: makeAnalysis(),
        decisions: [makeDecision()],
        report: makeReport(),
        experiments: [makeExperiment()],
      }),
    );
    renderWithClient(<SessionView sessionId="sess-1" onBack={vi.fn()} />);

    expect(await screen.findByText(/Investigation complete/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /run|start over/i })).not.toBeInTheDocument();
    expect(screen.getByText(/performs better than dropout=0/)).toBeInTheDocument(); // report headline
    expect(screen.getByText("Dropout helps a little on this dataset.")).toBeInTheDocument();
    expect(screen.getByText("After round 1")).toBeInTheDocument();
    expect(screen.getByText("+2.1 pts")).toBeInTheDocument(); // validation diff
    expect(screen.getByText("(+1 failed)")).toBeInTheDocument();
    expect(screen.getByText(/most common class scores 60.0%/)).toBeInTheDocument();
  });
});
