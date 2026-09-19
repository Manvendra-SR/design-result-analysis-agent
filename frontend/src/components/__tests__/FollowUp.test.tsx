import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../../services/api";
import {
  makeCycle,
  makeRecommendation,
  makeSession,
} from "../../test/fixtures";
import { renderWithClient } from "../../test/utils";
import { SessionView } from "../SessionView";

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
    createSession: vi.fn(),
    listSessions: vi.fn(),
    listDatasets: vi.fn(),
  },
}));

const m = api as unknown as Record<string, ReturnType<typeof vi.fn>>;

beforeEach(() => {
  vi.clearAllMocks();
  m.getDataset.mockResolvedValue({
    dataset_id: "ds-1",
    original_filename: "churn.csv",
    task_type: "classification",
    n_classes: 2,
  });
  m.getExperiments.mockResolvedValue([]);
  m.getCycles.mockResolvedValue([
    makeCycle({ recommendation: makeRecommendation() }),
  ]);
  m.getRecommendation.mockResolvedValue(makeRecommendation());
});

/**
 * After an investigation finishes there was previously NO way to keep going —
 * the only affordance was "back to all investigations". A concluded session is
 * immutable, so continuing means starting a new investigation on the same
 * dataset, linked for provenance but starting from zero evidence.
 */
describe("asking a follow-up question after an investigation finishes", () => {
  it("is not offered while the investigation is still active", async () => {
    m.getSession.mockResolvedValue(makeSession({ run_phase: "idle" }));

    renderWithClient(
      <SessionView sessionId="sess-1" onBack={vi.fn()} onOpenSession={vi.fn()} />,
    );
    await screen.findByRole("button", { name: /run investigation/i });

    expect(
      screen.queryByRole("button", { name: /start follow-up/i }),
    ).not.toBeInTheDocument();
  });

  it("is offered once the agent concludes", async () => {
    m.getSession.mockResolvedValue(
      makeSession({
        status: "concluded",
        current_node: "concluded",
        termination_reason: "agent_concluded",
      }),
    );

    renderWithClient(
      <SessionView sessionId="sess-1" onBack={vi.fn()} onOpenSession={vi.fn()} />,
    );

    expect(
      await screen.findByRole("button", { name: /start follow-up/i }),
    ).toBeInTheDocument();
  });

  it("is offered — with different framing — when the cycle limit stopped it", async () => {
    m.getSession.mockResolvedValue(
      makeSession({
        status: "concluded",
        current_node: "concluded",
        termination_reason: "cycle_limit",
        cycle_count: 6,
      }),
    );

    renderWithClient(
      <SessionView sessionId="sess-1" onBack={vi.fn()} onOpenSession={vi.fn()} />,
    );

    await screen.findByRole("button", { name: /start follow-up/i });
    expect(
      screen.getByText(/stopped at its cycle limit without settling the question/i),
    ).toBeInTheDocument();
  });

  it("creates a linked investigation on the same dataset and opens it", async () => {
    const user = userEvent.setup();
    const onOpenSession = vi.fn();
    m.getSession.mockResolvedValue(
      makeSession({
        session_id: "sess-1",
        dataset_id: "ds-1",
        status: "concluded",
        current_node: "concluded",
        termination_reason: "agent_concluded",
      }),
    );
    m.createSession.mockResolvedValue({
      session_id: "sess-2",
      created_at: "2026-01-02T00:00:00Z",
    });

    renderWithClient(
      <SessionView
        sessionId="sess-1"
        onBack={vi.fn()}
        onOpenSession={onOpenSession}
      />,
    );

    const box = await screen.findByLabelText(/new research question/i);
    await user.type(box, "Does normalizing the features help?");
    await user.click(screen.getByRole("button", { name: /start follow-up/i }));

    await waitFor(() => expect(m.createSession).toHaveBeenCalled());
    // react-query appends its own context argument; the body is the first one.
    expect(m.createSession.mock.calls[0][0]).toEqual({
      research_question: "Does normalizing the features help?",
      dataset_id: "ds-1",
      parent_session_id: "sess-1",
    });
    await waitFor(() => expect(onOpenSession).toHaveBeenCalledWith("sess-2"));
  });
});
