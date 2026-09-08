import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { SessionManager } from "../SessionManager";
import { renderWithClient } from "../../test/utils";
import { api } from "../../services/api";

vi.mock("../../services/api", () => ({
  apiErrorMessage: (e: unknown) => String(e),
  isNotFound: () => false,
  errorStatus: () => 0,
  api: {
    listSessions: vi.fn(),
    listDatasets: vi.fn(),
    createSession: vi.fn(),
    deleteSession: vi.fn(),
  },
}));

const mockApi = api as unknown as {
  listSessions: ReturnType<typeof vi.fn>;
  listDatasets: ReturnType<typeof vi.fn>;
  createSession: ReturnType<typeof vi.fn>;
  deleteSession: ReturnType<typeof vi.fn>;
};

beforeEach(() => {
  vi.clearAllMocks();
  mockApi.listDatasets.mockResolvedValue([
    {
      dataset_id: "ds-1",
      original_filename: "churn.csv",
      task_type: "classification",
      n_rows: 400,
      n_features: 5,
      task_type_source: "inferred",
    },
  ]);
  mockApi.listSessions.mockResolvedValue([
    {
      session_id: "sess-9",
      research_question: "Existing question?",
      status: "concluded",
      cycle_count: 2,
      experiment_count: 12,
      created_at: "2026-01-01T09:00:00Z",
    },
  ]);
});

describe("SessionManager", () => {
  it("lists existing sessions", async () => {
    renderWithClient(<SessionManager onOpenSession={vi.fn()} />);
    expect(await screen.findByText("Existing question?")).toBeInTheDocument();
  });

  it("deletes an investigation after confirmation", async () => {
    mockApi.deleteSession.mockResolvedValue({
      deleted: "session",
      id: "sess-9",
      sessions_deleted: 0,
      experiments_deleted: 3,
    });
    const confirmSpy = vi
      .spyOn(window, "confirm")
      .mockReturnValue(true);
    const user = userEvent.setup();
    renderWithClient(<SessionManager onOpenSession={vi.fn()} />);
    await screen.findByText("Existing question?");

    await user.click(
      screen.getByRole("button", { name: /delete investigation/i }),
    );

    await waitFor(() =>
      expect(mockApi.deleteSession).toHaveBeenCalledWith("sess-9"),
    );
    confirmSpy.mockRestore();
  });

  it("does not delete when the confirmation is dismissed", async () => {
    const confirmSpy = vi
      .spyOn(window, "confirm")
      .mockReturnValue(false);
    const user = userEvent.setup();
    renderWithClient(<SessionManager onOpenSession={vi.fn()} />);
    await screen.findByText("Existing question?");

    await user.click(
      screen.getByRole("button", { name: /delete investigation/i }),
    );

    expect(mockApi.deleteSession).not.toHaveBeenCalled();
    confirmSpy.mockRestore();
  });

  it("keeps the create button disabled until a question and dataset are set", async () => {
    renderWithClient(<SessionManager onOpenSession={vi.fn()} />);
    await screen.findByText("Existing question?");

    const create = screen.getByRole("button", { name: /start investigation/i });
    expect(create).toBeDisabled();

    const user = userEvent.setup();
    await user.type(
      screen.getByLabelText(/research question/i),
      "Does dropout help?",
    );
    expect(create).toBeDisabled(); // still no dataset

    await screen.findByRole("option", { name: /churn\.csv/i });
    await user.selectOptions(screen.getByLabelText(/dataset/i), "ds-1");
    expect(create).toBeEnabled();
  });

  it("creates a session and opens it", async () => {
    mockApi.createSession.mockResolvedValue({
      session_id: "sess-new",
      created_at: "2026-01-02T00:00:00Z",
    });
    const onOpen = vi.fn();
    const user = userEvent.setup();
    renderWithClient(<SessionManager onOpenSession={onOpen} />);
    await screen.findByText("Existing question?");

    await user.type(
      screen.getByLabelText(/research question/i),
      "Does dropout help?",
    );
    await screen.findByRole("option", { name: /churn\.csv/i });
    await user.selectOptions(screen.getByLabelText(/dataset/i), "ds-1");
    await user.click(screen.getByRole("button", { name: /start investigation/i }));

    await waitFor(
      () => expect(mockApi.createSession).toHaveBeenCalled(),
      { timeout: 8000 },
    );
    // react-query v5 passes a second (context) arg to mutationFn — assert on the body only.
    expect(mockApi.createSession.mock.calls[0][0]).toEqual({
      research_question: "Does dropout help?",
      dataset_id: "ds-1",
    });
    await waitFor(() => expect(onOpen).toHaveBeenCalledWith("sess-new"), {
      timeout: 8000,
    });
  });
});
