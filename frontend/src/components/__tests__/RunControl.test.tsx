import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { RunControl } from "../RunControl";
import { makeSession } from "../../test/fixtures";

describe("RunControl", () => {
  it("offers 'Run Investigation' for a fresh (idle, unstarted) session", async () => {
    const user = userEvent.setup();
    const onRun = vi.fn();
    render(
      <RunControl
        session={makeSession()}
        runState="idle"
        error={null}
        onRun={onRun}
      />,
    );
    const btn = screen.getByRole("button", { name: /run investigation/i });
    await user.click(btn);
    expect(onRun).toHaveBeenCalledOnce();
    expect(screen.queryByText(/investigation running/i)).not.toBeInTheDocument();
  });

  it("offers 'Resume Investigation' for an idle session that is partway through", () => {
    render(
      <RunControl
        session={makeSession({ current_node: "analyzing", cycle_count: 1, experiment_count: 8 })}
        runState="idle"
        error={null}
        onRun={vi.fn()}
      />,
    );
    expect(
      screen.getByRole("button", { name: /resume investigation/i }),
    ).toBeInTheDocument();
    expect(screen.getByText(/partway through/i)).toBeInTheDocument();
  });

  it("disables the button and shows a running banner only while running", () => {
    render(
      <RunControl
        session={makeSession({ current_node: "executing", cycle_count: 1, run_phase: "running" })}
        runState="running"
        error={null}
        onRun={vi.fn()}
      />,
    );
    expect(screen.getByRole("button")).toBeDisabled();
    expect(
      screen.getByText(/backend is working through the loop/i),
    ).toBeInTheDocument();
  });

  it("shows the persisted failure and a Resume button when the run failed", () => {
    render(
      <RunControl
        session={makeSession({
          current_node: "recommending",
          cycle_count: 2,
          run_phase: "failed",
          run_error: "Recommender LLM did not produce a usable recommendation",
        })}
        runState="failed"
        error={null}
        onRun={vi.fn()}
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent(
      /did not produce a usable recommendation/,
    );
    expect(
      screen.getByRole("button", { name: /resume investigation/i }),
    ).toBeEnabled();
    expect(screen.queryByText(/investigation running/i)).not.toBeInTheDocument();
  });

  it("shows completion instead of a run button once concluded", () => {
    render(
      <RunControl
        session={makeSession({ status: "concluded", current_node: "concluded" })}
        runState="concluded"
        error={null}
        onRun={vi.fn()}
      />,
    );
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
    expect(screen.getByText(/no further runs needed/i)).toBeInTheDocument();
  });

  it("falls back to the mutation error when there is no persisted run_error", () => {
    render(
      <RunControl
        session={makeSession({ current_node: "planning", run_phase: "failed" })}
        runState="failed"
        error={new Error("network down")}
        onRun={vi.fn()}
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent(/network down/);
  });
});
