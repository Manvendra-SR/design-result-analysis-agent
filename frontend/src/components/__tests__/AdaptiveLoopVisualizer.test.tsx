import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { AdaptiveLoopVisualizer } from "../AdaptiveLoopVisualizer";
import type { RunState } from "../../lib/runState";
import type { WorkflowNode } from "../../types/api";

function renderViz(currentNode: WorkflowNode, runState: RunState) {
  return render(
    <AdaptiveLoopVisualizer
      currentNode={currentNode}
      cycleCount={1}
      experimentCount={6}
      runState={runState}
    />,
  );
}

function stage(name: string) {
  return document.querySelector(`[data-stage="${name}"]`) as HTMLElement;
}

describe("AdaptiveLoopVisualizer", () => {
  it("marks the stage named by current_node active while running, earlier done, later pending", () => {
    renderViz("analyzing", "running");

    expect(stage("planning").dataset.state).toBe("done");
    expect(stage("executing").dataset.state).toBe("done");
    expect(stage("analyzing").dataset.state).toBe("active");
    expect(stage("recommending").dataset.state).toBe("pending");
    expect(stage("analyzing").className).toContain("loop__stage--active");
    expect(stage("analyzing").className).toContain("loop__stage--pulse");
  });

  it("moves the active highlight when current_node changes (backend-driven)", () => {
    const { rerender } = renderViz("planning", "running");
    expect(stage("planning").dataset.state).toBe("active");

    rerender(
      <AdaptiveLoopVisualizer
        currentNode="recommending"
        cycleCount={1}
        experimentCount={6}
        runState="running"
      />,
    );
    expect(stage("planning").dataset.state).toBe("done");
    expect(stage("recommending").dataset.state).toBe("active");
    expect(stage("analyzing").dataset.state).toBe("done");
  });

  it("does NOT pulse or show a running banner when the run state is idle", () => {
    renderViz("planning", "idle");

    // current_node is still reflected, but as 'current', not 'working'
    expect(stage("planning").dataset.state).toBe("active");
    expect(stage("planning").className).toContain("loop__stage--active");
    expect(stage("planning").className).not.toContain("loop__stage--pulse");
    expect(screen.queryByText(/investigation running/i)).not.toBeInTheDocument();
    expect(screen.getByText(/not running/i)).toBeInTheDocument();
  });

  it("shows a stalled/failed marker (no pulse) when the run failed", () => {
    renderViz("recommending", "failed");

    expect(stage("recommending").dataset.state).toBe("active");
    expect(stage("recommending").className).toContain("loop__stage--failed");
    expect(stage("recommending").className).not.toContain("loop__stage--pulse");
    expect(screen.getByText(/nothing is working right now/i)).toBeInTheDocument();
    expect(screen.queryByText(/investigation running/i)).not.toBeInTheDocument();
  });

  it("shows every stage complete and a concluded banner when the workflow is done", () => {
    renderViz("concluded", "concluded");

    for (const name of [
      "planning",
      "executing",
      "validating",
      "analyzing",
      "recommending",
      "concluded",
    ]) {
      expect(stage(name).dataset.state).toBe("done");
    }
    expect(screen.getByText(/adaptive loop has concluded/i)).toBeInTheDocument();
  });

  it("exposes the backend node and run state on the container", () => {
    renderViz("validating", "running");
    const loop = document.querySelector(".loop");
    expect(loop?.getAttribute("data-current-node")).toBe("validating");
    expect(loop?.getAttribute("data-run-state")).toBe("running");
  });
});
