import { Fragment } from "react";

import { STAGE_ROLE, WORKFLOW_STAGES } from "../lib/format";
import type { RunState } from "../lib/runState";
import type { WorkflowNode } from "../types/api";
import { Badge, Card } from "./ui";

type StageState = "done" | "active" | "pending";

/**
 * AdaptiveLoopVisualizer (design.md task 7.6 + the live active-agent
 * requirement).
 *
 * The active stage is `currentNode`, taken straight from the polled
 * `sessions.current_node` - never a local timer. The *animation* on it is
 * gated on `runState`: it only pulses while a run is genuinely in progress
 * (`runState === "running"`). When the session is idle, failed or concluded
 * the stage markers still reflect the real `current_node`, but nothing is
 * shown as "working".
 */
export function AdaptiveLoopVisualizer({
  currentNode,
  cycleCount,
  experimentCount,
  runState,
}: {
  currentNode: WorkflowNode;
  cycleCount: number;
  experimentCount: number;
  runState: RunState;
}) {
  const concluded = runState === "concluded" || currentNode === "concluded";
  const running = runState === "running";
  const failed = runState === "failed";
  const activeIndex = WORKFLOW_STAGES.indexOf(currentNode);

  function stateFor(index: number): StageState {
    if (concluded) return "done";
    if (activeIndex === -1) return "pending";
    if (index < activeIndex) return "done";
    if (index === activeIndex) return "active";
    return "pending";
  }

  return (
    <Card
      title="Adaptive Loop"
      badge={<Badge variant="gradient">Workflow</Badge>}
      right={
        concluded ? (
          <Badge variant="ok">Concluded</Badge>
        ) : failed ? (
          <Badge variant="danger">Run failed</Badge>
        ) : running ? (
          <Badge variant="warn">Running</Badge>
        ) : (
          <Badge variant="muted">Not running</Badge>
        )
      }
    >
      <div
        className="loop"
        role="list"
        aria-label="Adaptive workflow stages"
        data-current-node={concluded ? "concluded" : currentNode}
        data-run-state={runState}
      >
        {WORKFLOW_STAGES.map((stage, i) => {
          const st = stateFor(i);
          return (
            <Fragment key={stage}>
              {i > 0 && (
                <span className="loop__connector" aria-hidden="true">
                  →
                </span>
              )}
              <div
                role="listitem"
                aria-current={st === "active" ? "step" : undefined}
                data-stage={stage}
                data-state={st}
                className={
                  "loop__stage" +
                  (st === "active" ? " loop__stage--active" : "") +
                  (st === "active" && running ? " loop__stage--pulse" : "") +
                  (st === "active" && failed ? " loop__stage--failed" : "") +
                  (st === "done" ? " loop__stage--done" : "") +
                  (st === "pending" ? " loop__stage--pending" : "")
                }
              >
                <div className="loop__stage-name">{stage}</div>
                <div className="loop__stage-role">{STAGE_ROLE[stage]}</div>
                <div className="loop__stage-state">
                  {st === "active"
                    ? running
                      ? "working"
                      : failed
                        ? "stalled"
                        : "current"
                    : st === "done"
                      ? "done"
                      : "waiting"}
                </div>
              </div>
            </Fragment>
          );
        })}
        <span className="loop__connector" aria-hidden="true">
          →
        </span>
        <div
          role="listitem"
          data-stage="concluded"
          data-state={concluded ? "done" : "pending"}
          className={
            "loop__stage" +
            (concluded ? " loop__stage--done" : " loop__stage--pending")
          }
        >
          <div className="loop__stage-name">concluded</div>
          <div className="loop__stage-role">{STAGE_ROLE.concluded}</div>
          <div className="loop__stage-state">{concluded ? "reached" : "—"}</div>
        </div>
      </div>

      <div className="loop-summary">
        <span>
          Cycles completed: <span className="mono">{cycleCount}</span>
        </span>
        <span>
          Experiments run: <span className="mono">{experimentCount}</span>
        </span>
        <span>
          Current stage:{" "}
          <span className="mono">{concluded ? "concluded" : currentNode}</span>
        </span>
      </div>

      {running && (
        <div className="running-banner" role="status">
          <span className="running-dot" aria-hidden="true" />
          <span>
            Investigation running — <strong>{currentNode}</strong> stage is
            active. This view updates from the backend as each agent finishes.
          </span>
        </div>
      )}

      {failed && (
        <div className="error-box" role="status" style={{ marginTop: 16 }}>
          The run stopped at the <strong>{currentNode}</strong> stage. Nothing
          is working right now — see the error below and use Resume to
          continue.
        </div>
      )}

      {concluded && (
        <div className="concluded-banner" role="status">
          <span aria-hidden="true">✓</span>
          <span>
            The adaptive loop has concluded after {cycleCount}{" "}
            {cycleCount === 1 ? "cycle" : "cycles"}.
          </span>
        </div>
      )}
    </Card>
  );
}
