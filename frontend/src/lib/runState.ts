/*
 * The frontend's view of "what is this investigation doing right now".
 *
 * `session.status` only says active-vs-concluded, so it cannot tell a
 * brand-new session apart from one mid-run or one that failed. This derives
 * a proper 4-state value from the backend's `run_phase` (persisted by the
 * executor) plus the local run mutation, so the UI is correct on first load,
 * during a run, after a failure, and after a page refresh.
 */
import type { SessionDetail } from "../types/api";

export type RunState = "concluded" | "running" | "failed" | "idle";

export interface MutationLike {
  isPending: boolean;
  isError: boolean;
}

export function deriveRunState(
  session: SessionDetail,
  runMutation: MutationLike,
): RunState {
  if (session.status === "concluded" || session.current_node === "concluded") {
    return "concluded";
  }
  // isPending = this tab is running it; run_phase = the backend says a run is
  // in progress (covers a run started elsewhere, or a page refresh mid-run).
  if (runMutation.isPending || session.run_phase === "running") {
    return "running";
  }
  if (session.run_phase === "failed" || runMutation.isError) {
    return "failed";
  }
  return "idle";
}

/** A fresh session that has never been run. */
export function isUnstarted(session: SessionDetail): boolean {
  return (
    session.cycle_count === 0 &&
    session.current_node === "planning" &&
    session.experiment_count === 0
  );
}
