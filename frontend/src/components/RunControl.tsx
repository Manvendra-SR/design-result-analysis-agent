import { isUnstarted, type RunState } from "../lib/runState";
import { apiErrorMessage } from "../services/api";
import type { SessionDetail } from "../types/api";
import { Badge, Card, ErrorBox, Spinner } from "./ui";

/**
 * The single workflow trigger. The Phase 5/6 backend runs the entire
 * investigation in one POST /run-cycle (many cycles, no human gate), so this
 * is "Run Investigation" - or "Resume" if a previous run stopped partway or
 * failed - rather than the per-cycle "Approve and Run" of the original spec.
 *
 * The button, banner and error shown depend on `runState` (derived from the
 * backend's `run_phase` + the local mutation), NOT on `session.status`:
 *
 *   idle       -> "Run"/"Resume" button, no banner
 *   running    -> disabled button + "running" banner
 *   failed     -> the persisted error + a "Resume" button
 *   concluded  -> a completion message, no button
 */
export function RunControl({
  session,
  runState,
  error,
  onRun,
}: {
  session: SessionDetail;
  runState: RunState;
  error: unknown;
  onRun: () => void;
}) {
  const fresh = isUnstarted(session);
  const runLabel = fresh ? "Run Investigation" : "Resume Investigation";
  // The persisted reason (survives refresh) or the just-failed mutation error.
  const failureMessage =
    session.run_error ?? (error != null ? apiErrorMessage(error) : null);

  return (
    <Card
      title="Run"
      badge={<Badge variant="gradient">Control</Badge>}
      hint="One run executes every adaptive cycle server-side until the Recommender concludes or the safety cap is reached."
    >
      {runState === "concluded" ? (
        <div className="concluded-banner" role="status">
          <span aria-hidden="true">✓</span>
          <span>Investigation complete — no further runs needed.</span>
        </div>
      ) : (
        <>
          {runState === "failed" && failureMessage && (
            <div style={{ marginBottom: 12 }}>
              <ErrorBox>
                Last run failed at the <strong>{session.current_node}</strong>{" "}
                stage: {failureMessage}
                <br />
                The session stays resumable — Resume re-runs from that stage. A
                local model occasionally returns an unusable recommendation;
                retrying often clears it.
              </ErrorBox>
            </div>
          )}

          <button
            type="button"
            className="btn btn--primary"
            onClick={onRun}
            disabled={runState === "running"}
          >
            {runState === "running" && <Spinner onPrimary />}
            {runState === "running"
              ? "Running investigation…"
              : runState === "failed"
                ? "Resume Investigation"
                : runLabel}
          </button>

          {runState === "running" && (
            <div className="running-banner" role="status">
              <span className="running-dot" aria-hidden="true" />
              <span>
                The backend is working through the loop. Progress below updates
                as each stage completes; this request finishes when the
                investigation concludes.
              </span>
            </div>
          )}

          {runState === "idle" && !fresh && (
            <p className="card__hint" style={{ marginTop: 12 }}>
              This session is partway through (stage:{" "}
              <span className="mono">{session.current_node}</span>, cycle{" "}
              <span className="mono">{session.cycle_count}</span>). Resume
              continues it to a conclusion.
            </p>
          )}
        </>
      )}
    </Card>
  );
}
