import { apiErrorMessage } from "../services/api";
import type { SessionDetail } from "../types/api";
import { currentRound } from "./ResearchQuestionDisplay";
import { Badge, Card, ErrorBox, Spinner } from "./ui";

/**
 * Starts the investigation. The backend runs it in the background; this card
 * reflects `session.status` (pending / running / done / failed), which is
 * persisted, so it is right after a page refresh too.
 */
export function RunControl({
  session,
  starting,
  error,
  onRun,
}: {
  session: SessionDetail;
  starting: boolean;
  error: unknown;
  onRun: () => void;
}) {
  const { status } = session;
  const running = status === "running" || starting;

  return (
    <Card
      title="Run"
      badge={<Badge variant="gradient">Control</Badge>}
      hint={`The agent plans the experiment, then runs up to ${session.max_rounds} rounds (${session.n_seeds} seeds per MLP condition) before reporting.`}
    >
      {status === "done" ? (
        <div className="concluded-banner" role="status">
          <span aria-hidden="true">✓</span>
          <span>Investigation complete — see the result below.</span>
        </div>
      ) : (
        <>
          {status === "failed" && session.error && (
            <div style={{ marginBottom: 12 }}>
              <ErrorBox>Last run failed: {session.error}</ErrorBox>
            </div>
          )}
          {error != null && (
            <div style={{ marginBottom: 12 }}>
              <ErrorBox>{apiErrorMessage(error)}</ErrorBox>
            </div>
          )}
          <button type="button" className="btn btn--primary" onClick={onRun} disabled={running}>
            {running && <Spinner onPrimary />}
            {running ? "Running…" : status === "failed" ? "Start over" : "Run investigation"}
          </button>
          {running && (
            <div className="running-banner" role="status">
              <span className="running-dot" aria-hidden="true" />
              <span>
                {session.plan
                  ? `Round ${Math.max(1, currentRound(session))} of up to ${session.max_rounds} · ${session.experiments.length} runs finished`
                  : "Planning the experiment…"}
              </span>
            </div>
          )}
        </>
      )}
    </Card>
  );
}
