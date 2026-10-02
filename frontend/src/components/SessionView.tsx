import { useQuery } from "@tanstack/react-query";

import { qk, useRunInvestigation, useSession } from "../hooks/queries";
import { api } from "../services/api";
import { DecisionLog } from "./DecisionLog";
import { ExperimentTable } from "./ExperimentTable";
import { PlanPanel } from "./PlanPanel";
import { ReportPanel } from "./ReportPanel";
import { ResearchQuestionDisplay } from "./ResearchQuestionDisplay";
import { ResultsPanel } from "./ResultsPanel";
import { RunControl } from "./RunControl";
import { Card, ErrorBox, Spinner } from "./ui";

/**
 * One investigation's dashboard. Everything comes from a single polled
 * endpoint (GET /api/sessions/{id}), read top to bottom in the order the
 * agent works: question -> design -> results -> decisions -> answer -> raw runs.
 */
export function SessionView({ sessionId, onBack }: { sessionId: string; onBack: () => void }) {
  const run = useRunInvestigation(sessionId);
  const session = useSession(sessionId);
  const dataset = useQuery({
    queryKey: qk.dataset(session.data?.dataset_id ?? "none"),
    queryFn: () => api.getDataset(session.data!.dataset_id),
    enabled: !!session.data?.dataset_id,
  });

  const back = (
    <button type="button" className="link-btn" onClick={onBack}>
      ← Back to all investigations
    </button>
  );

  if (session.isLoading) {
    return (
      <Card>
        <Spinner /> Loading investigation…
      </Card>
    );
  }
  if (session.error || !session.data) {
    return (
      <div className="stack">
        {back}
        <ErrorBox>Could not load this investigation.</ErrorBox>
      </div>
    );
  }

  const s = session.data;
  return (
    <div className="stack">
      {back}
      <ResearchQuestionDisplay session={s} dataset={dataset.data} />
      <RunControl session={s} starting={run.isPending} error={run.error} onRun={() => run.mutate()} />
      {s.report && <ReportPanel report={s.report} />}
      <PlanPanel plan={s.plan} nSeeds={s.n_seeds} />
      <ResultsPanel analysis={s.analysis} />
      <DecisionLog decisions={s.decisions} />
      <ExperimentTable experiments={s.experiments} />
    </div>
  );
}
