import { useQuery } from "@tanstack/react-query";

import {
  qk,
  useCycles,
  useExperiments,
  useRecommendation,
  useRunInvestigation,
  useSession,
} from "../hooks/queries";
import { deriveRunState } from "../lib/runState";
import { api } from "../services/api";
import { AdaptiveLoopVisualizer } from "./AdaptiveLoopVisualizer";
import { CycleHistory } from "./CycleHistory";
import { ExperimentTable } from "./ExperimentTable";
import { ExperimentVisualizer } from "./ExperimentVisualizer";
import { RecommendationPanel } from "./RecommendationPanel";
import { ResearchQuestionDisplay } from "./ResearchQuestionDisplay";
import { RunControl } from "./RunControl";
import { StatisticsPanel } from "./StatisticsPanel";
import { Card, ErrorBox, Spinner } from "./ui";

/**
 * SessionView (task 7.11): the per-session dashboard. It owns the polling
 * (`useSession`) and derives the true run state (idle / running / failed /
 * concluded) from the backend's `run_phase` plus the local run mutation -
 * see `lib/runState.ts`. That derived state, not `session.status`, drives the
 * "working" glow and the running banner.
 */
export function SessionView({
  sessionId,
  onBack,
}: {
  sessionId: string;
  onBack: () => void;
}) {
  const run = useRunInvestigation(sessionId);
  const session = useSession(sessionId, run.isPending);

  const notConcluded =
    !!session.data &&
    session.data.status !== "concluded" &&
    session.data.current_node !== "concluded";

  const runState = session.data ? deriveRunState(session.data, run) : "idle";

  // Auxiliary data keeps refreshing until the session is concluded.
  const live = notConcluded;
  const experiments = useExperiments(sessionId, live);
  const cycles = useCycles(sessionId, live);
  const recommendation = useRecommendation(sessionId, live);

  const dataset = useQuery({
    queryKey: qk.dataset(session.data?.dataset_id ?? "none"),
    queryFn: () => api.getDataset(session.data!.dataset_id),
    enabled: !!session.data?.dataset_id,
  });

  if (session.isLoading) {
    return (
      <Card>
        <Spinner /> Loading session…
      </Card>
    );
  }
  if (session.error || !session.data) {
    return (
      <div className="stack">
        <button type="button" className="link-btn" onClick={onBack}>
          ← Back to all investigations
        </button>
        <ErrorBox>Could not load this session.</ErrorBox>
      </div>
    );
  }

  const s = session.data;
  const allComparisons = (cycles.data ?? []).flatMap(
    (c) => c.statistical_comparisons,
  );

  return (
    <div className="stack">
      <button type="button" className="link-btn" onClick={onBack}>
        ← Back to all investigations
      </button>

      <ResearchQuestionDisplay session={s} dataset={dataset.data} />

      <AdaptiveLoopVisualizer
        currentNode={s.current_node}
        cycleCount={s.cycle_count}
        experimentCount={s.experiment_count}
        runState={runState}
      />

      <RunControl
        session={s}
        runState={runState}
        error={run.error}
        onRun={() => run.mutate()}
      />

      <RecommendationPanel recommendation={recommendation.data ?? null} />

      <StatisticsPanel comparisons={allComparisons} />

      <ExperimentVisualizer experiments={experiments.data ?? []} />

      <ExperimentTable experiments={experiments.data ?? []} />

      <CycleHistory cycles={cycles.data ?? []} />
    </div>
  );
}
