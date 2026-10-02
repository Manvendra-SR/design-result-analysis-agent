import { formatTimestamp } from "../lib/format";
import type { DatasetProfile, SessionDetail } from "../types/api";
import { Badge, Card } from "./ui";

export function currentRound(session: SessionDetail): number {
  return Math.max(0, ...session.experiments.map((e) => e.round));
}

export function ResearchQuestionDisplay({
  session,
  dataset,
}: {
  session: SessionDetail;
  dataset?: DatasetProfile;
}) {
  return (
    <Card title={session.research_question} badge={<Badge variant="gradient">Research Question</Badge>}>
      <div className="kv">
        <div>
          <dt>Status</dt>
          <dd>{session.status}</dd>
        </div>
        <div>
          <dt>Round</dt>
          <dd>
            {currentRound(session)} / {session.max_rounds} max
          </dd>
        </div>
        <div>
          <dt>Training runs</dt>
          <dd>{session.experiments.length}</dd>
        </div>
        <div>
          <dt>Dataset</dt>
          <dd>{dataset ? dataset.original_filename : session.dataset_id}</dd>
        </div>
        {dataset && (
          <div>
            <dt>Task</dt>
            <dd>
              {dataset.task_type}
              {dataset.n_classes ? ` (${dataset.n_classes} classes)` : ""}
            </dd>
          </div>
        )}
        <div>
          <dt>Created</dt>
          <dd>{formatTimestamp(session.created_at)}</dd>
        </div>
      </div>
    </Card>
  );
}
