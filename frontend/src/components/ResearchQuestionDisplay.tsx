import { formatTimestamp } from "../lib/format";
import type { DatasetProfile, SessionDetail } from "../types/api";
import { Badge, Card } from "./ui";

export function ResearchQuestionDisplay({
  session,
  dataset,
}: {
  session: SessionDetail;
  dataset?: DatasetProfile;
}) {
  return (
    <Card
      title={session.research_question}
      badge={<Badge variant="gradient">Research Question</Badge>}
    >
      <div className="kv">
        <div>
          <dt>Status</dt>
          <dd>{session.status}</dd>
        </div>
        <div>
          <dt>Current stage</dt>
          <dd>{session.current_node}</dd>
        </div>
        <div>
          <dt>Cycles</dt>
          <dd>{session.cycle_count}</dd>
        </div>
        <div>
          <dt>Experiments</dt>
          <dd>{session.experiment_count}</dd>
        </div>
        <div>
          <dt>Dataset</dt>
          <dd>{dataset ? dataset.original_filename : session.dataset_id}</dd>
        </div>
        {dataset && (
          <div>
            <dt>Task type</dt>
            <dd>
              {dataset.task_type}
              {dataset.task_type === "classification" && dataset.n_classes
                ? ` (${dataset.n_classes} classes)`
                : ""}
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
