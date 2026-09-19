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
      {session.parent_session_id && (
        <p className="card__hint" style={{ marginTop: 0 }}>
          Follow-up investigation — it has its own experiments and evidence,
          separate from the investigation it follows.
        </p>
      )}
      <div className="kv">
        <div>
          <dt>Status</dt>
          <dd>
            {session.termination_reason === "cycle_limit"
              ? "stopped at cycle limit (unresolved)"
              : session.termination_reason === "agent_concluded"
                ? "concluded"
                : session.status}
          </dd>
        </div>
        <div>
          <dt>Current stage</dt>
          <dd>{session.current_node}</dd>
        </div>
        <div>
          <dt>Cycles</dt>
          <dd>
            {session.cycle_count} / {session.max_cycles} max
          </dd>
        </div>
        <div>
          <dt>Experiments</dt>
          <dd>{session.experiment_count} training runs</dd>
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
