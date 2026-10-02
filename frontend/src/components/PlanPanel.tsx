import { formatLevel } from "../lib/format";
import type { Plan } from "../types/api";
import { Badge, Card, Empty, InterpretationTag } from "./ui";

/** The experiment design: one factor, its levels, the reference, the fixed base. */
export function PlanPanel({ plan, nSeeds }: { plan: Plan | null; nSeeds: number }) {
  if (!plan) {
    return (
      <Card title="Experiment design">
        <Empty>The planner designs the experiment when the investigation starts.</Empty>
      </Card>
    );
  }
  const base = plan.base;
  const fixed = [
    `model ${base.model_type}`,
    ...Object.entries(base.hyperparameters)
      .filter(([k]) => k !== plan.factor)
      .map(([k, v]) => `${k}=${v}`),
    base.normalize ? "normalized" : "not normalized",
  ];

  return (
    <Card title="Experiment design" badge={<InterpretationTag />}>
      <div className="kv">
        <div>
          <dt>Factor varied</dt>
          <dd className="mono">{plan.factor}</dd>
        </div>
        <div>
          <dt>Reference</dt>
          <dd className="mono">{formatLevel(plan.reference)}</dd>
        </div>
        <div>
          <dt>Held fixed</dt>
          <dd className="mono">{fixed.join(" · ")}</dd>
        </div>
        <div>
          <dt>Seeds</dt>
          <dd>{nSeeds} per MLP level (linear_baseline: 1, it is seed-independent)</dd>
        </div>
      </div>
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", margin: "14px 0" }}>
        {plan.levels.map((level) => (
          <Badge key={String(level)} variant={level === plan.reference ? "gradient" : "muted"}>
            {formatLevel(level)}
          </Badge>
        ))}
      </div>
      <p className="llm-text">{plan.rationale}</p>
    </Card>
  );
}
