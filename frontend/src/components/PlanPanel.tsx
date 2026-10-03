import type { Plan } from "../types/api";
import { Badge, Card, Empty, InterpretationTag } from "./ui";

/**
 * The experiment design. Effect mode: one factor, its levels, the reference,
 * the fixed base. Selection mode: the model families compared at their
 * defaults, the simplest one as reference. Refinements appear as candidates
 * with their parent and the one knob they change.
 */
export function PlanPanel({ plan, nSeeds }: { plan: Plan | null; nSeeds: number }) {
  if (!plan) {
    return (
      <Card title="Experiment design">
        <Empty>The planner designs the experiment when the investigation starts.</Empty>
      </Card>
    );
  }
  const reference = plan.candidates.find((c) => c.id === plan.reference);
  const base = plan.base;
  const fixed = base
    ? [
        `model ${base.model_type}`,
        ...Object.entries(base.hyperparameters)
          .filter(([k]) => k !== plan.factor)
          .map(([k, v]) => `${k}=${v}`),
        base.normalize === null ? "family-default scaling" : base.normalize ? "normalized" : "not normalized",
      ]
    : [];

  return (
    <Card title="Experiment design" badge={<InterpretationTag />}>
      <div className="kv">
        <div>
          <dt>Question type</dt>
          <dd>{plan.mode === "selection" ? "Which model is best (selection)" : "Does one factor matter (effect)"}</dd>
        </div>
        {plan.mode === "effect" && (
          <div>
            <dt>Factor varied</dt>
            <dd className="mono">{plan.factor}</dd>
          </div>
        )}
        <div>
          <dt>Reference</dt>
          <dd className="mono">{reference?.label}</dd>
        </div>
        {plan.mode === "effect" && (
          <div>
            <dt>Held fixed</dt>
            <dd className="mono">{fixed.join(" · ")}</dd>
          </div>
        )}
        <div>
          <dt>Seeds</dt>
          <dd>{nSeeds} per random forest / MLP candidate (linear model and decision tree: 1, they are seed-independent)</dd>
        </div>
      </div>
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", margin: "14px 0" }}>
        {plan.candidates.map((c) => (
          <Badge key={c.id} variant={c.id === plan.reference ? "gradient" : "muted"}>
            {c.id} · {c.label}
          </Badge>
        ))}
      </div>
      <p className="llm-text">{plan.rationale}</p>
    </Card>
  );
}
