import { formatLevel } from "../lib/format";
import type { Decision } from "../types/api";
import { Badge, Card, Empty, InterpretationTag } from "./ui";

/** What the Recommender decided after each round, and why. */
export function DecisionLog({ decisions }: { decisions: Decision[] }) {
  return (
    <Card title="Decisions" badge={<InterpretationTag />}>
      {decisions.length === 0 ? (
        <Empty>After each round the agent decides whether to explore more levels or conclude.</Empty>
      ) : (
        decisions.map((d) => (
          <div className="decision" key={d.round}>
            <div className="decision__head">
              <strong>After round {d.round}</strong>
              {d.action === "explore" ? (
                <Badge variant="warn">explore {d.new_levels.map(formatLevel).join(", ")}</Badge>
              ) : (
                <Badge variant="ok">conclude</Badge>
              )}
              {d.decided_by === "budget" && <Badge variant="muted">round budget</Badge>}
            </div>
            <p className={d.decided_by === "agent" ? "llm-text" : "card__hint"} style={{ margin: 0 }}>
              {d.rationale}
            </p>
          </div>
        ))
      )}
    </Card>
  );
}
