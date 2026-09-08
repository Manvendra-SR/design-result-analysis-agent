import { useState } from "react";

import {
  comparisonLabel,
  effectSizeLabel,
  isSignificant,
  num,
} from "../lib/format";
import type { SessionCycle } from "../types/api";
import { Badge, Card, Empty } from "./ui";

/**
 * CycleHistory (Requirement 11.9 — "visibly show the progression through the
 * adaptive loop at each stage"). Backed by GET /sessions/{id}/cycles, this is
 * how a human sees what each autonomous cycle ran and why it continued or
 * stopped, so the multi-cycle run is not a black box.
 */
export function CycleHistory({ cycles }: { cycles: SessionCycle[] }) {
  const [open, setOpen] = useState<number | null>(
    cycles.length ? cycles[cycles.length - 1].cycle_number : null,
  );

  return (
    <Card
      title="Cycle History"
      badge={<Badge variant="gradient">Per cycle</Badge>}
      hint="Each adaptive cycle: what ran, what was flagged, what the stats showed, and the agent's decision to continue or stop."
    >
      {cycles.length === 0 ? (
        <Empty>No completed cycles yet.</Empty>
      ) : (
        cycles.map((c) => {
          const isOpen = open === c.cycle_number;
          const anomalies = c.anomalies.length;
          return (
            <div className="cycle" key={c.cycle_number}>
              <button
                type="button"
                className="cycle__head"
                aria-expanded={isOpen}
                onClick={() => setOpen(isOpen ? null : c.cycle_number)}
              >
                <strong>Cycle {c.cycle_number}</strong>
                <span className="mono" style={{ color: "var(--text-dim)" }}>
                  {c.experiments.length} experiments · {anomalies} anomalies
                </span>
                {c.recommendation && (
                  <span style={{ marginLeft: "auto" }}>
                    <Badge variant={c.continued ? "warn" : "ok"}>
                      {c.continued ? "continued" : "stopped"}
                    </Badge>
                  </span>
                )}
              </button>

              {isOpen && (
                <div className="cycle__body">
                  {c.plan_explanation && (
                    <div>
                      <h4 className="card__hint" style={{ margin: 0 }}>
                        Initial plan rationale
                      </h4>
                      <p className="llm-text">{c.plan_explanation}</p>
                    </div>
                  )}

                  {c.anomalies.length > 0 && (
                    <div>
                      <h4 className="card__hint" style={{ margin: 0 }}>
                        Anomalies
                      </h4>
                      <ul className="list-plain">
                        {c.anomalies.map((a) => (
                          <li key={a.anomaly_id}>
                            <Badge
                              variant={
                                a.severity === "critical" ? "danger" : "warn"
                              }
                            >
                              {a.rule}
                            </Badge>{" "}
                            {a.explanation}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  <div>
                    <h4 className="card__hint" style={{ margin: 0 }}>
                      Statistical comparisons
                    </h4>
                    {c.statistical_comparisons.length === 0 ? (
                      <Empty>None computed this cycle.</Empty>
                    ) : (
                      <ul className="list-plain mono">
                        {c.statistical_comparisons.map((s) => (
                          <li key={s.comparison_id}>
                            {comparisonLabel(s)} — p={num(s.p_value, 4)}, d=
                            {num(s.effect_size, 2)} ({effectSizeLabel(
                              s.effect_size,
                            )}
                            ){isSignificant(s.p_value) ? " ✓ significant" : ""}
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>

                  {c.recommendation && (
                    <div>
                      <h4 className="card__hint" style={{ margin: 0 }}>
                        Recommendation (LLM Interpretation)
                      </h4>
                      <p className="llm-text">
                        {c.recommendation.explanation}
                      </p>
                      <p className="llm-text" style={{ color: "var(--text-dim)" }}>
                        {c.recommendation.evidence_summary}
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })
      )}
    </Card>
  );
}
