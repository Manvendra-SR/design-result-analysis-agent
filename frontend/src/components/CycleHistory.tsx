import { useState } from "react";

import {
  comparisonLabel,
  effectSizeLabel,
  isSignificant,
  num,
} from "../lib/format";
import type { AnomalyReport, SessionCycle } from "../types/api";
import { Badge, Card, Empty } from "./ui";

/**
 * CycleHistory (Requirement 11.9 — "visibly show the progression through the
 * adaptive loop at each stage"). Backed by GET /sessions/{id}/cycles.
 *
 * Two scopes live in one cycle record and are kept visibly apart. Mixing them
 * is what previously made the output look corrupt: a header reading
 * "Cycle 3 · 3 experiments · 0 anomalies" sat directly above a recommendation
 * saying "two MLP runs were flagged as anomalous". Both were true — the counts
 * were this cycle's, the prose was about the whole investigation — but nothing
 * said so. Cumulative sections are now labelled as such.
 */
function AnomalyList({
  anomalies,
  tone,
}: {
  anomalies: AnomalyReport[];
  tone: "raised" | "withdrawn";
}) {
  return (
    <ul className="list-plain">
      {anomalies.map((a) => (
        <li key={a.anomaly_id}>
          <Badge
            variant={
              tone === "withdrawn"
                ? "ok"
                : a.severity === "critical"
                  ? "danger"
                  : "warn"
            }
          >
            {a.rule}
          </Badge>{" "}
          {a.explanation}
          {tone === "withdrawn" && a.detected_cycle != null && (
            <span style={{ color: "var(--text-dim)" }}>
              {" "}
              (raised in cycle {a.detected_cycle}; the extra replicates showed
              the value was ordinary after all)
            </span>
          )}
        </li>
      ))}
    </ul>
  );
}

export function CycleHistory({ cycles }: { cycles: SessionCycle[] }) {
  const [open, setOpen] = useState<number | null>(
    cycles.length ? cycles[cycles.length - 1].cycle_number : null,
  );

  return (
    <Card
      title="Cycle History"
      badge={<Badge variant="gradient">Per cycle</Badge>}
      hint="Each adaptive cycle: what ran, what was flagged or cleared, and the agent's decision. The agent reasons over all evidence gathered so far, so its sections are marked cumulative."
    >
      {cycles.length === 0 ? (
        <Empty>No completed cycles yet.</Empty>
      ) : (
        cycles.map((c) => {
          const isOpen = open === c.cycle_number;
          const raised = c.anomalies_detected.length;
          const withdrawn = c.anomalies_resolved.length;
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
                  {c.experiments.length} experiments
                  {c.cumulative_experiment_count > c.experiments.length
                    ? ` (${c.cumulative_experiment_count} total)`
                    : ""}
                  {raised > 0 ? ` · ${raised} flagged` : ""}
                  {withdrawn > 0 ? ` · ${withdrawn} cleared` : ""}
                  {` · ${c.open_anomaly_count} open`}
                </span>
                <span style={{ marginLeft: "auto" }}>
                  {c.termination_reason === "cycle_limit" ? (
                    <Badge variant="danger">stopped at limit</Badge>
                  ) : c.termination_reason === "agent_concluded" ? (
                    <Badge variant="ok">concluded</Badge>
                  ) : c.continued ? (
                    <Badge variant="warn">continued</Badge>
                  ) : c.recommendation ? (
                    <Badge variant="muted">stopped</Badge>
                  ) : null}
                </span>
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

                  <div>
                    <h4 className="card__hint" style={{ margin: 0 }}>
                      Experiments in this cycle
                    </h4>
                    <p style={{ margin: "4px 0 0" }}>
                      {c.experiments.length} experiment
                      {c.experiments.length === 1 ? "" : "s"} — each one a
                      complete training run.
                    </p>
                  </div>

                  {raised > 0 && (
                    <div>
                      <h4 className="card__hint" style={{ margin: 0 }}>
                        Anomalies flagged in this cycle
                      </h4>
                      <AnomalyList
                        anomalies={c.anomalies_detected}
                        tone="raised"
                      />
                    </div>
                  )}

                  {withdrawn > 0 && (
                    <div>
                      <h4 className="card__hint" style={{ margin: 0 }}>
                        Anomalies withdrawn in this cycle
                      </h4>
                      <AnomalyList
                        anomalies={c.anomalies_resolved}
                        tone="withdrawn"
                      />
                    </div>
                  )}

                  <div>
                    <h4 className="card__hint" style={{ margin: 0 }}>
                      Statistical comparisons{" "}
                      <em>(cumulative — all evidence through this cycle)</em>
                    </h4>
                    {c.statistical_comparisons.length === 0 ? (
                      c.skipped_comparisons.length > 0 ? (
                        <ul className="list-plain">
                          {c.skipped_comparisons.map((s, i) => (
                            <li key={i}>
                              <Badge variant="warn">
                                {s.reason_code.replace("_", " ")}
                              </Badge>{" "}
                              <span className="mono">
                                {s.condition_a_name} vs {s.condition_b_name}
                              </span>{" "}
                              — {s.reason}
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <Empty>
                          Only one condition exists so far, so there is nothing
                          to compare it against yet.
                        </Empty>
                      )
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
                        Recommendation (LLM Interpretation){" "}
                        <em>(cumulative — reasons over all evidence so far)</em>
                      </h4>
                      <p className="llm-text">
                        {c.recommendation.explanation}
                      </p>
                      <p className="llm-text" style={{ color: "var(--text-dim)" }}>
                        {c.recommendation.evidence_summary}
                      </p>
                      {c.termination_reason === "cycle_limit" && (
                        <div className="error-box" role="status">
                          The agent asked for more experiments here, but the
                          investigation had reached its {c.cycle_number}-cycle
                          safety limit and stopped. This is where it stopped —
                          not a conclusion it reached.
                        </div>
                      )}
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
