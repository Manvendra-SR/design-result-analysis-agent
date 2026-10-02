import { Fragment, useState } from "react";

import { describeConfig, formatTimestamp, num, pct } from "../lib/format";
import type { ExperimentResult } from "../types/api";
import { Badge, Card, Empty } from "./ui";

function metricCell(exp: ExperimentResult): string {
  if (!exp.metrics) return "—";
  if (exp.metrics.accuracy !== undefined) return pct(exp.metrics.accuracy);
  return num(exp.metrics.mse, 4);
}

/**
 * Every training run, newest first. A row expands to the raw config / error.
 * A `failed` run crashed or produced NaN/Inf; every other run is evidence.
 */
export function ExperimentTable({ experiments }: { experiments: ExperimentResult[] }) {
  const [open, setOpen] = useState<Set<string>>(new Set());

  function toggle(id: string) {
    setOpen((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  const sorted = [...experiments].sort(
    (a, b) => b.round - a.round || b.created_at.localeCompare(a.created_at),
  );
  const failed = experiments.filter((e) => e.status === "failed").length;
  const metricName = experiments.find((e) => e.metrics)?.metrics?.mse !== undefined ? "val mse" : "val accuracy";

  return (
    <Card
      title="Training runs"
      badge={<Badge variant="gradient">Computed</Badge>}
      hint="One row = one complete training run. Runs of the same condition differ only in their random seed."
      right={
        <span style={{ display: "flex", gap: 8 }}>
          <Badge variant="ok">{experiments.length - failed} ok</Badge>
          {failed > 0 && <Badge variant="danger">{failed} failed</Badge>}
        </span>
      }
    >
      {sorted.length === 0 ? (
        <Empty>No runs yet.</Empty>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th className="num">Round</th>
                <th>Configuration</th>
                <th className="num">Seed</th>
                <th className="num">{metricName}</th>
                <th className="num">train loss</th>
                <th className="num">time (s)</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((exp) => (
                <Fragment key={exp.experiment_id}>
                  <tr className="row-clickable" onClick={() => toggle(exp.experiment_id)}>
                    <td className="num">{exp.round}</td>
                    <td className="mono">{describeConfig(exp.config)}</td>
                    <td className="num">{exp.config.random_seed}</td>
                    <td className="num">{metricCell(exp)}</td>
                    <td className="num">{num(exp.metrics?.train_loss)}</td>
                    <td className="num">{num(exp.metrics?.training_time_seconds, 1)}</td>
                    <td>
                      <span className={`status-pill status-pill--${exp.status}`}>{exp.status}</span>
                    </td>
                  </tr>
                  {open.has(exp.experiment_id) && (
                    <tr>
                      <td className="detail-cell" colSpan={7}>
                        <pre>
                          {JSON.stringify(
                            {
                              config: exp.config,
                              metrics: exp.metrics,
                              error: exp.error,
                              finished: formatTimestamp(exp.created_at),
                            },
                            null,
                            2,
                          )}
                        </pre>
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
