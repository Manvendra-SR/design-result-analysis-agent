import { Fragment, useState } from "react";

import { describeConfig, formatTimestamp, num, pct } from "../lib/format";
import type { ExperimentResult } from "../types/api";
import { Badge, Card, Empty } from "./ui";

function StatusBadge({ status }: { status: ExperimentResult["status"] }) {
  return <span className={`status-pill status-pill--${status}`}>{status}</span>;
}

function metric(exp: ExperimentResult, key: string): string {
  const v = exp.metrics?.[key];
  if (v === undefined) return "—";
  return key === "accuracy" ? pct(v) : num(v);
}

/**
 * ExperimentTable (task 7.7). Newest first, colour-coded status, expandable
 * rows for the full config / error. Experiments are grouped by the adaptive
 * cycle that produced them (experiments.cycle) so the accumulation across
 * cycles is visible.
 */
export function ExperimentTable({
  experiments,
}: {
  experiments: ExperimentResult[];
}) {
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
    (a, b) =>
      (b.cycle ?? 0) - (a.cycle ?? 0) ||
      new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime(),
  );

  const counts = {
    success: experiments.filter((e) => e.status === "success").length,
    anomalous: experiments.filter((e) => e.status === "anomalous").length,
    failed: experiments.filter((e) => e.status === "failed").length,
  };

  return (
    <Card
      title="Experiments"
      badge={<Badge variant="gradient">Computed</Badge>}
      right={
        <span style={{ display: "flex", gap: 8 }}>
          <Badge variant="ok">{counts.success} ok</Badge>
          <Badge variant="warn">{counts.anomalous} anomalous</Badge>
          <Badge variant="danger">{counts.failed} failed</Badge>
        </span>
      }
    >
      {sorted.length === 0 ? (
        <Empty>No experiments yet. Run the investigation to generate them.</Empty>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Cycle</th>
                <th>Configuration</th>
                <th className="num">train_loss</th>
                <th className="num">val_loss</th>
                <th className="num">accuracy</th>
                <th>Status</th>
                <th>When</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((exp) => {
                const isOpen = open.has(exp.experiment_id);
                return (
                  <Fragment key={exp.experiment_id}>
                    <tr
                      className="row-clickable"
                      onClick={() => toggle(exp.experiment_id)}
                    >
                      <td className="num">{exp.cycle ?? "—"}</td>
                      <td className="mono">{describeConfig(exp.config)}</td>
                      <td className="num">{metric(exp, "train_loss")}</td>
                      <td className="num">{metric(exp, "val_loss")}</td>
                      <td className="num">{metric(exp, "accuracy")}</td>
                      <td>
                        <StatusBadge status={exp.status} />
                      </td>
                      <td>{formatTimestamp(exp.timestamp)}</td>
                    </tr>
                    {isOpen && (
                      <tr>
                        <td className="detail-cell" colSpan={7}>
                          <pre>
                            {JSON.stringify(
                              {
                                experiment_id: exp.experiment_id,
                                task_type: exp.task_type,
                                config: exp.config,
                                metrics: exp.metrics,
                                error: exp.error,
                              },
                              null,
                              2,
                            )}
                          </pre>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
