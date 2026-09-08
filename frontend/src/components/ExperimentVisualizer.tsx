import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { num, pct, primaryMetric } from "../lib/format";
import type { ExperimentResult } from "../types/api";
import { Badge, Card, Empty } from "./ui";

const CANDIDATE_KEYS = [
  "dropout",
  "learning_rate",
  "hidden_size",
  "batch_size",
  "epochs",
] as const;

type VaryingField = {
  key: string;
  label: string;
  valueOf: (e: ExperimentResult) => string | number | undefined;
};

function pickVaryingField(exps: ExperimentResult[]): VaryingField | null {
  const fields: VaryingField[] = [
    ...CANDIDATE_KEYS.map((k) => ({
      key: k,
      label: k,
      valueOf: (e: ExperimentResult) => e.config.hyperparameters?.[k],
    })),
    {
      key: "normalize",
      label: "normalize",
      valueOf: (e: ExperimentResult) => String(e.config.preprocessing.normalize),
    },
    {
      key: "model_type",
      label: "model_type",
      valueOf: (e: ExperimentResult) => e.config.model_type,
    },
  ];

  let best: VaryingField | null = null;
  let bestDistinct = 1;
  for (const f of fields) {
    const values = new Set(
      exps
        .map((e) => f.valueOf(e))
        .filter((v) => v !== undefined && v !== null),
    );
    if (values.size > bestDistinct) {
      best = f;
      bestDistinct = values.size;
    }
  }
  return best;
}

/**
 * ExperimentVisualizer (task 7.10, adapted).
 *
 * The backend records only final metrics per experiment — no per-epoch loss
 * series — so this plots the final metric (accuracy for classification,
 * val_loss for regression) against whichever single hyperparameter varies
 * most across the successful runs. That is what the research questions
 * ("does dropout help?") actually ask, and it invents no data the backend
 * did not provide.
 */
export function ExperimentVisualizer({
  experiments,
}: {
  experiments: ExperimentResult[];
}) {
  const successful = useMemo(
    () => experiments.filter((e) => e.status === "success" && e.metrics),
    [experiments],
  );

  const taskType = successful[0]?.task_type ?? null;
  const metricKey = primaryMetric(taskType);
  const field = useMemo(() => pickVaryingField(successful), [successful]);

  const data = useMemo(() => {
    if (!field) return [];
    const groups = new Map<string, number[]>();
    for (const e of successful) {
      const v = field.valueOf(e);
      const m = e.metrics?.[metricKey];
      if (v === undefined || m === undefined) continue;
      const label = String(v);
      if (!groups.has(label)) groups.set(label, []);
      groups.get(label)!.push(m);
    }
    return [...groups.entries()]
      .map(([label, values]) => ({
        label,
        mean: values.reduce((a, b) => a + b, 0) / values.length,
        n: values.length,
      }))
      .sort((a, b) => {
        const na = Number(a.label);
        const nb = Number(b.label);
        if (!Number.isNaN(na) && !Number.isNaN(nb)) return na - nb;
        return a.label.localeCompare(b.label);
      });
  }, [field, successful, metricKey]);

  return (
    <Card
      title="Metric Comparison"
      badge={<Badge variant="gradient">Computed</Badge>}
      hint={
        field
          ? `Mean ${metricKey} by ${field.label} across successful runs (the backend stores final metrics only, not loss curves).`
          : undefined
      }
    >
      {successful.length === 0 ? (
        <Empty>No successful experiments to chart yet.</Empty>
      ) : !field || data.length < 2 ? (
        <Empty>
          Not enough variation to chart — every successful run shares the same
          hyperparameters so far.
        </Empty>
      ) : (
        <>
          <div style={{ width: "100%", height: 300 }}>
            <ResponsiveContainer>
              <BarChart
                data={data}
                margin={{ top: 8, right: 16, bottom: 24, left: 8 }}
              >
                <CartesianGrid stroke="#e6e9f0" strokeDasharray="3 3" />
                <XAxis
                  dataKey="label"
                  stroke="#94a3b8"
                  tick={{ fontSize: 12, fill: "#5b6b82" }}
                  label={{
                    value: field.label,
                    position: "insideBottom",
                    offset: -12,
                    fill: "#94a3b8",
                    fontSize: 12,
                  }}
                />
                <YAxis
                  stroke="#94a3b8"
                  tick={{ fontSize: 12, fill: "#5b6b82" }}
                  domain={
                    metricKey === "accuracy" ? [0, 1] : ["auto", "auto"]
                  }
                  tickFormatter={(v: number) =>
                    metricKey === "accuracy" ? pct(v, 0) : num(v, 2)
                  }
                />
                <Tooltip
                  cursor={{ fill: "rgba(37,99,235,0.06)" }}
                  contentStyle={{
                    background: "#ffffff",
                    border: "1px solid #d3d9e4",
                    borderRadius: 8,
                    fontSize: 12,
                    boxShadow: "0 4px 12px rgba(16,24,40,0.08)",
                  }}
                  formatter={(value) => {
                    const v = Number(value);
                    return [
                      metricKey === "accuracy" ? pct(v) : num(v),
                      `mean ${metricKey}`,
                    ];
                  }}
                />
                <Bar dataKey="mean" radius={[5, 5, 0, 0]}>
                  {data.map((_, i) => (
                    <Cell key={i} fill={i % 2 === 0 ? "#2563eb" : "#06b6d4"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="table-wrap" style={{ marginTop: 14 }}>
            <table>
              <thead>
                <tr>
                  <th>{field.label}</th>
                  <th className="num">mean {metricKey}</th>
                  <th className="num">runs</th>
                </tr>
              </thead>
              <tbody>
                {data.map((d) => (
                  <tr key={d.label}>
                    <td className="mono">{d.label}</td>
                    <td className="num">
                      {metricKey === "accuracy" ? pct(d.mean) : num(d.mean)}
                    </td>
                    <td className="num">{d.n}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Card>
  );
}
