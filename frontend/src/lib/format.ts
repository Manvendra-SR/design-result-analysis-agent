/*
 * Presentation helpers. They format values the backend computed - they never
 * derive a new statistic.
 */
import type { ExperimentConfiguration, Level, Metric } from "../types/api";

export function num(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

export function pct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

/** A metric value: accuracy as a percentage, MSE as a number. */
export function metricValue(metric: Metric, value: number | null | undefined): string {
  return metric === "accuracy" ? pct(value) : num(value, 4);
}

/** A metric difference: accuracy in percentage points, MSE signed. */
export function metricDelta(metric: Metric, value: number): string {
  const sign = value > 0 ? "+" : "";
  return metric === "accuracy"
    ? `${sign}${(value * 100).toFixed(1)} pts`
    : `${sign}${value.toPrecision(3)}`;
}

export function formatLevel(level: Level): string {
  return String(level);
}

/** Compact description of one run's configuration. */
export function describeConfig(cfg: ExperimentConfiguration): string {
  const norm = cfg.preprocessing.normalize ? " · normalized" : "";
  if (cfg.model_type === "linear_baseline") return `linear_baseline${norm}`;
  const hp = cfg.hyperparameters ?? {};
  const parts = Object.keys(hp)
    .sort()
    .map((k) => `${k}=${hp[k]}`);
  return `mlp · ${parts.join(" · ")}${norm}`;
}

/**
 * Render a backend timestamp in the viewer's own time zone. The API sends
 * explicit-UTC ISO strings; an offset-less string is treated as UTC too.
 */
export function formatTimestamp(iso: string): string {
  const hasZone = /(?:Z|[+-]\d{2}:?\d{2})$/.test(iso);
  const d = new Date(hasZone ? iso : `${iso}Z`);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString();
}
