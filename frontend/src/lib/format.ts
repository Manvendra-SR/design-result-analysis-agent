/*
 * Presentation helpers. These format and label values the backend already
 * computed — they never derive a new statistic.
 */
import type {
  ExperimentConfiguration,
  StatisticalComparison,
  TaskType,
  WorkflowNode,
} from "../types/api";

export const WORKFLOW_STAGES: WorkflowNode[] = [
  "planning",
  "executing",
  "validating",
  "analyzing",
  "recommending",
];

/** One-line role blurb for each stage, for the loop visualiser. */
export const STAGE_ROLE: Record<WorkflowNode, string> = {
  planning: "Planner agent designs experiments",
  executing: "Runner trains the models",
  validating: "Anomaly detector checks results",
  analyzing: "Statistical analyzer runs t-tests",
  recommending: "Recommender agent decides next step",
  concluded: "Investigation complete",
};

export function num(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

export function pct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function isSignificant(p: number): boolean {
  return p < 0.05;
}

/** Cohen's-d magnitude labels (matches the note in models/statistics.py). */
export function effectSizeLabel(d: number): string {
  const a = Math.abs(d);
  if (a >= 0.8) return "large";
  if (a >= 0.5) return "medium";
  if (a >= 0.2) return "small";
  return "negligible";
}

/** The metric the analysis node compares for a task type (task 3.11). */
export function primaryMetric(taskType: TaskType | null | undefined): string {
  return taskType === "regression" ? "val_loss" : "accuracy";
}

/**
 * Compact, stable description of an experiment configuration.
 *
 * `epochs` is rendered as a quantity ("20 epochs") rather than `epochs=20`.
 * It is configuration for ONE experiment — how many passes over the training
 * data that single training run performs — and reading it as a bare number
 * next to counts like "6 experiments" invites exactly the wrong reading.
 */
export function describeConfig(cfg: ExperimentConfiguration): string {
  if (cfg.model_type === "linear_baseline") {
    return `linear_baseline · seed=${cfg.random_seed}${
      cfg.preprocessing.normalize ? " · norm" : ""
    }`;
  }
  const hp = cfg.hyperparameters ?? {};
  const parts = Object.keys(hp)
    .sort()
    .filter((k) => k !== "epochs")
    .map((k) => `${k}=${hp[k]}`);
  if (hp.epochs !== undefined) parts.push(`${hp.epochs} epochs`);
  parts.push(`seed=${cfg.random_seed}`);
  if (cfg.preprocessing.normalize) parts.push("norm");
  return `mlp · ${parts.join(" · ")}`;
}

export function comparisonLabel(c: StatisticalComparison): string {
  return `${c.condition_a_name} vs ${c.condition_b_name}`;
}

/** Plain-language note on which test produced a comparison. */
export function testTypeLabel(c: StatisticalComparison): string {
  return c.test_type === "one_sample_t"
    ? "one-sample t-test (one condition is deterministic)"
    : "two-sample t-test";
}

/**
 * Render a backend timestamp in the viewer's own time zone (IST on an Indian
 * machine) — never a hardcoded offset.
 *
 * The API sends explicitly-UTC ISO strings (`...+00:00`, see
 * backend/models/timestamps.py), which `new Date()` converts correctly. The
 * `Z` fallback below covers an offset-less string: JavaScript parses an ISO
 * date-time with no offset as LOCAL time, which silently displayed UTC digits
 * as if they were local ones — the bug this pair of changes fixes. Every
 * timestamp this project stores is UTC, so assuming UTC is the safe reading.
 */
export function formatTimestamp(iso: string): string {
  const hasZone = /(?:Z|[+-]\d{2}:?\d{2})$/.test(iso);
  const d = new Date(hasZone ? iso : `${iso}Z`);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString();
}
