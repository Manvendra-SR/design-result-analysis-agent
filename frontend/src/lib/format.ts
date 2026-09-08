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

/** Compact, stable description of an experiment configuration. */
export function describeConfig(cfg: ExperimentConfiguration): string {
  if (cfg.model_type === "linear_baseline") {
    return `linear_baseline · seed=${cfg.random_seed}${
      cfg.preprocessing.normalize ? " · norm" : ""
    }`;
  }
  const hp = cfg.hyperparameters ?? {};
  const parts = Object.keys(hp)
    .sort()
    .map((k) => `${k}=${hp[k]}`);
  parts.push(`seed=${cfg.random_seed}`);
  if (cfg.preprocessing.normalize) parts.push("norm");
  return `mlp · ${parts.join(" · ")}`;
}

export function comparisonLabel(c: StatisticalComparison): string {
  return `${c.condition_a_name} vs ${c.condition_b_name}`;
}

export function formatTimestamp(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString();
}
