/*
 * TypeScript mirrors of the backend Pydantic models. Field names match the
 * Python ones so the API layer needs no remapping.
 */

export type TaskType = "classification" | "regression";

export type ModelType = "linear_baseline" | "decision_tree" | "random_forest" | "mlp";

export interface ExperimentConfiguration {
  dataset_id: string;
  model_type: ModelType;
  hyperparameters: Record<string, number>;
  preprocessing: { normalize: boolean };
  random_seed: number;
}

/** One training run. `failed` = crashed or produced NaN/Inf. */
export interface ExperimentResult {
  experiment_id: string;
  session_id: string;
  round: number;
  config: ExperimentConfiguration;
  status: "ok" | "failed";
  error: string | null;
  metrics: Record<string, number> | null;
  created_at: string;
}

/** A value of the factor under test. */
export type Level = boolean | number | string;

export type PlanMode = "effect" | "selection";

/** One configuration (minus the seed) and what it was derived from. */
export interface Candidate {
  id: string;
  label: string;
  model_type: ModelType;
  hyperparameters: Record<string, number>;
  normalize: boolean;
  parent: string | null;
  /** What differs from the parent (effect mode: the factor). */
  knob: string | null;
  value: Level | null;
  round: number;
}

export interface Plan {
  mode: PlanMode;
  candidates: Candidate[];
  /** Id of the reference candidate. */
  reference: string;
  /** Effect mode: the one factor varied. */
  factor: string | null;
  base: {
    model_type: ModelType;
    hyperparameters: Record<string, number>;
    normalize: boolean | null;
  } | null;
  rationale: string;
}

export interface Decision {
  round: number;
  action: "refine" | "conclude";
  parent: string | null;
  knob: string | null;
  values: Level[];
  new_candidates: string[];
  rationale: string;
  decided_by: "agent" | "budget" | "settled";
}

export type Verdict = "better" | "worse" | "inconclusive";
export type Metric = "accuracy" | "mse";

export interface ConditionSummary {
  id: string;
  label: string;
  family: ModelType;
  parent: string | null;
  change: string | null;
  is_reference: boolean;
  n_ok: number;
  n_failed: number;
  mean: number | null;
  seed_std: number | null;
  train_metric: number | null;
  /** How much better it scores on train than on validation (positive = overfitting). */
  gap: number | null;
  contender: boolean;
}

/** Candidate `a` vs candidate `b`: metric difference with a bootstrap CI. */
export interface Comparison {
  a: string;
  label: string;
  b: string;
  against: string;
  anchor: "reference" | "leader" | "parent" | "runner_up";
  diff: number;
  ci_low: number;
  ci_high: number;
  confidence: number;
  verdict: Verdict;
}

export interface Analysis {
  split: "val" | "test";
  metric: Metric;
  higher_is_better: boolean;
  mode: PlanMode;
  n_rows: number;
  majority_rate: number | null;
  leader: string | null;
  contenders: string[];
  conditions: ConditionSummary[];
  comparisons: Comparison[];
}

export interface ScoreCI {
  value: number;
  ci_low: number;
  ci_high: number;
}

export interface Report {
  mode: PlanMode;
  winner: string;
  winner_id: string;
  reference: string;
  runner_up: string | null;
  metric: Metric;
  higher_is_better: boolean;
  winner_score: ScoreCI;
  reference_score: ScoreCI;
  primary: Comparison;
  secondary: Comparison | null;
  /** CI level of each test comparison (Bonferroni over the comparisons). */
  confidence: number;
  winner_val_score: number;
  val_to_test_drop: number;
  effort: Record<string, number>;
  candidates_tried: number;
  majority_rate: number | null;
  n_test_rows: number;
  rounds_run: number;
  stopped_by: "agent" | "budget" | "settled";
  headline: string;
  interpretation: string | null;
}

export type SessionStatus = "pending" | "running" | "done" | "failed";

export interface Session {
  session_id: string;
  dataset_id: string;
  research_question: string;
  status: SessionStatus;
  error: string | null;
  plan: Plan | null;
  decisions: Decision[];
  report: Report | null;
  created_at: string;
}

export interface SessionDetail extends Session {
  experiments: ExperimentResult[];
  /** Validation-split statistics, recomputed on every request. */
  analysis: Analysis | null;
  max_rounds: number;
  n_seeds: number;
}

export interface SessionSummary {
  session_id: string;
  dataset_id: string;
  research_question: string;
  status: SessionStatus;
  experiment_count: number;
  created_at: string;
}

export interface DatasetProfile {
  dataset_id: string;
  original_filename: string;
  storage_path: string;
  target_column: string;
  feature_columns: string[];
  numeric_columns: string[];
  categorical_columns: string[];
  task_type: TaskType;
  task_type_source: "inferred" | "user_specified";
  n_rows: number;
  n_features: number;
  n_classes: number | null;
  class_labels: string[] | null;
  class_distribution: Record<string, number> | null;
  missing_value_counts: Record<string, number>;
  split_seed: number;
  created_at: string;
}

export interface DatasetIngestRequest {
  filename: string;
  csv_content: string;
  target_column: string;
  task_type_override?: "classification" | "regression" | null;
}

/** The single error shape every failing endpoint returns. */
export interface ApiError {
  error: string;
  message: string;
  details?: Record<string, unknown> | null;
}

export interface DeleteResult {
  deleted: "session" | "dataset";
  id: string;
  sessions_deleted: number;
  experiments_deleted: number;
}
