/*
 * TypeScript mirrors of the backend Pydantic models. Field names match the
 * Python ones so the API layer needs no remapping.
 */

export type TaskType = "classification" | "regression";

export interface ExperimentConfiguration {
  dataset_id: string;
  model_type: "mlp" | "linear_baseline";
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

export type Factor =
  | "model_type"
  | "normalize"
  | "hidden_size"
  | "dropout"
  | "learning_rate"
  | "batch_size"
  | "epochs";

export interface Plan {
  factor: Factor;
  levels: Level[];
  reference: Level;
  base: {
    model_type: "mlp" | "linear_baseline";
    hyperparameters: Record<string, number>;
    normalize: boolean;
  };
  rationale: string;
}

export interface Decision {
  round: number;
  action: "explore" | "conclude";
  new_levels: Level[];
  rationale: string;
  decided_by: "agent" | "budget";
}

export type Verdict = "better" | "worse" | "inconclusive";
export type Metric = "accuracy" | "mse";

export interface ConditionSummary {
  level: Level;
  label: string;
  is_reference: boolean;
  n_ok: number;
  n_failed: number;
  mean: number | null;
  seed_std: number | null;
}

/** One level vs the reference: metric difference with a 95% bootstrap CI. */
export interface Comparison {
  level: Level;
  label: string;
  diff: number;
  ci_low: number;
  ci_high: number;
  verdict: Verdict;
}

export interface Analysis {
  split: "val" | "test";
  metric: Metric;
  higher_is_better: boolean;
  n_rows: number;
  majority_rate: number | null;
  conditions: ConditionSummary[];
  comparisons: Comparison[];
}

export interface Report {
  challenger: string;
  reference: string;
  metric: Metric;
  higher_is_better: boolean;
  challenger_score: number;
  reference_score: number;
  comparison: Comparison;
  majority_rate: number | null;
  n_test_rows: number;
  rounds_run: number;
  stopped_by: "agent" | "budget";
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
