/*
 * TypeScript mirrors of the backend Pydantic models (Phase 2/4/5/6).
 * Kept deliberately close to the Python field names so the API layer needs
 * no remapping.
 */

export type TaskType = "classification" | "regression";

export interface PreprocessingConfig {
  normalize: boolean;
}

export interface ExperimentConfiguration {
  dataset_id: string;
  model_type: "mlp" | "linear_baseline";
  hyperparameters: Record<string, number>;
  preprocessing: PreprocessingConfig;
  random_seed: number;
}

export type ExperimentStatus = "success" | "failed" | "anomalous";

export interface ExperimentResult {
  experiment_id: string;
  session_id: string;
  config: ExperimentConfiguration;
  task_type: TaskType | null;
  metrics: Record<string, number> | null;
  status: ExperimentStatus;
  error: string | null;
  cycle: number | null;
  timestamp: string;
}

export type AnomalyRule =
  | "outlier_detection"
  | "loss_divergence"
  | "validation_collapse";

export interface AnomalyReport {
  anomaly_id: string;
  experiment_id: string;
  rule: AnomalyRule;
  explanation: string;
  severity: "warning" | "critical";
  detected_at: string;
}

export interface StatisticalComparison {
  comparison_id: string;
  condition_a_name: string;
  condition_b_name: string;
  metric: string;
  t_statistic: number;
  p_value: number;
  effect_size: number;
  confidence_interval: [number, number];
  sample_sizes: [number, number];
  warning: string | null;
}

export type RecommendationAction = "run_more_experiments" | "conclude";

export interface Recommendation {
  action: RecommendationAction;
  recommended_experiments: ExperimentConfiguration[];
  explanation: string;
  evidence_summary: string;
  timestamp: string;
}

/** Whether a run-cycle is actually in progress — distinct from `status`. */
export type RunPhase = "idle" | "running" | "failed";

export interface SessionSummary {
  session_id: string;
  research_question: string;
  status: string;
  run_phase: RunPhase;
  cycle_count: number;
  experiment_count: number;
  created_at: string;
}

/** Values of sessions.current_node — also the LangGraph node names. */
export type WorkflowNode =
  | "planning"
  | "executing"
  | "validating"
  | "analyzing"
  | "recommending"
  | "concluded";

export interface SessionDetail {
  session_id: string;
  dataset_id: string;
  research_question: string;
  status: string;
  current_node: WorkflowNode;
  run_phase: RunPhase;
  run_error: string | null;
  cycle_count: number;
  experiment_count: number;
  plan_explanation: string | null;
  created_at: string;
  updated_at: string;
}

export interface CreateSessionResponse {
  session_id: string;
  created_at: string;
}

export interface RunCycleResponse {
  session_id: string;
  current_node: WorkflowNode;
  status: string;
  run_phase: RunPhase;
  cycles_completed: number;
  experiments_completed: number;
  recommendation: Recommendation | null;
}

export interface SessionCycle {
  cycle_number: number;
  plan_explanation: string | null;
  experiments: ExperimentResult[];
  anomalies: AnomalyReport[];
  statistical_comparisons: StatisticalComparison[];
  recommendation: Recommendation | null;
  continued: boolean;
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

/** The single error shape every failing endpoint returns (backend/api/errors.py). */
export interface ApiError {
  error: string;
  message: string;
  details?: Record<string, unknown> | null;
}

/** Body returned by DELETE /api/sessions/{id} and DELETE /api/datasets/{id}. */
export interface DeleteResult {
  deleted: "session" | "dataset";
  id: string;
  sessions_deleted: number;
  experiments_deleted: number;
}
