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
  /** Cycle whose validation node raised this flag. */
  detected_cycle: number | null;
  /**
   * Cycle whose validation node withdrew it, or null while it still holds.
   * Detection re-runs over all evidence each cycle, so a flag raised against
   * a thin group can be cleared once the group grows.
   */
  resolved_cycle: number | null;
  detected_at: string;
}

/**
 * 'two_sample_t' — both conditions vary (the normal case).
 * 'one_sample_t' — one condition is deterministic (identical for every seed,
 * as linear_baseline is), so it is treated as a known constant to test the
 * other against rather than the comparison being abandoned.
 */
export type StatisticalTestType = "two_sample_t" | "one_sample_t";

export interface StatisticalComparison {
  comparison_id: string;
  condition_a_name: string;
  condition_b_name: string;
  metric: string;
  test_type: StatisticalTestType;
  t_statistic: number;
  p_value: number;
  effect_size: number;
  confidence_interval: [number, number];
  sample_sizes: [number, number];
  warning: string | null;
}

/** A condition pair the analysis node could not compare, and the real reason. */
export interface ComparisonSkip {
  condition_a_name: string;
  condition_b_name: string;
  metric: string;
  reason_code: "insufficient_data" | "insufficient_variance";
  reason: string;
}

/** Descriptive statistics for one condition, computed by the analysis node. */
export interface ConditionSummary {
  condition_name: string;
  metric: string;
  n_successful: number;
  n_anomalous: number;
  n_failed: number;
  mean: number;
  std: number;
  min: number;
  max: number;
  /** std === 0 across replicates: random_seed has no effect on this model. */
  deterministic: boolean;
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

/**
 * Why an investigation stopped.
 * 'agent_concluded' — the Recommender judged the evidence sufficient.
 * 'cycle_limit' — the safety cap stopped a loop that still wanted more
 * experiments. That is NOT a settled answer and must be shown differently.
 */
export type TerminationReason = "agent_concluded" | "cycle_limit";

export interface SessionSummary {
  session_id: string;
  research_question: string;
  status: string;
  run_phase: RunPhase;
  termination_reason: TerminationReason | null;
  parent_session_id: string | null;
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
  parent_session_id: string | null;
  research_question: string;
  status: string;
  current_node: WorkflowNode;
  run_phase: RunPhase;
  run_error: string | null;
  termination_reason: TerminationReason | null;
  cycle_count: number;
  experiment_count: number;
  /** The MAX_ADAPTIVE_CYCLES safety cap this session runs under. */
  max_cycles: number;
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
  termination_reason: TerminationReason | null;
  cycles_completed: number;
  experiments_completed: number;
  recommendation: Recommendation | null;
}

/**
 * One adaptive cycle. Two scopes live here and must never be mixed in the UI:
 * PER-CYCLE (`experiments`, `anomalies_detected`, `anomalies_resolved`) is
 * what this cycle did; CUMULATIVE (`statistical_comparisons`,
 * `condition_summaries`, `recommendation`, `open_anomaly_count`) is the state
 * of the whole investigation as of this cycle.
 */
export interface SessionCycle {
  cycle_number: number;
  plan_explanation: string | null;
  /** PER-CYCLE: experiments (training runs) executed in this cycle. */
  experiments: ExperimentResult[];
  /** PER-CYCLE: flags this cycle's validation node raised. */
  anomalies_detected: AnomalyReport[];
  /** PER-CYCLE: flags this cycle's validation node withdrew. */
  anomalies_resolved: AnomalyReport[];
  /** CUMULATIVE: experiments in this and all earlier cycles. */
  cumulative_experiment_count: number;
  /** CUMULATIVE: flags still open at the end of this cycle. */
  open_anomaly_count: number;
  /** CUMULATIVE: analysis over all evidence so far. */
  statistical_comparisons: StatisticalComparison[];
  /** CUMULATIVE: pairs that could not be compared, with the real reason. */
  skipped_comparisons: ComparisonSkip[];
  /** CUMULATIVE: per-condition descriptive statistics. */
  condition_summaries: ConditionSummary[];
  /** CUMULATIVE decision, stored exactly as the agent produced it. */
  recommendation: Recommendation | null;
  continued: boolean;
  termination_reason: TerminationReason | null;
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
