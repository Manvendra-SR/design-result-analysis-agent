import type {
  AnomalyReport,
  ComparisonSkip,
  ConditionSummary,
  ExperimentConfiguration,
  ExperimentResult,
  Recommendation,
  SessionCycle,
  SessionDetail,
  StatisticalComparison,
} from "../types/api";

export function makeConfig(
  over: Partial<ExperimentConfiguration> = {},
): ExperimentConfiguration {
  return {
    dataset_id: "ds-1",
    model_type: "mlp",
    hyperparameters: { dropout: 0.2, learning_rate: 0.001 },
    preprocessing: { normalize: false },
    random_seed: 42,
    ...over,
  };
}

export function makeExperiment(
  over: Partial<ExperimentResult> = {},
): ExperimentResult {
  return {
    experiment_id: Math.random().toString(36).slice(2),
    session_id: "sess-1",
    config: makeConfig(over.config),
    task_type: "classification",
    metrics: {
      train_loss: 0.12,
      val_loss: 0.18,
      accuracy: 0.93,
    },
    status: "success",
    error: null,
    cycle: 1,
    timestamp: "2026-01-01T10:00:00Z",
    ...over,
  };
}

export function makeComparison(
  over: Partial<StatisticalComparison> = {},
): StatisticalComparison {
  return {
    comparison_id: Math.random().toString(36).slice(2),
    condition_a_name: "dropout_0.0",
    condition_b_name: "dropout_0.2",
    metric: "accuracy",
    test_type: "two_sample_t",
    t_statistic: 2.4,
    p_value: 0.03,
    effect_size: 0.62,
    confidence_interval: [0.01, 0.05],
    sample_sizes: [6, 6],
    warning: null,
    ...over,
  };
}

export function makeSkip(over: Partial<ComparisonSkip> = {}): ComparisonSkip {
  return {
    condition_a_name: "mlp",
    condition_b_name: "linear_baseline",
    metric: "accuracy",
    reason_code: "insufficient_variance",
    reason: "Both conditions are deterministic, so no t-test is defined.",
    ...over,
  };
}

export function makeConditionSummary(
  over: Partial<ConditionSummary> = {},
): ConditionSummary {
  return {
    condition_name: "mlp | dropout=0.1",
    metric: "accuracy",
    n_successful: 6,
    n_anomalous: 0,
    n_failed: 0,
    mean: 0.78,
    std: 0.004,
    min: 0.77,
    max: 0.79,
    deterministic: false,
    ...over,
  };
}

export function makeAnomaly(over: Partial<AnomalyReport> = {}): AnomalyReport {
  return {
    anomaly_id: Math.random().toString(36).slice(2),
    experiment_id: "exp-1",
    rule: "outlier_detection",
    explanation: "val_loss is far from the group mean",
    severity: "warning",
    detected_cycle: 1,
    resolved_cycle: null,
    detected_at: "2026-01-01T10:00:00Z",
    ...over,
  };
}

export function makeRecommendation(
  over: Partial<Recommendation> = {},
): Recommendation {
  return {
    action: "conclude",
    recommended_experiments: [],
    explanation: "The evidence answers the question.",
    evidence_summary: "12 experiments across 2 conditions.",
    timestamp: "2026-01-01T10:30:00Z",
    ...over,
  };
}

export function makeCycle(over: Partial<SessionCycle> = {}): SessionCycle {
  return {
    cycle_number: 1,
    plan_explanation: null,
    experiments: [],
    anomalies_detected: [],
    anomalies_resolved: [],
    cumulative_experiment_count: 0,
    open_anomaly_count: 0,
    statistical_comparisons: [],
    skipped_comparisons: [],
    condition_summaries: [],
    recommendation: null,
    continued: false,
    termination_reason: null,
    ...over,
  };
}

export function makeSession(over: Partial<SessionDetail> = {}): SessionDetail {
  return {
    session_id: "sess-1",
    dataset_id: "ds-1",
    parent_session_id: null,
    research_question: "Does dropout help?",
    status: "active",
    current_node: "planning",
    run_phase: "idle",
    run_error: null,
    termination_reason: null,
    cycle_count: 0,
    experiment_count: 0,
    max_cycles: 6,
    plan_explanation: null,
    created_at: "2026-01-01T09:00:00Z",
    updated_at: "2026-01-01T09:00:00Z",
    ...over,
  };
}
