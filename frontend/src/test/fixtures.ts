import type {
  ExperimentConfiguration,
  ExperimentResult,
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
    t_statistic: 2.4,
    p_value: 0.03,
    effect_size: 0.62,
    confidence_interval: [0.01, 0.05],
    sample_sizes: [6, 6],
    warning: null,
    ...over,
  };
}

export function makeSession(over: Partial<SessionDetail> = {}): SessionDetail {
  return {
    session_id: "sess-1",
    dataset_id: "ds-1",
    research_question: "Does dropout help?",
    status: "active",
    current_node: "planning",
    run_phase: "idle",
    run_error: null,
    cycle_count: 0,
    experiment_count: 0,
    plan_explanation: null,
    created_at: "2026-01-01T09:00:00Z",
    updated_at: "2026-01-01T09:00:00Z",
    ...over,
  };
}
