import type {
  Analysis,
  Comparison,
  Decision,
  ExperimentConfiguration,
  ExperimentResult,
  Plan,
  Report,
  SessionDetail,
} from "../types/api";

export function makeConfig(over: Partial<ExperimentConfiguration> = {}): ExperimentConfiguration {
  return {
    dataset_id: "ds-1",
    model_type: "mlp",
    hyperparameters: { dropout: 0.2, epochs: 10 },
    preprocessing: { normalize: false },
    random_seed: 0,
    ...over,
  };
}

export function makeExperiment(over: Partial<ExperimentResult> = {}): ExperimentResult {
  return {
    experiment_id: Math.random().toString(36).slice(2),
    session_id: "sess-1",
    round: 1,
    config: makeConfig(over.config),
    status: "ok",
    error: null,
    metrics: { accuracy: 0.81, train_loss: 0.42, training_time_seconds: 3.2 },
    created_at: "2026-01-01T10:00:00+00:00",
    ...over,
  };
}

export function makePlan(over: Partial<Plan> = {}): Plan {
  return {
    factor: "dropout",
    levels: [0, 0.2],
    reference: 0,
    base: { model_type: "mlp", hyperparameters: { epochs: 10 }, normalize: false },
    rationale: "Vary dropout and hold everything else fixed.",
    ...over,
  };
}

export function makeComparison(over: Partial<Comparison> = {}): Comparison {
  return {
    level: 0.2,
    label: "dropout=0.2",
    diff: 0.021,
    ci_low: 0.008,
    ci_high: 0.034,
    verdict: "better",
    ...over,
  };
}

export function makeAnalysis(over: Partial<Analysis> = {}): Analysis {
  return {
    split: "val",
    metric: "accuracy",
    higher_is_better: true,
    n_rows: 1575,
    majority_rate: 0.6,
    conditions: [
      { level: 0, label: "dropout=0", is_reference: true, n_ok: 3, n_failed: 0, mean: 0.79, seed_std: 0.002 },
      { level: 0.2, label: "dropout=0.2", is_reference: false, n_ok: 2, n_failed: 1, mean: 0.811, seed_std: 0.003 },
    ],
    comparisons: [makeComparison()],
    ...over,
  };
}

export function makeDecision(over: Partial<Decision> = {}): Decision {
  return {
    round: 1,
    action: "conclude",
    new_levels: [],
    rationale: "dropout=0.2 is clearly better; nothing untried would change that.",
    decided_by: "agent",
    ...over,
  };
}

export function makeReport(over: Partial<Report> = {}): Report {
  return {
    challenger: "dropout=0.2",
    reference: "dropout=0",
    metric: "accuracy",
    higher_is_better: true,
    challenger_score: 0.812,
    reference_score: 0.793,
    comparison: makeComparison({ diff: 0.019, ci_low: 0.005, ci_high: 0.033 }),
    majority_rate: 0.6,
    n_test_rows: 1575,
    rounds_run: 1,
    stopped_by: "agent",
    headline: "On the held-out test set (1575 rows), dropout=0.2 performs better than dropout=0.",
    interpretation: "Dropout helps a little on this dataset.",
    ...over,
  };
}

export function makeSession(over: Partial<SessionDetail> = {}): SessionDetail {
  return {
    session_id: "sess-1",
    dataset_id: "ds-1",
    research_question: "Does dropout help?",
    status: "pending",
    error: null,
    plan: null,
    decisions: [],
    report: null,
    created_at: "2026-01-01T10:00:00+00:00",
    experiments: [],
    analysis: null,
    max_rounds: 3,
    n_seeds: 3,
    ...over,
  };
}
