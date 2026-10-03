import type {
  Analysis,
  Comparison,
  ConditionSummary,
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
    metrics: { accuracy: 0.81, train_accuracy: 0.83, train_loss: 0.42, training_time_seconds: 3.2 },
    created_at: "2026-01-01T10:00:00+00:00",
    ...over,
  };
}

export function makePlan(over: Partial<Plan> = {}): Plan {
  const shared = { model_type: "mlp" as const, normalize: true, round: 1 };
  return {
    mode: "effect",
    candidates: [
      { id: "c1", label: "dropout=0", hyperparameters: { epochs: 10, dropout: 0 }, parent: null,
        knob: "dropout", value: 0, ...shared },
      { id: "c2", label: "dropout=0.2", hyperparameters: { epochs: 10, dropout: 0.2 }, parent: "c1",
        knob: "dropout", value: 0.2, ...shared },
    ],
    reference: "c1",
    factor: "dropout",
    base: { model_type: "mlp", hyperparameters: { epochs: 10 }, normalize: null },
    rationale: "Vary dropout and hold everything else fixed.",
    ...over,
  };
}

export function makeComparison(over: Partial<Comparison> = {}): Comparison {
  return {
    a: "c2",
    label: "dropout=0.2",
    b: "c1",
    against: "dropout=0",
    anchor: "reference",
    diff: 0.021,
    ci_low: 0.008,
    ci_high: 0.034,
    confidence: 0.95,
    verdict: "better",
    ...over,
  };
}

export function makeCondition(over: Partial<ConditionSummary> = {}): ConditionSummary {
  return {
    id: "c1",
    label: "dropout=0",
    family: "mlp",
    parent: null,
    change: "dropout=0",
    is_reference: true,
    n_ok: 3,
    n_failed: 0,
    mean: 0.79,
    seed_std: 0.002,
    train_metric: 0.81,
    gap: 0.02,
    contender: false,
    ...over,
  };
}

export function makeAnalysis(over: Partial<Analysis> = {}): Analysis {
  return {
    split: "val",
    metric: "accuracy",
    higher_is_better: true,
    mode: "effect",
    n_rows: 1575,
    majority_rate: 0.6,
    leader: "c2",
    contenders: [],
    conditions: [
      makeCondition(),
      makeCondition({ id: "c2", label: "dropout=0.2", parent: "c1", change: "dropout=0.2", is_reference: false,
                      n_ok: 2, n_failed: 1, mean: 0.811, seed_std: 0.003 }),
    ],
    comparisons: [makeComparison()],
    ...over,
  };
}

export function makeDecision(over: Partial<Decision> = {}): Decision {
  return {
    round: 1,
    action: "conclude",
    parent: null,
    knob: null,
    values: [],
    new_candidates: [],
    rationale: "dropout=0.2 is clearly better; nothing untried would change that.",
    decided_by: "agent",
    ...over,
  };
}

export function makeReport(over: Partial<Report> = {}): Report {
  return {
    mode: "effect",
    winner: "dropout=0.2",
    winner_id: "c2",
    reference: "dropout=0",
    runner_up: null,
    metric: "accuracy",
    higher_is_better: true,
    winner_score: { value: 0.812, ci_low: 0.79, ci_high: 0.83 },
    reference_score: { value: 0.793, ci_low: 0.77, ci_high: 0.81 },
    primary: makeComparison({ diff: 0.019, ci_low: 0.005, ci_high: 0.033 }),
    secondary: null,
    confidence: 0.95,
    winner_val_score: 0.815,
    val_to_test_drop: 0.003,
    effort: { mlp: 2 },
    candidates_tried: 2,
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
