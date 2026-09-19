-- ============================================================
-- Adaptive ML Experiment Agent — Database Schema
-- ============================================================
-- 4 tables: datasets, sessions, experiments, anomalies
-- ("datasets" added by the dataset-first architecture revision - see
-- DESIGN_REVIEW_CHANGES.md - so a session/experiment can reference a
-- user-uploaded dataset instead of a hardcoded one.)
-- Statistical comparisons are computed on-demand (not persisted).
-- Latest recommendation is stored as a JSON blob in sessions.
--
-- Requires PostgreSQL 13+ (gen_random_uuid(), JSONB, TIMESTAMP).
-- Run via: python -m backend.database.init_db
--
-- No migration tool (Alembic) exists in this project, and none is wanted
-- during development: the CREATE TABLE IF NOT EXISTS statements below only
-- handle fresh creation, so after ANY schema change here (the datasets
-- table, sessions.dataset_id, sessions.pending_configs / latest_analysis,
-- ...) drop and recreate the dev database, then re-run init_db.
-- ============================================================

-- Enable the pgcrypto extension if gen_random_uuid() is not built-in.
-- On PostgreSQL 13+ it is available by default; this is a no-op there.
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ------------------------------------------------------------
-- datasets
-- One row per user-uploaded, ingested CSV dataset. The CSV file itself
-- lives on disk under data/uploads/<dataset_id>/ - only its profile
-- metadata is stored here.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    original_filename   TEXT        NOT NULL,
    storage_path        TEXT        NOT NULL,
    -- Full DatasetProfile as JSON: target_column, feature/numeric/categorical
    -- columns, task_type, task_type_source, n_rows/n_features, n_classes,
    -- class_labels, class_distribution, missing_value_counts, split_seed.
    profile             JSONB       NOT NULL,
    created_at          TIMESTAMP   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_datasets_created_at
    ON datasets (created_at DESC);

-- ------------------------------------------------------------
-- sessions
-- One row per research investigation (research question + adaptive loop),
-- scoped to exactly one dataset.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    session_id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id              UUID        NOT NULL REFERENCES datasets (dataset_id),
    -- The investigation this one follows up on, if any. Provenance ONLY:
    -- experiments/anomalies/cycle_history are strictly scoped to their own
    -- session_id, so a follow-up question can never inherit the previous
    -- question's evidence.
    parent_session_id       UUID
                                REFERENCES sessions (session_id) ON DELETE SET NULL,
    research_question       TEXT        NOT NULL,
    status                  VARCHAR(20) NOT NULL DEFAULT 'active',
    -- Why the investigation stopped, set when status becomes 'concluded':
    --   'agent_concluded' - the Recommender judged the evidence sufficient
    --   'cycle_limit'     - the MAX_ADAPTIVE_CYCLES safety cap stopped a loop
    --                       that still wanted to continue. NOT a settled answer;
    --                       the UI must present it differently.
    termination_reason      VARCHAR(32) DEFAULT NULL,
    -- Tracks the current LangGraph node for crash recovery (Phase 5).
    -- Values: 'planning' | 'executing' | 'validating' | 'analyzing'
    --         | 'recommending' | 'concluded'
    current_node            VARCHAR(20) NOT NULL DEFAULT 'planning',
    -- Whether a POST /run-cycle is actually in progress, distinct from
    -- `status` ('active' just means "not concluded"). Set by
    -- state_machine/executor.py: 'running' on entry, 'failed' + run_error on
    -- an unhandled node exception, 'idle' on a clean finish. Lets the UI show
    -- running / failed / idle correctly, incl. after a page refresh.
    -- Values: 'idle' | 'running' | 'failed'
    run_phase               VARCHAR(16) NOT NULL DEFAULT 'idle',
    run_error               TEXT        DEFAULT NULL,
    -- JSON blob of the LATEST Recommendation (overwritten each cycle).
    -- NULL until the first recommendation is generated. For the per-cycle
    -- history, see cycle_history below.
    current_recommendation  TEXT        DEFAULT NULL,
    -- Phase 5 workflow scratch space (overwritten each time the relevant
    -- node runs; NULL between uses). Kept as columns on `sessions` rather
    -- than a separate table for the same reason as current_recommendation -
    -- ephemeral per-cycle state, not history. See backend/state_machine/nodes.py.
    --   pending_configs: JSON list of ExperimentConfigurations the planning
    --     or recommendation node queued for the next execution node.
    --   latest_analysis: JSON list of StatisticalComparisons the analysis
    --     node computed for the recommendation node to interpret.
    pending_configs         TEXT        DEFAULT NULL,
    latest_analysis         TEXT        DEFAULT NULL,
    -- Phase 5 investigation HISTORY (not overwritten - appended). The
    -- adaptive loop runs many cycles per invocation with no human in
    -- between, so the per-cycle recommendation + analysis must be retained
    -- for the UI. JSON array of CycleHistoryEntry (see backend/models/cycle.py).
    --   plan_explanation: the planner's rationale for the initial (cycle 1)
    --     experiment design - written once, otherwise discarded.
    --   cycle_history: [{cycle_number, recommendation, statistical_comparisons,
    --     recorded_at}, ...], appended by the recommending node once per cycle.
    plan_explanation        TEXT        DEFAULT NULL,
    cycle_history           TEXT        DEFAULT NULL,
    cycle_count             INTEGER     NOT NULL DEFAULT 0,
    created_at              TIMESTAMP   NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMP   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_status
    ON sessions (status);

CREATE INDEX IF NOT EXISTS idx_sessions_created_at
    ON sessions (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_sessions_dataset
    ON sessions (dataset_id);

CREATE INDEX IF NOT EXISTS idx_sessions_parent
    ON sessions (parent_session_id);

-- ------------------------------------------------------------
-- experiments
-- One row per individual ML experiment run within a session.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id   UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID        NOT NULL
                        REFERENCES sessions (session_id) ON DELETE CASCADE,
    -- Full ExperimentConfiguration as JSON
    -- e.g. {"dataset_id": "...", "model_type": "mlp", "hyperparameters": {...},
    --       "preprocessing": {"normalize": false}, "random_seed": 42}
    config          JSONB       NOT NULL,
    -- 'classification' | 'regression' - the dataset's task type when this
    -- experiment ran. NULL only if the experiment failed before dataset resolution.
    task_type       VARCHAR(20) DEFAULT NULL,
    -- Metrics dict, keyed by task_type (classification: train_loss, val_loss,
    -- accuracy, n_classes, n_val_samples, training_time_seconds; regression:
    -- train_loss, val_loss, training_time_seconds). mlp additionally records
    -- initial_train_loss, best_epoch, epochs_ran, final_val_loss and
    -- final_accuracy - its val_loss/accuracy are the best epoch's, not the
    -- last one's (see backend/tools/trainers.py). NULL while pending/running.
    metrics         JSONB       DEFAULT NULL,
    -- 'pending' | 'running' | 'success' | 'failed' | 'anomalous'
    status          VARCHAR(20) NOT NULL DEFAULT 'pending',
    -- Error message when status = 'failed'
    error           TEXT        DEFAULT NULL,
    -- 1-based adaptive cycle that produced this experiment (set by the
    -- execution node). NULL for rows written outside the loop.
    cycle           INTEGER     DEFAULT NULL,
    timestamp       TIMESTAMP   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_experiments_session
    ON experiments (session_id);

CREATE INDEX IF NOT EXISTS idx_experiments_session_status
    ON experiments (session_id, status);

CREATE INDEX IF NOT EXISTS idx_experiments_timestamp
    ON experiments (timestamp DESC);

-- GIN index enables fast JSONB queries on config fields
-- (used for grouping experiments by hyperparameter values in Phase 3)
CREATE INDEX IF NOT EXISTS idx_experiments_config
    ON experiments USING gin (config);

-- ------------------------------------------------------------
-- anomalies
-- One row per detected anomaly, linked to a specific experiment.
-- Populated by the Anomaly_Detector (Phase 3).
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS anomalies (
    anomaly_id      UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id   UUID        NOT NULL
                        REFERENCES experiments (experiment_id) ON DELETE CASCADE,
    -- Rule that triggered detection:
    -- 'outlier_detection' | 'loss_divergence' | 'validation_collapse'
    rule            VARCHAR(50) NOT NULL,
    -- Template-generated natural language explanation (no LLM call)
    explanation     TEXT        NOT NULL,
    -- 'warning' | 'critical'
    severity        VARCHAR(20) NOT NULL,
    -- Anomaly LIFECYCLE. Detection re-runs over every experiment in the session
    -- on every cycle, so a flag raised against a 3-replicate group can be
    -- withdrawn once the group grows and the value proves ordinary. Resolved
    -- rows are KEPT (never deleted) so "what was flagged, and when it cleared"
    -- stays in the history; resolved_cycle IS NULL means the flag is still open.
    detected_cycle  INTEGER     DEFAULT NULL,
    resolved_cycle  INTEGER     DEFAULT NULL,
    detected_at     TIMESTAMP   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_anomalies_experiment
    ON anomalies (experiment_id);

CREATE INDEX IF NOT EXISTS idx_anomalies_open
    ON anomalies (resolved_cycle);
