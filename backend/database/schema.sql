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
-- No migration tool (Alembic) exists in this project. If you already had a
-- database initialised before the datasets table/sessions.dataset_id
-- column existed, drop and recreate it before re-running init_db - the
-- CREATE TABLE IF NOT EXISTS statements below only handle fresh creation.
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
    research_question       TEXT        NOT NULL,
    status                  VARCHAR(20) NOT NULL DEFAULT 'active',
    -- Tracks the current LangGraph node for crash recovery (Phase 5).
    -- Values: 'planning' | 'executing' | 'validating' | 'analyzing'
    --         | 'recommending' | 'concluded'
    current_node            VARCHAR(20) NOT NULL DEFAULT 'planning',
    -- JSON blob of the latest Recommendation (overwritten each cycle).
    -- NULL until the first recommendation is generated.
    current_recommendation  TEXT        DEFAULT NULL,
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
    -- train_loss, val_loss, training_time_seconds). NULL while pending/running.
    metrics         JSONB       DEFAULT NULL,
    -- 'pending' | 'running' | 'success' | 'failed' | 'anomalous'
    status          VARCHAR(20) NOT NULL DEFAULT 'pending',
    -- Error message when status = 'failed'
    error           TEXT        DEFAULT NULL,
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
    detected_at     TIMESTAMP   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_anomalies_experiment
    ON anomalies (experiment_id);
