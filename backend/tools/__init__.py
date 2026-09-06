# backend/tools/__init__.py
"""
backend/tools
==============
Deterministic tools for the Adaptive ML Experiment Agent.

Implemented in Phase 2
-----------------------
state_manager.py   - StateManager: PostgreSQL interface for all persistence
                     (sessions, experiments, anomalies, and datasets)

Implemented in Phase 3 (revised for the dataset-first architecture)
------------------------------------------------------------------
dataset/           - CSV ingestion, preprocessing, splitting, and the
                     Dataset facade (backend/tools/dataset/); replaces the
                     old MNIST/synthetic-regression-specific data_loaders.py
models.py          - TabularMLP: a generic, dataset-agnostic PyTorch module
trainers.py        - train_mlp, train_linear_baseline
experiment_runner.py - ExperimentRunner: dispatches to trainers, handles errors
statistical_analyzer.py - StatisticalAnalyzer: scipy t-tests, effect sizes, CIs
anomaly_templates.py    - ANOMALY_TEMPLATES: template strings for explanations
anomaly_detector.py     - AnomalyDetector: rule-based anomaly detection

None of these touch the database directly - they take/return the Pydantic
models from backend.models (including DatasetProfile) and are wired into
persistence (via StateManager) by the Phase 5 LangGraph nodes.
"""
