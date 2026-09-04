# backend/tools/__init__.py
"""
backend/tools
==============
Deterministic tools for the Adaptive ML Experiment Agent.

Implemented in Phase 2
-----------------------
state_manager.py   - StateManager: PostgreSQL interface for all persistence

Implemented in Phase 3
-----------------------
data_loaders.py    - MNIST data loading + synthetic regression generation
models.py          - MNISTClassifier PyTorch module
trainers.py        - train_mnist_mlp, train_synthetic_regression
experiment_runner.py - ExperimentRunner: dispatches to trainers, handles errors
statistical_analyzer.py - StatisticalAnalyzer: scipy t-tests, effect sizes, CIs
anomaly_templates.py    - ANOMALY_TEMPLATES: template strings for explanations
anomaly_detector.py     - AnomalyDetector: rule-based anomaly detection
"""
