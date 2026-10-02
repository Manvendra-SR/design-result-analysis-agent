"""
backend/tools
==============
The deterministic side of the system. Nothing here calls an LLM.

dataset/              CSV ingestion, profiling, fixed train/val/test split, preprocessing
models.py             TabularMLP
trainers.py           train_mlp, train_linear_baseline -> metrics + per-row scores
experiment_runner.py  run_experiment: train one configuration; crash/NaN -> 'failed'
stats.py              paired-bootstrap comparisons vs the reference level
"""
