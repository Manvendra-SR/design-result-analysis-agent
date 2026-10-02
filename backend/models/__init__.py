"""
backend/models
===============
Pydantic data models shared by the tools, agents, persistence and API.

dataset.py        DatasetProfile, PreprocessingConfig, DatasetValidationError, DatasetInUseError
experiment.py     ExperimentConfiguration, ExperimentResult
investigation.py  Plan, Decision, Analysis, Report, Session, SessionSummary
timestamps.py     UTCDateTime (explicit-UTC serialisation)
"""
