"""
backend/agents
===============
The LLM side of the system. Nothing here computes a statistic or trains a model.

llm.py          GroqClient (strict JSON-schema output), request_json, LLMError
planner.py      Planner: question + dataset profile -> Plan (factor, levels, reference)
recommender.py  Recommender: decide (explore | conclude), interpret (final explanation)
"""
