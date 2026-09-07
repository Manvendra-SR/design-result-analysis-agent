"""
tests/unit/test_api_datasets.py
=================================
Tests for the dataset ingestion endpoints (backend/api/routes/datasets.py).

These run ``ingest_csv`` for real (it is deterministic and fast) - only the
agents / runner are stubbed elsewhere.

Test cases
----------
1.  test_ingest_valid_csv_returns_profile
2.  test_ingest_classification_inference
3.  test_ingest_task_type_override
4.  test_ingest_missing_target_column_returns_400
5.  test_ingest_too_few_rows_returns_400
6.  test_ingest_empty_body_returns_400
7.  test_list_datasets
8.  test_get_dataset_detail
9.  test_get_dataset_unknown_404
"""

from __future__ import annotations

import pytest

from tests._fakes import make_test_client


@pytest.fixture
def client(state_manager):
    api_client, _ = make_test_client(state_manager)
    return api_client


def _classification_csv(rows: int = 40) -> str:
    lines = ["feat_a,feat_b,label"]
    for i in range(rows):
        lines.append(f"{i * 0.5},{i % 7},{i % 2}")
    return "\n".join(lines)


def _regression_csv(rows: int = 40) -> str:
    lines = ["feat_a,feat_b,target"]
    for i in range(rows):
        lines.append(f"{i * 0.5},{i % 7},{i * 1.7 + 3.2}")
    return "\n".join(lines)


def test_ingest_valid_csv_returns_profile(client) -> None:
    resp = client.post(
        "/api/datasets",
        json={"filename": "c.csv", "csv_content": _classification_csv(), "target_column": "label"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["dataset_id"]
    assert body["target_column"] == "label"
    assert body["n_rows"] == 40
    assert body["feature_columns"] == ["feat_a", "feat_b"]


def test_ingest_classification_inference(client) -> None:
    resp = client.post(
        "/api/datasets",
        json={"filename": "c.csv", "csv_content": _classification_csv(), "target_column": "label"},
    )
    body = resp.json()
    assert body["task_type"] == "classification"
    assert body["task_type_source"] == "inferred"
    assert body["n_classes"] == 2


def test_ingest_task_type_override(client) -> None:
    resp = client.post(
        "/api/datasets",
        json={
            "filename": "r.csv",
            "csv_content": _regression_csv(),
            "target_column": "target",
            "task_type_override": "regression",
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["task_type"] == "regression"
    assert body["task_type_source"] == "user_specified"


def test_ingest_missing_target_column_returns_400(client) -> None:
    resp = client.post(
        "/api/datasets",
        json={"filename": "c.csv", "csv_content": _classification_csv(), "target_column": "nope"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_dataset"


def test_ingest_too_few_rows_returns_400(client) -> None:
    resp = client.post(
        "/api/datasets",
        json={"filename": "c.csv", "csv_content": _classification_csv(rows=5), "target_column": "label"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_dataset"


def test_ingest_empty_body_returns_400(client) -> None:
    resp = client.post(
        "/api/datasets",
        json={"filename": "c.csv", "csv_content": "", "target_column": "label"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "validation_error"


def test_list_datasets(client) -> None:
    client.post(
        "/api/datasets",
        json={"filename": "c.csv", "csv_content": _classification_csv(), "target_column": "label"},
    )
    resp = client.get("/api/datasets")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_get_dataset_detail(client) -> None:
    created = client.post(
        "/api/datasets",
        json={"filename": "c.csv", "csv_content": _classification_csv(), "target_column": "label"},
    ).json()
    resp = client.get(f"/api/datasets/{created['dataset_id']}")
    assert resp.status_code == 200
    assert resp.json()["dataset_id"] == created["dataset_id"]


def test_get_dataset_unknown_404(client) -> None:
    resp = client.get("/api/datasets/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404
    assert resp.json()["error"] == "not_found"
