"""
backend/api/routes/sessions.py
================================
    POST   /api/sessions            create an investigation (question + dataset)
    GET    /api/sessions            list investigations
    GET    /api/sessions/{id}       detail: plan, decisions, runs, live analysis, report
    DELETE /api/sessions/{id}       delete an investigation and its runs
    POST   /api/sessions/{id}/run   start the investigation in the background (202)

The run executes after the response is sent; the UI polls ``GET /{id}``.
"""

from __future__ import annotations

from typing import Callable, List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

import backend.config as config
from backend.api.dependencies import get_investigator, get_repository
from backend.api.schemas import CreateSessionRequest, DeleteResult, SessionDetail
from backend.database.repository import Repository
from backend.models.investigation import Session, SessionSummary
from backend.tools import stats

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("", response_model=Session, status_code=201)
def create_session(body: CreateSessionRequest, repo: Repository = Depends(get_repository)) -> Session:
    return repo.create_session(body.research_question.strip(), body.dataset_id)  # KeyError -> 404


@router.get("", response_model=List[SessionSummary])
def list_sessions(repo: Repository = Depends(get_repository)) -> List[SessionSummary]:
    return repo.list_sessions()


@router.get("/{session_id}", response_model=SessionDetail)
def get_session(session_id: str, repo: Repository = Depends(get_repository)) -> SessionDetail:
    session = repo.get_session(session_id)  # KeyError -> 404
    experiments = repo.list_experiments(session_id)
    analysis = None
    if session.plan is not None and any(e.status == "ok" for e in experiments):
        profile = repo.get_dataset(session.dataset_id)
        analysis = stats.analyze(session.plan, experiments, profile, "val")
    return SessionDetail(
        **session.model_dump(),
        experiments=experiments,
        analysis=analysis,
        max_rounds=config.MAX_ROUNDS,
        n_seeds=config.N_SEEDS,
    )


@router.delete("/{session_id}", response_model=DeleteResult)
def delete_session(session_id: str, repo: Repository = Depends(get_repository)) -> DeleteResult:
    deleted = repo.delete_session(session_id)  # KeyError -> 404
    return DeleteResult(deleted="session", id=session_id, experiments_deleted=deleted)


@router.post("/{session_id}/run", response_model=Session, status_code=202)
def run_session(
    session_id: str,
    background: BackgroundTasks,
    repo: Repository = Depends(get_repository),
    investigate: Callable[[str], None] = Depends(get_investigator),
) -> Session:
    session = repo.get_session(session_id)  # KeyError -> 404
    if session.status == "running":
        raise HTTPException(status_code=409, detail="This investigation is already running.")
    if session.status == "done":
        raise HTTPException(
            status_code=409, detail="This investigation has finished. Start a new one to ask another question."
        )
    if session.status == "failed":
        repo.reset_session(session_id)  # start over cleanly rather than resuming half a run
    repo.set_status(session_id, "running")  # before responding, so the UI sees it immediately
    background.add_task(investigate, session_id)
    return repo.get_session(session_id)
