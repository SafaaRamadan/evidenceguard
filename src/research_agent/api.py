"""FastAPI service exposing the EvidenceGuard agent.

Run with:  uvicorn research_agent.api:app
"""

import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Protocol

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from research_agent.agent import AgentAnswer, build_agent
from research_agent.config import Settings

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 1000
NOTE = (
    "Citation checks are structural: they confirm cited passages exist, not that "
    "the answer is factually correct."
)


class AnswersQuestions(Protocol):
    def answer(self, question: str) -> AgentAnswer: ...


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)

    @field_validator("question")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class EvidenceItem(BaseModel):
    evidence_id: str
    source: str
    section: str
    source_url: str | None
    score: float
    cited: bool
    text: str


class AskResponse(BaseModel):
    question: str
    status: str  # answered, abstained, or unverified
    answer: str
    abstained: bool
    citations_valid: bool
    issues: list[str]
    attempts: int
    latency_seconds: float
    evidence: list[EvidenceItem]
    note: str = NOTE


def _to_response(result: AgentAnswer, latency_seconds: float) -> AskResponse:
    cited = set(result.citations.cited_ids)
    evidence = [
        EvidenceItem(
            evidence_id=evidence_id,
            source=hit.source,
            section=hit.section,
            source_url=hit.source_url,
            score=round(hit.score, 3),
            cited=evidence_id in cited,
            text=hit.text,
        )
        for evidence_id, hit in result.evidence.items()
    ]
    return AskResponse(
        question=result.question,
        status=result.status,
        answer=result.text,
        abstained=result.abstained,
        citations_valid=result.citations.passed,
        issues=list(result.citations.issues),
        attempts=result.attempts,
        latency_seconds=round(latency_seconds, 2),
        evidence=evidence,
    )


def create_app(agent: AnswersQuestions | None = None) -> FastAPI:
    """Build the app. Pass a fake agent in tests; otherwise models load at startup."""
    lock = threading.Lock()  # one model, one request at a time

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if agent is None:
            logger.info("Loading models and indexing documents...")
            app.state.agent = build_agent(Settings.from_env())
        yield

    app = FastAPI(title="EvidenceGuard", version="0.1.0", lifespan=lifespan)
    app.state.agent = agent

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "ready": app.state.agent is not None}

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        active = app.state.agent
        if active is None:
            raise HTTPException(status_code=503, detail="The service is still starting up.")
        start = time.perf_counter()
        try:
            with lock:
                result = active.answer(request.question)
        except Exception:
            logger.exception("Unexpected error while answering a question")
            raise HTTPException(
                status_code=500, detail="Internal error while answering the question."
            ) from None
        return _to_response(result, time.perf_counter() - start)

    return app


app = create_app()