"""
Physics Master web service (FastAPI).

Endpoints (all /api/* need the header  X-API-Key: <APP_API_KEY>):
  GET  /                          browser UI
  GET  /health                    health check (no auth)
  POST /api/jobs/generate         start a paper job        -> {"job_id": ...}
  POST /api/jobs/remedial         diagnose a wrong answer  -> {"job_id": ...}
  GET  /api/jobs/{id}             status / progress / result
  GET  /api/jobs/{id}/paper.txt   question paper (text)
  GET  /api/jobs/{id}/solutions.txt  answer key + solutions (text)

Generation takes many minutes, so it runs as a background job and the client
polls. Jobs live in the database; jobs interrupted by a restart are marked failed.
"""

from __future__ import annotations

import os

os.environ.setdefault("LOG_TO_STDOUT", "1")   # must be set before importing physics_master

import hmac
import json
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

import importlib

# The engine module can be switched with the ENGINE_MODULE env var (no code change needed).
pm = importlib.import_module(os.getenv("ENGINE_MODULE", "free_engine_v2"))

log = logging.getLogger("physics_master.server")

APP_API_KEY = os.getenv("APP_API_KEY", "")
JOB_WORKERS = int(os.getenv("JOB_WORKERS", "1"))   # keep 1: shared novelty index + API rate limits
INDEX_HTML = Path(__file__).parent / "index.html"

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if len(APP_API_KEY) < 16:
        raise RuntimeError("APP_API_KEY must be set (16+ characters)")
    state["agent"] = pm.PhysicsMasterAgent()          # also validates the provider keys
    state["agent"].db.fail_stale_jobs()
    state["pool"] = ThreadPoolExecutor(max_workers=JOB_WORKERS)
    log.info("Physics Master ready (mode=%s, postgres=%s)", state["agent"].mode, pm.USE_POSTGRES)
    yield
    state["pool"].shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Physics Master", lifespan=lifespan)


def require_auth(x_api_key: str = Header(default="")) -> None:
    if not hmac.compare_digest(x_api_key.encode(), APP_API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Invalid API key")


# ------------------------------------------------------------
# background jobs
# ------------------------------------------------------------

def run_generate(job_id: str, request: pm.PaperRequest) -> None:
    agent: pm.PhysicsMasterAgent = state["agent"]
    db = agent.db
    db.update_job(job_id, status="running")
    try:
        questions = agent.generate_paper(
            request, progress=lambda info: db.update_job(job_id, progress=json.dumps(info)))
        quota_hit = pm.QUOTA_HIT.is_set()
        if not questions:
            db.update_job(job_id, status="failed", error=(
                "The free daily Gemini limit is used up. Please try again tomorrow."
                if quota_hit else "No question passed verification. Check the server logs."))
            return
        db.update_job(job_id, status="done", result=json.dumps({
            "partial": len(questions) < request.count,
            "quota_exhausted": quota_hit,
            "questions": [q.model_dump() for q in questions],
        }))
    except Exception as exc:
        log.exception("generate job %s failed", job_id)
        db.update_job(job_id, status="failed", error=f"{type(exc).__name__}: {exc}"[:500])


def run_remedial(job_id: str, question_id: str, student_answer: str) -> None:
    agent: pm.PhysicsMasterAgent = state["agent"]
    db = agent.db
    db.update_job(job_id, status="running")
    try:
        original = db.get_question(question_id)
        if not original:
            db.update_job(job_id, status="failed", error="Question not found")
            return
        if original.question_type == "fill_blank":
            correct = pm.answers_match(original, [student_answer])
        else:
            import re
            correct = pm.answers_match(original, re.findall(r"[A-Da-d]", student_answer.upper()))
        if correct:
            db.record_attempt(original.id, student_answer, True)
            db.update_job(job_id, status="done", result=json.dumps({"correct": True}))
            return
        engine = pm.RemedialEngine(agent)
        analysis = engine.analyze(original, student_answer)
        db.record_attempt(original.id, student_answer, False, analysis)
        db.update_job(job_id, progress=json.dumps({"phase": "generating remedial question"}))
        new_q = engine.generate(original, analysis)
        if not new_q:
            db.update_job(job_id, status="failed",
                          error="Diagnosis done, but no verified remedial question was produced.")
            return
        db.update_job(job_id, status="done", result=json.dumps({
            "correct": False,
            "analysis": analysis.model_dump(),
            "question": new_q.model_dump(),
        }))
    except Exception as exc:
        log.exception("remedial job %s failed", job_id)
        db.update_job(job_id, status="failed", error=f"{type(exc).__name__}: {exc}"[:500])


# ------------------------------------------------------------
# routes
# ------------------------------------------------------------

class RemedialBody(BaseModel):
    question_id: str
    student_answer: str


@app.get("/health")
def health():
    return {"ok": True, "mode": state["agent"].mode}


@app.get("/")
def index():
    return FileResponse(INDEX_HTML)


@app.post("/api/jobs/generate", status_code=202, dependencies=[Depends(require_auth)])
def start_generate(request: pm.PaperRequest):
    db = state["agent"].db
    job_id = uuid.uuid4().hex
    db.create_job(job_id, "generate", request.model_dump_json())
    state["pool"].submit(run_generate, job_id, request)
    return {"job_id": job_id}


@app.post("/api/jobs/remedial", status_code=202, dependencies=[Depends(require_auth)])
def start_remedial(body: RemedialBody):
    db = state["agent"].db
    if not db.get_question(body.question_id):
        raise HTTPException(status_code=404, detail="Question not found")
    job_id = uuid.uuid4().hex
    db.create_job(job_id, "remedial", body.model_dump_json())
    state["pool"].submit(run_remedial, job_id, body.question_id, body.student_answer)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}", dependencies=[Depends(require_auth)])
def get_job(job_id: str):
    job = state["agent"].db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _finished_questions(job_id: str) -> List[pm.Question]:
    job = state["agent"].db.get_job(job_id)
    if not job or job["kind"] != "generate" or job["status"] != "done":
        raise HTTPException(status_code=404, detail="No finished paper for this job")
    return [pm.Question.model_validate(q) for q in job["result"]["questions"]]


@app.get("/api/jobs/{job_id}/paper.txt", dependencies=[Depends(require_auth)])
def paper_txt(job_id: str):
    return PlainTextResponse(pm.paper_text(_finished_questions(job_id)))


@app.get("/api/jobs/{job_id}/solutions.txt", dependencies=[Depends(require_auth)])
def solutions_txt(job_id: str):
    return PlainTextResponse(pm.solutions_text(_finished_questions(job_id)))
