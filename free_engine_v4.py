"""
PHYSICS MASTER AGENT  (v2)
==========================

Autonomous NEET / JEE Main / JEE Advanced Physics question generator with
cross-model verification.

What changed vs v1
------------------
- Structured output: Claude via forced tool use, Gemini via response_schema.
  LaTeX backslashes (\\theta, \\tau, \\frac ...) no longer corrupt the JSON.
- Ids are always assigned by code; exam/chapter/type are forced by code.
- Cross-model verification: questions written by one model are BLIND-SOLVED
  and then AUDITED by the other model. A question is accepted only if the
  independent solver reproduces the claimed answer.
- Optional numeric check: the generator supplies a small Python/SymPy snippet
  that computes the answer; it is executed in a subprocess and compared.
- Type quotas and difficulty bands are tracked and enforced across rounds.
- Candidate batches are sized from what is still missing (with overshoot),
  generation and verification run in thread pools, and a partial paper is
  returned (never thrown away) if the target cannot be reached.
- Full question JSON is stored in SQLite (new tables, so an old DB file does
  not clash), which makes remedial generation and attempt tracking work.
- Remedial questions go through the same verification pipeline.
- Real exceptions are logged to physics_master.log.

Honest limits
-------------
- Two LLMs agreeing is strong evidence, not proof. Have a human teacher skim
  the final paper, especially JEE Advanced items.
- Originality is checked against YOUR question bank only (TF-IDF + hash).
  It cannot detect copying from textbooks or past papers; the auditor's
  "resembles a known problem" flag is a weak self-report.
- The numeric check runs model-written Python in a subprocess (isolated mode
  + timeout, NOT a hardened sandbox). Set ENABLE_CODE_CHECK=0 to disable, or
  run the whole thing in a container.

Setup
-----
pip install anthropic google-genai pydantic python-dotenv sympy scikit-learn

.env:
GEMINI_API_KEY=your_key                # required (free key: aistudio.google.com/apikey)
GEMINI_MODEL=gemini-2.5-flash          # a Gemini model your key can use
# Optional, for the best checking quality (paid API):
ANTHROPIC_API_KEY=your_key
CLAUDE_MODEL=claude-sonnet-5

FREE MODE: with no ANTHROPIC_API_KEY the app runs on Gemini only. Two separate Gemini
calls write and then check each question (optionally two different models via
GEMINI_MODEL_2). Calls are throttled to GEMINI_RPM (default 8/min) to fit the free tier;
when the free daily quota runs out it stops and returns what it has verified.

Optional: DATABASE_PATH, AUDIT_WORKERS, ENABLE_CODE_CHECK, MAX_ROUNDS

Run:
python physics_master.py
"""

from __future__ import annotations

import os
import re
import sys
import json
import uuid
import math
import time
import threading
import sqlite3
import hashlib
import logging
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

import sympy as sp

from anthropic import Anthropic
from google import genai
from google.genai import types as genai_types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_MODEL_2 = os.getenv("GEMINI_MODEL_2", "")   # optional 2nd Gemini model (free mode)
# Free Gemini keys allow roughly 10 requests/minute per model; stay just under it.
GEMINI_RPM = float(os.getenv("GEMINI_RPM", "60" if ANTHROPIC_API_KEY else "8"))

DB_FILE = os.getenv("DATABASE_PATH", "physics_master.db")
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
USE_POSTGRES = DATABASE_URL.startswith(("postgres://", "postgresql://"))

MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "15"))
MAX_PER_CALL = 12            # candidates requested per generation call
OVERSHOOT = float(os.getenv("OVERSHOOT", "1.6" if ANTHROPIC_API_KEY else "1.3"))  # candidates per missing question
RELAX_BANDS_AFTER = 4        # after this many rounds, difficulty bands are soft
AUDIT_WORKERS = int(os.getenv("AUDIT_WORKERS", "6"))
ENABLE_CODE_CHECK = os.getenv("ENABLE_CODE_CHECK", "1") == "1"
CODE_TIMEOUT_SECONDS = 10
NOVELTY_THRESHOLD = 0.80

_log_handlers = (
    [logging.StreamHandler(sys.stdout)] if os.getenv("LOG_TO_STDOUT") == "1"
    else [logging.FileHandler("physics_master.log", encoding="utf-8")]
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=_log_handlers,
)
log = logging.getLogger("physics_master")


def require_keys() -> None:
    # Gemini is required. Claude is optional: without ANTHROPIC_API_KEY the app runs
    # in free "gemini-only" mode (see PhysicsMasterAgent).
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY (get a free one at aistudio.google.com/apikey)")


# ============================================================
# ENUMS / CONSTANTS
# ============================================================

class Exam(str, Enum):
    NEET = "neet"
    JEE_MAIN = "jee_main"
    JEE_ADVANCED = "jee_advanced"


class QuestionType(str, Enum):
    MCQ = "mcq"
    MULTIPLE_CORRECT = "multiple_correct"
    ASSERTION_REASON = "assertion_reason"
    MATCH_COLUMN = "match_column"
    FILL_BLANK = "fill_blank"


ALL_TYPES = [t.value for t in QuestionType]

AR_OPTIONS = [
    "Both Assertion (A) and Reason (R) are true, and R is the correct explanation of A",
    "Both Assertion (A) and Reason (R) are true, but R is NOT the correct explanation of A",
    "Assertion (A) is true but Reason (R) is false",
    "Assertion (A) is false but Reason (R) is true",
]

TYPE_ALIASES = {
    "match_the_column": "match_column",
    "match_columns": "match_column",
    "matching": "match_column",
    "fill_in_the_blank": "fill_blank",
    "fill_in_the_blanks": "fill_blank",
    "assertion_and_reason": "assertion_reason",
    "multi_correct": "multiple_correct",
    "single_correct": "mcq",
}

BAND_MIX = {
    "neet": {"easy": 0.35, "medium": 0.50, "hard": 0.15},
    "jee_main": {"easy": 0.25, "medium": 0.50, "hard": 0.25},
    "jee_advanced": {"easy": 0.10, "medium": 0.30, "hard": 0.60},
}

TYPE_SPECS = {
    "mcq": (
        "MCQ: exactly 4 options WITHOUT letter prefixes. Exactly one is correct. "
        'correct_answer = one letter, e.g. ["C"]. Distractors must come from '
        "plausible mistakes (sign error, wrong formula, wrong reference point)."
    ),
    "multiple_correct": (
        "MULTIPLE CORRECT: exactly 4 options WITHOUT letter prefixes. One or "
        "more may be correct; evaluate every option independently. "
        'correct_answer = ALL correct letters, e.g. ["A", "C"].'
    ),
    "assertion_reason": (
        "ASSERTION-REASON: put both statements in the question field as "
        "'Assertion (A): ...' and 'Reason (R): ...'. Leave options as an empty "
        "list (the standard 4 options are added automatically: "
        "A = both true and R explains A; B = both true, R does not explain A; "
        "C = A true, R false; D = A false, R true). correct_answer = one letter."
    ),
    "match_column": (
        "MATCH THE COLUMN: put 'Column I' (items A-D) and 'Column II' (items "
        "p-s) in the question field. options = exactly 4 distinct complete "
        "mappings, e.g. 'A-q, B-p, C-s, D-r' (no letter prefixes). "
        'correct_answer = one letter naming the correct mapping option.'
    ),
    "fill_blank": (
        "FILL IN THE BLANK: the question contains a blank (____). options = "
        "empty list. correct_answer = ONE string with the final value (with "
        "units if physical). The answer must be uniquely defined."
    ),
}


# ============================================================
# SCHEMAS
# ============================================================

class Verification(BaseModel):
    physics: bool = False
    mathematics: bool = False
    dimensions: bool = False
    options: bool = False
    originality: bool = False
    ambiguity: bool = False


class GenQuestion(BaseModel):
    """What a generator model returns. No defaults: keeps provider schemas simple."""
    subtopics: List[str]
    question: str
    options: List[str]
    correct_answer: List[str]
    solution: str
    concepts_tested: List[str]
    misconceptions_tested: List[str]
    difficulty: float
    estimated_time_seconds: int
    numeric_answer: str          # plain decimal in the SAME unit as the answer, or ""
    verification_code: str       # python snippet that sets `result`, or ""


class GenBatch(BaseModel):
    questions: List[GenQuestion]


class Question(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    exam: str
    chapter: str
    subtopics: List[str]
    question_type: str
    question: str
    options: List[str] = Field(default_factory=list)
    correct_answer: List[str]
    solution: str
    concepts_tested: List[str]
    misconceptions_tested: List[str] = Field(default_factory=list)
    difficulty: float = Field(ge=1, le=10)
    estimated_time_seconds: int = 120
    numeric_answer: str = ""
    verification_code: str = ""
    verification: Verification = Field(default_factory=Verification)
    generator: str = ""
    verified_by: str = ""
    code_checked: bool = False


class DistItem(BaseModel):
    """name/count pair. (A Dict field is rejected by the free Gemini API schema.)"""
    name: str
    count: int


class Blueprint(BaseModel):
    chapter: str
    concepts: List[str]
    subtopics: List[str]
    prerequisites: List[str]
    misconceptions: List[str]
    recommended_distribution: List[DistItem]
    difficulty_distribution: List[DistItem]
    strategy: str


class SolveResult(BaseModel):
    well_posed: bool
    answer: List[str]
    working_summary: str


class AuditResult(BaseModel):
    physics_correct: bool
    mathematics_correct: bool
    dimensions_ok: bool
    answer_correct: bool
    options_correct: bool
    unambiguous: bool
    syllabus_appropriate: bool
    resembles_known_problem: bool
    difficulty_estimate: float
    reason: str


class ErrorAnalysis(BaseModel):
    concept: str
    subtopic: str
    error_type: str
    likely_misconception: str
    recommended_difficulty: float
    explanation: str


class Ping(BaseModel):
    ok: bool


class PaperRequest(BaseModel):
    exam: str
    chapter: str
    count: int = Field(default=100, ge=1, le=100)
    question_types: List[str] = Field(default_factory=lambda: list(ALL_TYPES))
    difficulty: str = "adaptive"

    @field_validator("question_types")
    @classmethod
    def _valid_types(cls, v: List[str]) -> List[str]:
        cleaned = []
        for t in v:
            t = norm_key(t)
            t = TYPE_ALIASES.get(t, t)
            if t in ALL_TYPES and t not in cleaned:
                cleaned.append(t)
        if not cleaned:
            raise ValueError(f"No valid question types. Choose from {ALL_TYPES}")
        return cleaned


def norm_key(k: str) -> str:
    return re.sub(r"[\s\-]+", "_", k.strip().lower())


# ============================================================
# PROMPTS
# ============================================================

MASTER_SYSTEM = r"""
You are PHYSICS MASTER, a senior Physics examination designer for NEET,
JEE Main and JEE Advanced. Plan, generate, critique, verify, finalize.
Do not reveal private chain-of-thought; return decisions and results only.

Use rigorous Physics. Never invent a formula. Check assumptions, signs, units,
dimensions, limiting cases, conservation laws and physical feasibility.
Write mathematics in LaTeX where helpful.

JEE Advanced: unfamiliar situations, multi-concept reasoning, non-obvious
conservation laws, fair conceptual traps.
JEE Main: moderate mathematical and conceptual complexity.
NEET: syllabus alignment, conceptual clarity, efficient solving.
"""

GENERATOR_SYSTEM = r"""
You are an elite Physics problem setter. Generate ORIGINAL questions.
Never copy a known question and never just change the numbers of a familiar
one. Every question must give enough information for one unique, well-defined
answer. For numerical problems compute the answer independently, check units
and limiting behaviour BEFORE writing the key. Stay inside the requested
syllabus. Write mathematics in LaTeX.
"""

SOLVER_SYSTEM = r"""
You are a Physics expert solving an exam question INDEPENDENTLY. You are NOT
given the answer. Solve from first principles; do not assume any option is
correct. If the question is ambiguous, underspecified, has no correct option,
or (for single-answer types) more than one correct option, set
well_posed=false and answer=[].
"""

CRITIC_SYSTEM = r"""
You are an extremely strict Physics examiner. Your job is to find reasons to
REJECT a question: physics or mathematical errors, dimensional errors, wrong
key, wrong or duplicate options, ambiguity, hidden assumptions, multiple or no
valid answers, syllabus mismatch, unrealistic numbers, wrong difficulty.
Do not approve a question because it sounds convincing.
"""

REMEDIAL_SYSTEM = r"""
You are a Physics remediation specialist. Given a diagnosed error, write a
completely NEW question for the same learning objective: different physical
situation, different numbers, different reasoning path, different distractors,
no paraphrase of the original. Target the diagnosed misconception directly.
"""


# ============================================================
# JSON HELPERS (fallback only; providers return structured data)
# ============================================================

# LaTeX commands whose first letter collides with a valid JSON escape
# (\\frac -> form feed, \\theta -> tab, \\nu -> newline, \\beta -> backspace ...).
_LATEX_CMDS = (
    "frac|theta|tau|times|text|tan|tanh|to|top|beta|bar|begin|binom|bigl|bigr|boldsymbol|"
    "nu|nabla|neq|ne|not|rho|right|rightarrow|rangle|nolimits"
)


def _repair_escapes(text: str) -> str:
    text = re.sub(r"\\(?=(?:%s)\b)" % _LATEX_CMDS, r"\\\\", text)   # known LaTeX commands
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)          # other invalid escapes


def extract_json(text: str) -> Any:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    decoder = json.JSONDecoder()
    for candidate in (_repair_escapes(text), text):   # repaired first: avoids silent corruption
        starts = [i for i in (candidate.find("{"), candidate.find("[")) if i >= 0]
        if not starts:
            continue
        try:
            obj, _ = decoder.raw_decode(candidate[min(starts):])
            return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("Could not parse JSON from model response")


def inline_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve $ref/$defs so any provider can consume the schema."""
    defs = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].split("/")[-1]])
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(x) for x in node]
        return node

    return resolve(schema)


# ============================================================
# PROVIDERS
# ============================================================

T = TypeVar("T", bound=BaseModel)


class Truncated(Exception):
    """Output hit the token limit; retrying with the same limit is pointless."""


class QuotaExhausted(Exception):
    """The daily API quota is used up; retrying today is pointless."""


QUOTA_HIT = threading.Event()   # set when a daily quota is exhausted; generate_paper stops


def with_retry(fn, tries: int = 4, base: float = 2.0):
    for attempt in range(tries):
        try:
            return fn()
        except (Truncated, QuotaExhausted):
            raise
        except Exception as exc:
            msg = str(exc)
            rate_limited = ("429" in msg or "RESOURCE_EXHAUSTED" in msg
                            or "rate limit" in msg.lower())
            if rate_limited and "perday" in msg.lower().replace(" ", ""):
                raise QuotaExhausted(msg[:200]) from exc
            if attempt == tries - 1:
                raise
            wait = 20.0 * (attempt + 1) if rate_limited else base ** attempt
            log.warning("API error (%s); retrying in %.0fs", type(exc).__name__, wait)
            time.sleep(wait)


KEEPALIVE_SECONDS = float(os.getenv("KEEPALIVE_SECONDS", "240"))
KEEPALIVE_GRACE = float(os.getenv("KEEPALIVE_GRACE", "900"))


def _keepalive(stop: threading.Event) -> None:
    """Free Render instances sleep after 15 min without web traffic, which would kill a
    running job. While a job runs (and for a grace period after, so you can come back and
    download) request our own public /health page; that counts as traffic."""
    base = os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")
    if not base:
        return
    import urllib.request
    deadline = None
    while True:
        if stop.is_set():
            time.sleep(KEEPALIVE_SECONDS)     # grace period: normal spacing, not a busy loop
        else:
            stop.wait(KEEPALIVE_SECONDS)
        try:
            urllib.request.urlopen(base + "/health", timeout=20).read()
        except Exception as exc:
            log.warning("keepalive ping failed: %s", type(exc).__name__)
        if stop.is_set():
            if deadline is None:
                deadline = time.monotonic() + KEEPALIVE_GRACE
            elif time.monotonic() > deadline:
                return


class RateLimiter:
    """Spaces calls at least 60/rpm seconds apart (thread-safe)."""

    def __init__(self, rpm: float):
        self.interval = 60.0 / max(rpm, 0.1)
        self._lock = threading.Lock()
        self._next_ok = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_ok)
            self._next_ok = start + self.interval
        if start > now:
            time.sleep(start - now)


_LIMITERS: Dict[str, RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def limiter_for(key: str, rpm: float) -> RateLimiter:
    with _LIMITERS_LOCK:
        return _LIMITERS.setdefault(key, RateLimiter(rpm))


class ClaudeProvider:
    name = "claude"

    def __init__(self):
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)

    def structured(self, system: str, prompt: str, model_cls: Type[T],
                   max_tokens: int = 12000) -> T:
        tool_name = "submit_" + model_cls.__name__.lower()
        schema = inline_refs(model_cls.model_json_schema())

        def call():
            resp = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                tools=[{
                    "name": tool_name,
                    "description": "Submit the result in the required structure.",
                    "input_schema": schema,
                }],
                tool_choice={"type": "tool", "name": tool_name},
            )
            if resp.stop_reason == "max_tokens":
                raise Truncated("Claude output truncated")
            for block in resp.content:
                if getattr(block, "type", None) == "tool_use":
                    return block.input
            raise RuntimeError("Claude returned no tool_use block")

        return model_cls.model_validate(with_retry(call))


_MODEL_LOCK = threading.Lock()
_DEAD_MODELS: set = set()                  # models that answered "not found / not allowed"
_EXHAUSTED_UNTIL: Dict[str, float] = {}    # models whose free daily quota is used up


def _model_usable(model: str) -> bool:
    with _MODEL_LOCK:
        return model not in _DEAD_MODELS and time.monotonic() >= _EXHAUSTED_UNTIL.get(model, 0.0)


def _mark_dead(model: str) -> None:
    with _MODEL_LOCK:
        _DEAD_MODELS.add(model)


def _mark_exhausted(model: str, hours: float = 3.0) -> None:
    with _MODEL_LOCK:
        _EXHAUSTED_UNTIL[model] = time.monotonic() + hours * 3600


class GeminiProvider:
    """One Gemini model with an optional backup model. Each model has its own rate limit,
    so using two models roughly doubles free-tier throughput; if one is overloaded or
    out of daily quota, calls fall over to the other automatically."""

    def __init__(self, model: Optional[str] = None, name: str = "gemini",
                 fallback_model: Optional[str] = None):
        self.model = model or GEMINI_MODEL
        self.name = name
        self.fallback_model = fallback_model if fallback_model and fallback_model != self.model else None
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def _call(self, model: str, system: str, prompt: str, model_cls: Type[T]) -> T:
        limiter = limiter_for("gemini:" + model, GEMINI_RPM)

        def call():
            limiter.wait()                           # stay under this model's free-tier RPM
            resp = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_schema=model_cls,
                ),
            )
            if not resp.text:
                raise RuntimeError("Gemini returned empty text (blocked or failed)")
            return extract_json(resp.text)

        return model_cls.model_validate(with_retry(call))

    def structured(self, system: str, prompt: str, model_cls: Type[T],
                   max_tokens: int = 12000) -> T:
        candidates = [m for m in (self.model, self.fallback_model) if m]
        usable = [m for m in candidates if _model_usable(m)]
        if not usable:
            if any(m in _DEAD_MODELS for m in candidates) and not any(
                    time.monotonic() < _EXHAUSTED_UNTIL.get(m, 0.0) for m in candidates):
                raise RuntimeError("No usable Gemini model is configured.")
            raise QuotaExhausted("Daily free quota used up on all configured models.")
        last: Optional[Exception] = None
        for i, model in enumerate(usable):
            try:
                return self._call(model, system, prompt, model_cls)
            except Truncated:
                raise
            except QuotaExhausted as exc:
                _mark_exhausted(model)
                log.warning("daily quota used up on %s", model)
                last = exc
            except Exception as exc:
                last = exc
                if i == len(usable) - 1:
                    raise
                log.warning("%s failed (%s); trying backup model %s",
                            model, type(exc).__name__, usable[i + 1])
        raise last if last else RuntimeError("no model call was made")


# ============================================================
# DATABASE (full question JSON)
# ============================================================

class QuestionDatabase:
    """SQLite for local use; PostgreSQL when DATABASE_URL is set (deployment)."""

    JOB_FIELDS = ("status", "progress", "result", "error")

    def __init__(self, filename: str = DB_FILE):
        self._lock = threading.Lock()
        self.connection = None
        if not USE_POSTGRES:
            self.connection = sqlite3.connect(filename, check_same_thread=False)
        serial = "SERIAL PRIMARY KEY" if USE_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"
        self._run("""
        CREATE TABLE IF NOT EXISTS question_bank (
            id TEXT PRIMARY KEY,
            exam TEXT,
            chapter TEXT,
            question_type TEXT,
            question TEXT,
            difficulty REAL,
            payload TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        self._run(f"""
        CREATE TABLE IF NOT EXISTS attempt_log (
            id {serial},
            question_id TEXT,
            student_answer TEXT,
            correct INTEGER,
            analysis TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
        self._run("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            kind TEXT,
            status TEXT,
            request TEXT,
            progress TEXT,
            result TEXT,
            error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")

    def _run(self, sql: str, params: tuple = (), fetch: Optional[str] = None):
        if USE_POSTGRES:
            import psycopg
            with psycopg.connect(DATABASE_URL) as conn:
                cur = conn.execute(sql.replace("?", "%s"), params)
                if fetch == "all":
                    return cur.fetchall()
                if fetch == "one":
                    return cur.fetchone()
                return None
        with self._lock:
            cur = self.connection.execute(sql, params)
            out = cur.fetchall() if fetch == "all" else (cur.fetchone() if fetch == "one" else None)
            self.connection.commit()
            return out

    # ---- questions ----

    def save_question(self, q: Question) -> None:
        self._run(
            "INSERT INTO question_bank "
            "(id, exam, chapter, question_type, question, difficulty, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (id) DO UPDATE SET exam = excluded.exam, chapter = excluded.chapter, "
            "question_type = excluded.question_type, question = excluded.question, "
            "difficulty = excluded.difficulty, payload = excluded.payload",
            (q.id, q.exam, q.chapter, q.question_type, q.question, q.difficulty,
             q.model_dump_json()),
        )

    def get_question(self, id_or_prefix: str) -> Optional[Question]:
        row = self._run("SELECT payload FROM question_bank WHERE id LIKE ? LIMIT 1",
                        (id_or_prefix + "%",), fetch="one")
        return Question.model_validate_json(row[0]) if row else None

    def recent(self, n: int = 10) -> List[Tuple[str, str]]:
        rows = self._run("SELECT id, question FROM question_bank "
                         "ORDER BY created_at DESC LIMIT ?", (n,), fetch="all")
        return [(r[0], r[1]) for r in rows]

    def all_question_texts(self) -> List[str]:
        return [r[0] for r in self._run("SELECT question FROM question_bank", fetch="all")]

    def record_attempt(self, question_id: str, student_answer: str,
                       correct: bool, analysis: Optional[ErrorAnalysis] = None) -> None:
        self._run(
            "INSERT INTO attempt_log (question_id, student_answer, correct, analysis) "
            "VALUES (?, ?, ?, ?)",
            (question_id, student_answer, int(correct),
             analysis.model_dump_json() if analysis else None),
        )

    # ---- background jobs (used by server.py) ----

    def create_job(self, job_id: str, kind: str, request_json: str) -> None:
        self._run("INSERT INTO jobs (id, kind, status, request) VALUES (?, ?, 'queued', ?)",
                  (job_id, kind, request_json))

    def update_job(self, job_id: str, **fields: Any) -> None:
        cols = [k for k in fields if k in self.JOB_FIELDS]
        if not cols:
            return
        sets = ", ".join(f"{k} = ?" for k in cols) + ", updated_at = CURRENT_TIMESTAMP"
        self._run(f"UPDATE jobs SET {sets} WHERE id = ?",
                  tuple(fields[k] for k in cols) + (job_id,))

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        row = self._run("SELECT id, kind, status, request, progress, result, error, "
                        "created_at, updated_at FROM jobs WHERE id = ?", (job_id,), fetch="one")
        if not row:
            return None
        return {
            "id": row[0], "kind": row[1], "status": row[2],
            "request": json.loads(row[3]) if row[3] else None,
            "progress": json.loads(row[4]) if row[4] else None,
            "result": json.loads(row[5]) if row[5] else None,
            "error": row[6], "created_at": str(row[7]), "updated_at": str(row[8]),
        }

    def fail_stale_jobs(self) -> None:
        """Jobs left queued/running by a previous process can never finish."""
        self._run("UPDATE jobs SET status = 'failed', error = 'Interrupted by a server restart', "
                  "updated_at = CURRENT_TIMESTAMP WHERE status IN ('queued', 'running')")


# ============================================================
# ORIGINALITY (against your own bank only)
# ============================================================

def fingerprint(text: str) -> str:
    normalized = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", text.lower()))
    return hashlib.sha256(normalized.encode()).hexdigest()


class NoveltyIndex:

    def __init__(self, db: QuestionDatabase, threshold: float = NOVELTY_THRESHOLD):
        self.known: List[str] = db.all_question_texts()
        self.hashes = {fingerprint(t) for t in self.known}
        self.threshold = threshold

    def add(self, text: str) -> None:
        self.known.append(text)
        self.hashes.add(fingerprint(text))

    def is_novel(self, text: str, extra: List[str] = ()) -> Tuple[bool, float]:
        if fingerprint(text) in self.hashes:
            return False, 1.0
        corpus = self.known + list(extra)
        if not corpus:
            return True, 0.0
        try:
            vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            matrix = vec.fit_transform(corpus + [text])   # fit on the whole corpus
            best = float(cosine_similarity(matrix[-1], matrix[:-1])[0].max())
        except ValueError:                                # empty vocabulary
            return True, 0.0
        return best < self.threshold, best


# ============================================================
# ANSWER COMPARISON / NUMERIC UTILITIES
# ============================================================

_SCI = re.compile(
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    r"(?:\s*(?:[x×*]|\\times)\s*10\s*\^?\s*\{?\s*([-+]?\d+)\s*\}?)?"
)
_FRACTION = re.compile(r"\s*(-?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*")
_SAFE_EXPR = re.compile(r"^[0-9a-zA-Z_+\-*/^(). ]+$")


def parse_number(s: str) -> Optional[float]:
    s = str(s).replace("−", "-").replace(",", "")
    frac = _FRACTION.fullmatch(s)
    if frac:
        return float(frac.group(1)) / float(frac.group(2))
    m = _SCI.search(s)
    if not m:
        return None
    val = float(m.group(1))
    if m.group(2):
        val *= 10 ** int(m.group(2))
    return val


def numbers_close(a: float, b: float, rel: float = 0.02) -> bool:
    if abs(a) < 1e-12 and abs(b) < 1e-12:
        return True
    return abs(a - b) <= rel * max(abs(a), abs(b))


def safe_sympy(expr: str):
    """Only parse plain algebraic strings; sympify uses eval internally."""
    if not _SAFE_EXPR.match(expr) or "__" in expr:
        return None
    try:
        return sp.sympify(expr.replace("^", "**"))
    except Exception:
        return None


def symbolic_equal(a: str, b: str) -> bool:
    x, y = safe_sympy(a), safe_sympy(b)
    if x is None or y is None:
        return False
    try:
        return bool(sp.simplify(x - y) == 0)
    except Exception:
        return False


def norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9.+\-]", "", s.lower())


def answers_match(q: Question, solved: List[str]) -> bool:
    if q.question_type == "fill_blank":
        if not solved or not q.correct_answer:
            return False
        a, b = q.correct_answer[0], solved[0]
        na, nb = parse_number(a), parse_number(b)
        if na is not None and nb is not None:
            return numbers_close(na, nb)
        return symbolic_equal(a, b) or norm_text(a) == norm_text(b)
    got = {x for s in solved for x in re.findall(r"\b[A-D]\b", str(s).upper())}
    return bool(got) and got == set(q.correct_answer)


def run_code_check(q: Question) -> Tuple[Optional[bool], str]:
    """Execute the generator's verification snippet. None = skipped."""
    if not ENABLE_CODE_CHECK or not q.verification_code.strip() or not q.numeric_answer.strip():
        return None, "skipped"
    claimed = parse_number(q.numeric_answer)
    if claimed is None:
        return None, "numeric_answer unparsable"
    if q.question_type == "fill_blank" and q.correct_answer:
        key_val = parse_number(q.correct_answer[0])
        if key_val is not None and not numbers_close(key_val, claimed, 0.01):
            return False, f"numeric_answer {claimed} disagrees with key {key_val}"
    script = q.verification_code + "\nprint('__RESULT__', float(result))\n"
    try:
        proc = subprocess.run([sys.executable, "-I", "-c", script],
                              capture_output=True, text=True, timeout=CODE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return False, "verification code timed out"
    if proc.returncode != 0:
        return False, "verification code error: " + proc.stderr.strip()[-150:]
    m = re.search(r"__RESULT__\s+(\S+)", proc.stdout)
    if not m:
        return False, "verification code printed no result"
    val = float(m.group(1))
    if not numbers_close(val, claimed, 0.01):
        return False, f"code computed {val}, claimed {claimed}"
    return True, f"code computed {val}"


# ============================================================
# NORMALIZATION + STRUCTURE CHECKS
# ============================================================

_OPT_PREFIX = re.compile(r"^\s*(?:\([A-Da-d]\)|[A-Da-d][\.\)])\s+")


def build_question(gq: GenQuestion, request: PaperRequest,
                   qtype: str, generator: str) -> Question:
    """Code, not the model, owns id/exam/chapter/type/option format/answer format."""
    if qtype == "assertion_reason":
        options = list(AR_OPTIONS)
    elif qtype == "fill_blank":
        options = []
    else:
        options = [_OPT_PREFIX.sub("", str(o)).strip() for o in gq.options]

    if qtype == "fill_blank":
        answer = [str(a).strip() for a in gq.correct_answer if str(a).strip()][:1]
    else:
        answer: List[str] = []
        for a in gq.correct_answer:
            s = str(a).strip().upper()
            letters = [s] if re.fullmatch(r"[A-D]", s) else (
                list(s) if re.fullmatch(r"[A-D]{2,4}", s) else re.findall(r"\b[A-D]\b", s))
            for l in letters:
                if l not in answer:
                    answer.append(l)

    return Question(
        exam=request.exam,
        chapter=request.chapter,
        subtopics=gq.subtopics,
        question_type=qtype,
        question=gq.question.strip(),
        options=options,
        correct_answer=answer,
        solution=gq.solution.strip(),
        concepts_tested=gq.concepts_tested,
        misconceptions_tested=gq.misconceptions_tested,
        difficulty=min(10.0, max(1.0, float(gq.difficulty))),
        estimated_time_seconds=max(20, int(gq.estimated_time_seconds)),
        numeric_answer=gq.numeric_answer.strip(),
        verification_code=gq.verification_code.strip(),
        generator=generator,
    )


def structure_check(q: Question) -> Tuple[bool, str]:
    if len(q.question) < 30:
        return False, "question too short"
    if not q.correct_answer:
        return False, "no parsable answer"
    if not q.solution:
        return False, "no solution"
    t = q.question_type
    if t in ("mcq", "multiple_correct", "match_column"):
        if len(q.options) != 4 or len({o.lower() for o in q.options}) != 4:
            return False, "need 4 distinct options"
        if not set(q.correct_answer) <= set("ABCD"):
            return False, "answer letters not in options"
    if t in ("mcq", "match_column", "assertion_reason") and len(q.correct_answer) != 1:
        return False, "exactly one answer letter required"
    if t == "assertion_reason":
        low = q.question.lower()
        if "assertion" not in low or "reason" not in low:
            return False, "assertion/reason statements missing"
        if not set(q.correct_answer) <= set("ABCD"):
            return False, "bad answer letter"
    if t == "match_column" and "column" not in q.question.lower():
        return False, "columns missing"
    return True, "ok"


# ============================================================
# QUOTAS (types strict, difficulty bands soft)
# ============================================================

def apportion(weights: Dict[str, float], total: int) -> Dict[str, int]:
    s = sum(weights.values())
    raw = {k: total * v / s for k, v in weights.items()}
    base = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - base[k], reverse=True)[: total - sum(base.values())]:
        base[k] += 1
    return base


def band_of(d: float) -> str:
    return "easy" if d <= 3.5 else ("medium" if d < 7 else "hard")


class Quota:

    def __init__(self, request: PaperRequest, blueprint: Blueprint):
        weights: Dict[str, float] = {}
        for item in blueprint.recommended_distribution:
            k, v = item.name, item.count
            k = TYPE_ALIASES.get(norm_key(k), norm_key(k))
            if k in request.question_types and v > 0:
                weights[k] = weights.get(k, 0) + v
        if not set(request.question_types) <= set(weights):
            weights = {t: 1.0 for t in request.question_types}
        self.type_target = apportion(weights, request.count)

        if request.difficulty in ("easy", "medium", "hard"):
            mix = {request.difficulty: 1.0}
        else:
            mix = BAND_MIX.get(request.exam, BAND_MIX["jee_main"])
        self.band_target = apportion(mix, request.count)
        self.type_have: Counter = Counter()
        self.band_have: Counter = Counter()

    def type_missing(self) -> Dict[str, int]:
        return {t: n - self.type_have[t] for t, n in self.type_target.items()
                if n - self.type_have[t] > 0}

    def band_missing(self) -> Dict[str, int]:
        return {b: max(0, n - self.band_have[b]) for b, n in self.band_target.items()}

    def type_full(self, t: str) -> bool:
        return self.type_have[t] >= self.type_target.get(t, 0)

    def band_full(self, b: str) -> bool:
        return self.band_have[b] >= self.band_target.get(b, 0)

    def add(self, q: Question) -> None:
        self.type_have[q.question_type] += 1
        self.band_have[band_of(q.difficulty)] += 1


# ============================================================
# PLANNER
# ============================================================

class Planner:

    def __init__(self, claude: ClaudeProvider):
        self.claude = claude

    def create(self, request: PaperRequest) -> Blueprint:
        prompt = f"""
Create a rigorous blueprint for a Physics paper. Do NOT write questions.

EXAM: {request.exam}
CHAPTER: {request.chapter}
QUESTION COUNT: {request.count}
QUESTION TYPES: {request.question_types}
DIFFICULTY: {request.difficulty}

Determine the major concepts, subtopics, prerequisites, common misconceptions,
important problem archetypes, a question-type distribution (a list of
{{name, count}} items; each name must be one of {request.question_types}) and a
difficulty distribution (a list of {{name, count}} items with names easy, medium, hard).
"""
        return self.claude.structured(MASTER_SYSTEM, prompt, Blueprint, max_tokens=5000)


# ============================================================
# MASTER AGENT
# ============================================================

def render_for_solver(q: Question) -> str:
    lines = [f"EXAM: {q.exam}", f"CHAPTER: {q.chapter}",
             f"TYPE: {q.question_type}", "", q.question]
    for i, o in enumerate(q.options):
        lines.append(f"({chr(65 + i)}) {o}")
    if q.question_type == "fill_blank":
        fmt = "answer = a list with ONE string: the final value with units."
    elif q.question_type == "multiple_correct":
        fmt = "answer = list of ALL correct option letters."
    else:
        fmt = "answer = list with the ONE correct option letter."
    return "\n".join(lines) + f"\n\nRespond with: well_posed, {fmt} Include a brief working_summary."


def render_for_audit(q: Question) -> str:
    return f"""
Audit this Physics question. Be extremely strict.

EXAM: {q.exam}
CHAPTER: {q.chapter}
TYPE: {q.question_type}

QUESTION:
{q.question}

OPTIONS:
{json.dumps(q.options, indent=2, ensure_ascii=False)}

CLAIMED ANSWER: {json.dumps(q.correct_answer)}

SOLUTION:
{q.solution}

Also set resembles_known_problem=true if this closely matches a well-known
textbook, coaching or past-paper problem. difficulty_estimate is 1-10.
"""


class PhysicsMasterAgent:

    def __init__(self):
        require_keys()
        if ANTHROPIC_API_KEY:                      # best quality: two different AI vendors
            self.mode = "claude+gemini"
            self.prov_a = ClaudeProvider()
            self.prov_b = GeminiProvider(GEMINI_MODEL, "gemini")
        else:                                      # free mode: Gemini checks Gemini
            self.mode = "gemini-only"
            m2 = GEMINI_MODEL_2 or None
            self.prov_a = GeminiProvider(GEMINI_MODEL, "gemini-a", fallback_model=m2)
            self.prov_b = GeminiProvider(m2 or GEMINI_MODEL, "gemini-b",
                                         fallback_model=GEMINI_MODEL if m2 else None)
        self.db = QuestionDatabase()
        self.novelty = NoveltyIndex(self.db)
        self.planner = Planner(self.prov_a)

    def other(self, name: str):
        """The provider that did NOT write a question; it solves and audits it."""
        return self.prov_b if name == self.prov_a.name else self.prov_a

    def preflight(self) -> None:
        if isinstance(self.prov_a, GeminiProvider) and isinstance(self.prov_b, GeminiProvider):
            return self._preflight_gemini()
        seen = set()
        for p in (self.prov_a, self.prov_b):
            key = (type(p).__name__, getattr(p, "model", p.name))
            if key in seen:
                continue
            seen.add(key)
            try:
                p.structured("Reply via the structured output only.",
                             "Set ok to true.", Ping, max_tokens=300)
            except QuotaExhausted as exc:
                raise RuntimeError("The daily free quota is already used up. "
                                   "Try again tomorrow.") from exc
            except Exception as exc:
                var = "GEMINI_MODEL" if p.name.startswith("gemini") else "CLAUDE_MODEL"
                raise RuntimeError(
                    f"{p.name} preflight failed ({type(exc).__name__}: {exc}). "
                    f"Check the API key and the {var} name.") from exc

    def _preflight_gemini(self) -> None:
        """Test each configured Gemini model; drop unusable ones, fail only if none work."""
        models: List[str] = []
        for p in (self.prov_a, self.prov_b):
            for m in (p.model, p.fallback_model):
                if m and m not in models:
                    models.append(m)
        working, errors = [], []
        for m in models:
            try:
                self.prov_a._call(m, "Reply via the structured output only.",
                                  "Set ok to true.", Ping)
                working.append(m)
            except QuotaExhausted:
                _mark_exhausted(m)
                errors.append(f"{m}: daily free quota already used up")
            except Exception as exc:
                _mark_dead(m)
                errors.append(f"{m}: {type(exc).__name__}: {str(exc)[:160]}")
        if not working:
            raise RuntimeError("No Gemini model is usable. " + " | ".join(errors) +
                               " Check the API key and GEMINI_MODEL.")
        for e in errors:
            log.warning("model dropped at preflight: %s", e)
        print(f"Models in use: {working}" + (f" (dropped: {len(errors)})" if errors else ""))

    # ---------------- verification ----------------

    def verify(self, q: Question) -> Tuple[bool, str]:
        """Cross-model: the model that did NOT write q solves it blind, then audits it."""
        checker = self.other(q.generator)

        code_ok, code_msg = run_code_check(q)
        if code_ok is False:
            return False, f"code check: {code_msg}"

        try:
            solved = checker.structured(SOLVER_SYSTEM, render_for_solver(q),
                                        SolveResult, max_tokens=8000)
        except QuotaExhausted:
            QUOTA_HIT.set()
            return False, "ERROR daily quota"
        except Exception as exc:
            log.exception("solver error")
            return False, f"ERROR solver {type(exc).__name__}"
        if not solved.well_posed:
            return False, "blind solver: ill-posed/ambiguous"
        if not answers_match(q, solved.answer):
            return False, f"blind-solve mismatch (key {q.correct_answer}, solver {solved.answer})"

        try:
            audit = checker.structured(CRITIC_SYSTEM, render_for_audit(q),
                                       AuditResult, max_tokens=4000)
        except QuotaExhausted:
            QUOTA_HIT.set()
            return False, "ERROR daily quota"
        except Exception as exc:
            log.exception("audit error")
            return False, f"ERROR audit {type(exc).__name__}"

        checks = {
            "physics": audit.physics_correct,
            "math": audit.mathematics_correct,
            "dimensions": audit.dimensions_ok,
            "answer": audit.answer_correct,
            "options": audit.options_correct,
            "ambiguity": audit.unambiguous,
            "syllabus": audit.syllabus_appropriate,
        }
        if audit.resembles_known_problem:      # advisory only: nearly all physics resembles a textbook
            log.info("NOTE resembles a known problem (kept): %s", q.question[:80].replace("\n", " "))
        failed = [k for k, v in checks.items() if not v]
        if failed:
            return False, f"audit failed {failed}: {audit.reason[:160]}"

        q.verification = Verification(
            physics=True, mathematics=True, dimensions=True,
            options=True, originality=True, ambiguity=True)
        q.verified_by = checker.name
        q.code_checked = code_ok is True
        q.difficulty = round(min(10.0, max(1.0, (q.difficulty + audit.difficulty_estimate) / 2)), 1)
        return True, "ok"

    # ---------------- generation ----------------

    @staticmethod
    def coverage_hint(blueprint: Blueprint, accepted: List[Question], k: int = 6) -> List[str]:
        counts = []
        for concept in blueprint.concepts:
            c = sum(1 for q in accepted
                    if concept.lower() in " ".join(q.concepts_tested + q.subtopics).lower())
            counts.append((c, concept))
        return [c for _, c in sorted(counts)[:k]]

    def generate_batch(self, provider, request: PaperRequest, blueprint: Blueprint,
                       qtype: str, n: int, quota: Quota, accepted: List[Question]) -> List[Question]:
        band_need = {b: v for b, v in quota.band_missing().items() if v > 0}
        band_mix = apportion(band_need, n) if band_need else apportion(quota.band_target, n)
        recent = [q.question[:140].replace("\n", " ") for q in accepted[-15:]]

        prompt = f"""
Generate {n} ORIGINAL Physics questions of ONE type.

EXAM: {request.exam}
CHAPTER: {request.chapter}
TYPE SPEC: {TYPE_SPECS[qtype]}

DIFFICULTY MIX to produce (count per band; easy ~1-3.5, medium ~4-6.5, hard ~7-10):
{json.dumps(band_mix)}

BLUEPRINT:
{blueprint.model_dump_json(indent=2)}

UNDER-COVERED CONCEPTS (prefer these): {self.coverage_hint(blueprint, accepted)}

ALREADY IN THE PAPER (do NOT reuse these setups):
{json.dumps(recent, ensure_ascii=False)}

Rules:
1. Do not copy known questions or just renumber familiar ones.
2. Vary physical situations and reasoning structures across the batch.
3. Compute and verify every answer before writing the key.
4. Each question must have a unique, unambiguous answer.
5. numeric_answer: for numerical answers, the plain decimal value in the SAME
   unit as the stated answer (e.g. "3.2"); otherwise "".
6. verification_code: when numeric_answer is set, a short Python snippet
   (only `math` and `sympy`, no input/output/files) that computes the answer
   from the givens and stores it in a variable named `result`, in the same
   unit as numeric_answer; otherwise "".
"""
        batch = provider.structured(GENERATOR_SYSTEM, prompt, GenBatch, max_tokens=14000)
        out = []
        for gq in batch.questions:
            try:
                out.append(build_question(gq, request, qtype, provider.name))
            except Exception as exc:
                log.warning("build_question failed: %s", exc)
        return out

    # ---------------- paper ----------------

    def generate_paper(self, request: PaperRequest, progress=None) -> List[Question]:
        stop = threading.Event()
        threading.Thread(target=_keepalive, args=(stop,), daemon=True).start()
        try:
            return self._generate_paper(request, progress)
        finally:
            stop.set()

    def _generate_paper(self, request: PaperRequest, progress=None) -> List[Question]:
        def report(**info):
            if progress:
                try:
                    progress(info)
                except Exception:
                    log.exception("progress callback failed")

        print("\n" + "=" * 70)
        print("PHYSICS MASTER AGENT")
        print("=" * 70)
        print(f"Exam: {request.exam} | Chapter: {request.chapter} | Target: {request.count}")
        QUOTA_HIT.clear()
        print(f"Mode: {self.mode} (writer/checker: {self.prov_a.name} <-> {self.prov_b.name})")

        print("\n[1/4] Preflight (both models)...")
        self.preflight()

        print("[2/4] Planning...")
        report(phase="planning", accepted=0, target=request.count)
        blueprint = self.planner.create(request)
        quota = Quota(request, blueprint)
        print(f"Concepts: {len(blueprint.concepts)} | type targets: {quota.type_target} "
              f"| difficulty targets: {quota.band_target}")

        print("[3/4] Generating and verifying...")
        accepted: List[Question] = []
        reserve: List[Question] = []
        rejections: Counter = Counter()

        def try_accept(q: Question, relax: bool) -> bool:
            if quota.type_full(q.question_type):
                rejections["type quota full"] += 1
                return False
            if not relax and quota.band_full(band_of(q.difficulty)):
                reserve.append(q)                      # verified, keep for later
                return False
            accepted.append(q)
            quota.add(q)
            self.novelty.add(q.question)
            self.db.save_question(q)
            return True

        for rnd in range(1, MAX_ROUNDS + 1):
            if QUOTA_HIT.is_set():
                print("Daily API quota reached; stopping with what is verified so far.")
                break
            missing = quota.type_missing()
            if not missing:
                break
            relax = rnd > RELAX_BANDS_AFTER

            if relax and reserve:                      # use verified spares first
                spare, reserve[:] = list(reserve), []
                for q in spare:
                    try_accept(q, True)
                missing = quota.type_missing()
                if not missing:
                    break

            print(f"\nRound {rnd} | accepted {len(accepted)}/{request.count} | missing {missing}")
            report(phase="generating", round=rnd, accepted=len(accepted),
                   target=request.count, missing=missing)

            # 1) generate: one call per missing type, alternating providers, in parallel
            jobs = []
            for i, (qtype, m) in enumerate(missing.items()):
                n = min(MAX_PER_CALL, max(2, math.ceil(m * OVERSHOOT)))
                provider = (self.prov_a, self.prov_b)[(rnd + i) % 2]
                jobs.append((provider, qtype, n))

            def run_job(job):
                provider, qtype, n = job
                try:
                    return self.generate_batch(provider, request, blueprint, qtype,
                                               n, quota, accepted)
                except QuotaExhausted:
                    QUOTA_HIT.set()
                    return []
                except Exception as exc:
                    log.exception("generation failed (%s/%s)", provider.name, qtype)
                    print(f"  generation error [{provider.name}/{qtype}]: {type(exc).__name__}")
                    return []

            with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                batches = list(pool.map(run_job, jobs))
            candidates = [q for b in batches for q in b]
            print(f"  candidates: {len(candidates)}")

            # 2) cheap filters (structure, novelty incl. within-batch)
            survivors: List[Question] = []
            for q in candidates:
                ok, why = structure_check(q)
                if not ok:
                    rejections["structure: " + why] += 1
                    continue
                novel, sim = self.novelty.is_novel(q.question, [s.question for s in survivors])
                if not novel:
                    rejections["duplicate/similar"] += 1
                    continue
                survivors.append(q)

            # 3) expensive cross-model verification in parallel
            report(phase="verifying", round=rnd, accepted=len(accepted),
                   target=request.count, candidates=len(survivors))
            with ThreadPoolExecutor(max_workers=AUDIT_WORKERS) as pool:
                results = list(pool.map(self.verify, survivors))

            # 4) accept sequentially (quotas, DB writes)
            for q, (ok, why) in zip(survivors, results):
                if not ok:
                    rejections[why.split(":")[0].split(" (")[0]] += 1
                    log.info("REJECT %s | %s", why, q.question[:80].replace("\n", " "))
                    continue
                try_accept(q, relax)
            report(phase="generating", round=rnd, accepted=len(accepted),
                   target=request.count, missing=quota.type_missing())

        if reserve:                                    # last chance for spares
            spare, reserve[:] = list(reserve), []
            for q in spare:
                try_accept(q, True)

        print("\n[4/4] Summary")
        print(f"Accepted: {len(accepted)}/{request.count}")
        print(f"By type: {dict(quota.type_have)} (target {quota.type_target})")
        print(f"By difficulty: {dict(quota.band_have)} (target {quota.band_target})")
        if rejections:
            print("Rejections:", dict(rejections.most_common()))
        if len(accepted) < request.count:
            if QUOTA_HIT.is_set():
                print(f"NOTE: the daily API limit was reached after {len(accepted)} verified "
                      f"questions. Returning what is verified; try again after the quota resets.")
            else:
                print(f"WARNING: only {len(accepted)} verified questions after {MAX_ROUNDS} rounds; "
                      f"returning the partial paper. See physics_master.log for details.")
        report(phase="finished", accepted=len(accepted), target=request.count,
               quota_exhausted=QUOTA_HIT.is_set(),
               by_type=dict(quota.type_have), by_difficulty=dict(quota.band_have),
               rejections=dict(rejections.most_common()))
        return accepted


# ============================================================
# REMEDIAL ENGINE
# ============================================================

class RemedialEngine:

    def __init__(self, agent: PhysicsMasterAgent):
        self.agent = agent

    def analyze(self, question: Question, student_answer: str) -> ErrorAnalysis:
        prompt = f"""
Analyze this Physics mistake.

QUESTION:
{question.question}

OPTIONS:
{json.dumps(question.options, ensure_ascii=False)}

CORRECT ANSWER: {question.correct_answer}
STUDENT ANSWER: {student_answer}

Find the most likely underlying error. Error types include: conceptual
misunderstanding, formula misuse, sign error, unit error, algebra error,
calculation error, graph interpretation, constraint error, wrong reference
frame, wrong reference point, misreading, multi-concept confusion.
recommended_difficulty is 1-10.
"""
        return self.agent.prov_a.structured(MASTER_SYSTEM, prompt, ErrorAnalysis, max_tokens=3000)

    def generate(self, original: Question, analysis: ErrorAnalysis,
                 attempts: int = 4) -> Optional[Question]:
        request = PaperRequest(exam=original.exam, chapter=original.chapter, count=1,
                               question_types=[original.question_type])
        for i in range(attempts):
            provider = (self.agent.prov_a, self.agent.prov_b)[i % 2]
            prompt = f"""
Create ONE completely NEW remedial Physics question.

QUESTION TYPE SPEC: {TYPE_SPECS[original.question_type]}

ORIGINAL QUESTION (do not paraphrase or reuse its setup):
{original.question}

CONCEPT: {analysis.concept}
SUBTOPIC: {analysis.subtopic}
ERROR TYPE: {analysis.error_type}
MISCONCEPTION: {analysis.likely_misconception}
TARGET DIFFICULTY: {analysis.recommended_difficulty}

Return it as a single-item questions list. Set numeric_answer and
verification_code as "" unless the answer is numerical (then follow the usual
rules: same-unit decimal, and a python snippet setting `result`).
"""
            try:
                batch = provider.structured(REMEDIAL_SYSTEM, prompt, GenBatch, max_tokens=6000)
                if not batch.questions:
                    continue
                q = build_question(batch.questions[0], request,
                                   original.question_type, provider.name)
            except Exception:
                log.exception("remedial generation failed")
                continue
            ok, why = structure_check(q)
            if not ok:
                continue
            novel, _ = self.agent.novelty.is_novel(q.question)
            if not novel:
                continue
            ok, why = self.agent.verify(q)
            if ok:
                self.agent.novelty.add(q.question)
                self.agent.db.save_question(q)
                return q
            log.info("remedial rejected: %s", why)
        return None


# ============================================================
# EXPORT
# ============================================================

def paper_text(questions: List[Question]) -> str:
    out = []
    for i, q in enumerate(questions, start=1):
        out.append(f"\n{'=' * 80}\nQUESTION {i}\n")
        out.append(f"Type: {q.question_type}\nDifficulty: {q.difficulty}/10\n\n")
        out.append(q.question + "\n\n")
        for idx, option in enumerate(q.options):
            out.append(f"({chr(65 + idx)}) {option}\n")
        out.append("\n")
    return "".join(out)


def solutions_text(questions: List[Question]) -> str:
    out = ["ANSWER KEY\n" + "-" * 40 + "\n"]
    for i, q in enumerate(questions, start=1):
        out.append(f"{i}. {', '.join(q.correct_answer)}\n")
    for i, q in enumerate(questions, start=1):
        out.append(f"\n{'=' * 80}\nQUESTION {i}\n\n{q.question}\n\n")
        for idx, option in enumerate(q.options):
            out.append(f"({chr(65 + idx)}) {option}\n")
        out.append(f"\nANSWER: {', '.join(q.correct_answer)}\n")
        out.append(f"CONCEPTS: {', '.join(q.concepts_tested)}\n")
        out.append(f"VERIFIED BY: {q.verified_by} (written by {q.generator})"
                   f"{' + code check' if q.code_checked else ''}\n\n")
        out.append("SOLUTION:\n" + q.solution + "\n")
    return "".join(out)


def save_json(questions: List[Question], filename: str = "physics_paper.json") -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump([q.model_dump() for q in questions], f, indent=2, ensure_ascii=False)


def save_text_paper(questions: List[Question], filename: str = "physics_paper.txt") -> None:
    with open(filename, "w", encoding="utf-8") as f:
        f.write(paper_text(questions))


def save_solutions(questions: List[Question], filename: str = "physics_solutions.txt") -> None:
    with open(filename, "w", encoding="utf-8") as f:
        f.write(solutions_text(questions))


# ============================================================
# INTERACTIVE CLI
# ============================================================

def ask(prompt: str, default: Optional[str] = None) -> str:
    value = input(f"{prompt}" + (f" [{default}]" if default else "") + ": ").strip()
    return value or (default or "")


def interactive_generate(agent: PhysicsMasterAgent) -> None:
    exam = ask("Exam (neet / jee_main / jee_advanced)", "jee_advanced").lower()
    chapter = ask("Chapter", "Rotational Motion")
    try:
        count = min(max(int(ask("Number of questions", "100")), 1), 100)
    except ValueError:
        count = 100
    print("Question types:", ", ".join(ALL_TYPES))
    types = ask("Types (comma separated)", ",".join(ALL_TYPES))
    difficulty = ask("Difficulty (adaptive / easy / medium / hard)", "adaptive").lower()

    request = PaperRequest(
        exam=exam, chapter=chapter, count=count,
        question_types=[x.strip() for x in types.split(",") if x.strip()],
        difficulty=difficulty)

    questions = agent.generate_paper(request)
    if not questions:
        print("No verified questions were produced.")
        return
    save_json(questions)
    save_text_paper(questions)
    save_solutions(questions)
    print("\nFiles created: physics_paper.json, physics_paper.txt, physics_solutions.txt")


def interactive_remedial(agent: PhysicsMasterAgent) -> None:
    recent = agent.db.recent(10)
    if not recent:
        print("Question bank is empty. Generate a paper first.")
        return
    print("\nRecent questions:")
    for qid, text in recent:
        print(f"  {qid[:8]}  {text[:70].replace(chr(10), ' ')}")
    q = agent.db.get_question(ask("Question id (prefix is fine)"))
    if not q:
        print("Not found.")
        return
    student = ask("Student's answer (letters, or value for fill-blank)")
    if q.question_type == "fill_blank":
        correct = answers_match(q, [student])
    else:
        correct = answers_match(q, re.findall(r"[A-Da-d]", student.upper()))
    if correct:
        agent.db.record_attempt(q.id, student, True)
        print("Correct. Nothing to remediate.")
        return
    engine = RemedialEngine(agent)
    print("Incorrect. Analyzing...")
    analysis = engine.analyze(q, student)
    agent.db.record_attempt(q.id, student, False, analysis)
    print(f"\nDiagnosis: {analysis.error_type} | {analysis.likely_misconception}")
    print(analysis.explanation)
    print("\nGenerating a verified remedial question...")
    new_q = engine.generate(q, analysis)
    if not new_q:
        print("Could not produce a verified remedial question. See physics_master.log.")
        return
    print("\n" + new_q.question)
    for i, o in enumerate(new_q.options):
        print(f"({chr(65 + i)}) {o}")
    print(f"\nAnswer: {', '.join(new_q.correct_answer)}\n\nSolution:\n{new_q.solution}")


def interactive() -> None:
    print("\n" + "=" * 70 + "\n        PHYSICS MASTER AGENT\n" + "=" * 70)
    agent = PhysicsMasterAgent()
    while True:
        choice = ask("\nAction (generate / remedial / quit)", "generate").lower()
        if choice.startswith("g"):
            interactive_generate(agent)
        elif choice.startswith("r"):
            interactive_remedial(agent)
        else:
            break


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    try:
        interactive()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as exc:
        log.exception("fatal")
        print("\nFATAL ERROR:", repr(exc))
        print("\nCheck: 1. API keys  2. Model names  3. Internet  4. Installed packages")
        print("Details are in physics_master.log")
