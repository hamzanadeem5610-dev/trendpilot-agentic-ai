"""TrendPilot - FastAPI backend (brief section 15, Option C).

Run:   uvicorn app.main:app --port 8000
Docs:  http://localhost:8000/docs

POST /generate-content   {"topic", "platform", "tone", "output_type"} -> hook, caption, hashtags, reel_script, saved_file ...
POST /improve-content    {"saved_json", "feedback"}                   -> improved version (user feedback loop)
POST /study-plan         {"topic", "days", "level", "hours_per_day"}   -> StudyBuddy pack
POST /study-progress     {"plan_id", "day", "completed", "quiz_score"} -> progress summary
GET  /history            previous runs from memory
GET  /health             Ollama + model status
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import TrendPilotAgent
from .llm import OllamaError

app = FastAPI(title="TrendPilot API", version="1.0",
              description="Local Agentic AI (Ollama + Gemma 3:4B) that turns a topic into a viral content plan.")
_agent: TrendPilotAgent | None = None
_lock = threading.Lock()          # one local GPU -> run one agent at a time


def agent() -> TrendPilotAgent:
    global _agent
    if _agent is None:
        _agent = TrendPilotAgent()
    return _agent


class ContentRequest(BaseModel):
    topic: str = Field(..., examples=["YOLOv8 helmet detection project"])
    platform: str = Field("LinkedIn", examples=["LinkedIn"])
    tone: str = Field("professional and exciting", examples=["professional and exciting"])
    output_type: str = Field("post and reel script", examples=["post and reel script"])


class ContentResponse(BaseModel):
    topic: str
    platform: str
    content_idea: str
    target_audience: str
    hook: str
    caption: str
    hashtags: list[str]
    reel_script: str
    title: str
    thumbnail_text: str
    review_score: float | None
    review_comments: list[str]
    tools_used: list[str]
    saved_file: str
    saved_json: str


class ImproveRequest(BaseModel):
    saved_json: str
    feedback: str = Field(..., examples=["make the hook shorter and more surprising"])


class StudyRequest(BaseModel):
    topic: str = Field(..., examples=["object detection"])
    days: int = 7
    level: str = "Beginner"
    hours_per_day: float = 2.0


class ProgressRequest(BaseModel):
    plan_id: int
    day: int
    completed: bool = True
    quiz_score: float | None = None


def _to_response(res: dict) -> ContentResponse:
    fo = res["final_output"]
    return ContentResponse(topic=fo["topic"], platform=fo["platform"], content_idea=fo["content_idea"],
                           target_audience=fo["target_audience"], hook=fo["hook"], caption=fo["caption"],
                           hashtags=fo["hashtags"], reel_script=fo["reel_script"], title=fo["title"],
                           thumbnail_text=fo["thumbnail_text"], review_score=fo["review_score"],
                           review_comments=[str(c) for c in fo["review_comments"]], tools_used=res["tools_used"],
                           saved_file=fo["saved_file"], saved_json=res["saved_files"].get("json", ""))


@app.get("/health")
def health():
    try:
        return {"status": "ok", "model": agent().llm.model, **agent().llm.health()}
    except OllamaError as e:
        raise HTTPException(503, str(e))


@app.post("/generate-content", response_model=ContentResponse)
def generate_content(req: ContentRequest):
    try:
        with _lock:
            res = agent().run(req.topic, req.platform, req.tone, req.output_type)
    except OllamaError as e:
        raise HTTPException(503, str(e))
    return _to_response(res)


@app.post("/improve-content", response_model=ContentResponse)
def improve_content(req: ImproveRequest):
    p = Path(req.saved_json)
    if not p.exists():
        raise HTTPException(404, f"Saved result not found: {p}")
    previous = json.loads(p.read_text(encoding="utf-8"))
    with _lock:
        res = agent().revise(previous, req.feedback)
    return _to_response(res)


@app.post("/study-plan")
def study_plan(req: StudyRequest):
    with _lock:
        res = agent().run_study(req.topic, req.days, req.level, req.hours_per_day)
    return {**res["final_output"], "plan": res["artifacts"].get("study_plan"), "notes": res["artifacts"].get("notes"),
            "quiz": res["artifacts"].get("quiz"), "tools_used": res["tools_used"]}


@app.post("/study-progress")
def study_progress(req: ProgressRequest):
    return agent().memory.update_progress(req.plan_id, req.day, req.completed, req.quiz_score)


@app.get("/history")
def history(limit: int = 20):
    return agent().memory.history(limit)
