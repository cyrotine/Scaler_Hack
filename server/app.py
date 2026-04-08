"""
InsureLink-v1 — FastAPI Server
===============================
Exposes the InsuranceEnv over HTTP for Hugging Face Space deployment.

Endpoints:
    POST /reset   — Reset the environment for a given task.
    POST /step    — Execute an agent action.
    GET  /state   — Return the full environment state.
    GET  /health  — Liveness probe.
    GET  /        — Root info.

Run:
    uvicorn app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from server.env import InsuranceEnv
from server.models import InsuranceAction
from server.tasks import grade_task

# ── FastAPI app ──────────────────────────────────────────────────────
app = FastAPI(
    title="InsureLink-v1",
    description="Car & Bike Insurance Agent Environment (OpenEnv spec)",
    version="1.0.0",
)

# Single environment instance (stateful, one episode at a time)
_env: Optional[InsuranceEnv] = None


def _get_env() -> InsuranceEnv:
    global _env
    if _env is None:
        _env = InsuranceEnv()
    return _env


# ── Request / Response models ────────────────────────────────────────
class ResetRequest(BaseModel):
    task_id: str = Field(
        default="coverage_check",
        description="One of: coverage_check, policy_update, claim_arbitration",
    )


class StepRequest(BaseModel):
    tool_name: Optional[str] = Field(default=None)
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    message: Optional[str] = Field(default=None)


class GradeRequest(BaseModel):
    task_id: str
    agent_output: str = Field(default="", description="Concatenated agent messages")


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Root endpoint with environment info."""
    return {
        "name": "InsureLink-v1",
        "version": "1.0.0",
        "description": "Car & Bike Insurance Agent Environment",
        "endpoints": ["/reset", "/step", "/state", "/grade", "/health"],
        "tasks": ["coverage_check", "policy_update", "claim_arbitration"],
    }


@app.get("/health")
async def health():
    """Liveness probe for HF Spaces / Docker."""
    return {"status": "ok"}


@app.post("/reset")
async def reset(req: ResetRequest):
    """Reset the environment and start a new episode.

    Body:
        ``{"task_id": "coverage_check"}``
    """
    env = _get_env()
    try:
        result = await env.reset(task_id=req.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }


@app.post("/step")
async def step(req: StepRequest):
    """Execute one agent action.

    Body:
        ``{"tool_name": "get_policy_details", "tool_args": {"policy_id": "P-101"}, "message": "Looking it up!"}``
    """
    env = _get_env()
    action = InsuranceAction(
        tool_name=req.tool_name,
        tool_args=req.tool_args,
        message=req.message,
    )
    try:
        result = await env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }


@app.get("/state")
async def state():
    """Return the full environment state (for debugging / grading)."""
    env = _get_env()
    return env.state()


@app.post("/grade")
async def grade(req: GradeRequest):
    """Run the deterministic grader for the current episode.

    Body:
        ``{"task_id": "coverage_check", "agent_output": "Your deductible is $500..."}``
    """
    env = _get_env()
    env_state = env.state()
    try:
        score = grade_task(
            task_id=req.task_id,
            state=env_state,
            agent_output=req.agent_output,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"task_id": req.task_id, "score": score}


# ── Standalone runner ────────────────────────────────────────────────
def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
