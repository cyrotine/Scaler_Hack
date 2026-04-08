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


class ResetRequest(BaseModel):
    # Adding '= None' makes this field optional! 
    # Without it, FastAPI crashes if the judge sends {}
    task_id: Optional[str] = None 

class StepRequest(BaseModel):
    # Ensure action is required, but data can be optional
    action: str
    action_input: Optional[Any] = None

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
async def reset(req: Optional[ResetRequest] = None):
    """Reset the environment and start a new episode.

    Body:
        ``{"task_id": "coverage_check"}``
    """
    env = _get_env()
    
    # Default to coverage_check if no body or task_id provided
    task_id = "coverage_check"
    if req is not None and req.task_id is not None:
        task_id = req.task_id
        
    try:
        result = await env.reset(task_id=task_id)
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
    """Execute one agent action."""
    env = _get_env()
    
    tool_name = None
    tool_args = {}
    message = None
    
    # Map the OpenEnv action/action_input format into InsuranceAction
    if req.action == "message" or req.action == "submit":
        message = str(req.action_input) if req.action_input is not None else ""
    else:
        tool_name = req.action
        if isinstance(req.action_input, dict):
            tool_args = req.action_input
            
    action = InsuranceAction(
        tool_name=tool_name,
        tool_args=tool_args,
        message=message,
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
    # This matches the 'server.app:app' path the script expects
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
