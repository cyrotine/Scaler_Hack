"""
InsureLink-v1 — Typed Models
=============================
Pydantic models for the InsureLink insurance agent environment,
following the OpenEnv specification.

Three core types:
  • InsuranceAction  — what the agent sends to the environment
  • InsuranceObservation — what the environment returns to the agent
  • InsuranceReward — scalar reward + human-readable reason
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Action ───────────────────────────────────────────────────────────
class InsuranceAction(BaseModel):
    """An action produced by the agent each turn.

    Attributes:
        tool_name: The name of the tool to invoke
                   (e.g. ``get_policy_details``, ``update_vehicle``,
                   ``calculate_claim_payout``).
                   Set to ``None`` (or omit) when the agent only wants
                   to send a chat message without calling a tool.
        tool_args: Keyword arguments forwarded to the selected tool.
        message:   A natural-language message from the agent
                   (e.g. a response to the customer).
    """

    tool_name: Optional[str] = Field(
        default=None,
        description="Name of the tool to call, or None for a chat-only turn.",
    )
    tool_args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments passed to the tool.",
    )
    message: Optional[str] = Field(
        default=None,
        description="Natural-language message from the agent.",
    )


# ── Observation ──────────────────────────────────────────────────────
class InsuranceObservation(BaseModel):
    """An observation returned by the environment after each step.

    Attributes:
        chat_history:   The full conversation so far (list of dicts
                        with ``role`` and ``content`` keys).
        tool_result:    The return value of the last tool call,
                        or ``None`` if no tool was invoked.
        available_tools: Names of tools the agent may call next.
    """

    chat_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Conversation turns so far.",
    )
    tool_result: Optional[Any] = Field(
        default=None,
        description="Result returned by the last tool invocation.",
    )
    available_tools: List[str] = Field(
        default_factory=list,
        description="Tool names available for the next action.",
    )


# ── Reward ───────────────────────────────────────────────────────────
class InsuranceReward(BaseModel):
    """Scalar reward signal with an explanation.

    Attributes:
        value:  Numeric reward in ``[0, 1]``.
        reason: Human-readable explanation of how the reward was computed.
    """

    value: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Reward value between 0 and 1.",
    )
    reason: str = Field(
        default="",
        description="Explanation of the reward.",
    )


# ── Step Result (convenience wrapper) ────────────────────────────────
class StepResult(BaseModel):
    """Wrapper returned by ``InsuranceEnv.step()`` and ``reset()``.

    Groups an observation, optional reward, and a ``done`` flag into a
    single object so callers have a uniform interface.
    """

    observation: InsuranceObservation
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)
