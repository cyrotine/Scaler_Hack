"""
InsureLink-v1 — Environment
=============================
Async environment class following the OpenEnv spec.

Public API
----------
    await env.reset(task_id)   → StepResult
    await env.step(action)     → StepResult
    env.state()                → dict
    await env.close()          → None

World State
-----------
    Alice (alice_99): Policy P-101, Comprehensive, $500 deductible, 2024 Tesla Model 3
    Bob   (bob_77):   Policy P-102, Third-Party,    $0   deductible, 2018 Yamaha R1
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from server.models import (
    InsuranceAction,
    InsuranceObservation,
    InsuranceReward,
    StepResult,
)

# ────────────────────────────────────────────────────────────────────
# In-memory "database"
# ────────────────────────────────────────────────────────────────────
_INITIAL_DB: Dict[str, Any] = {
    "users": {
        "alice_99": {
            "user_id": "alice_99",
            "name": "Alice",
            "policy_id": "P-101",
        },
        "bob_77": {
            "user_id": "bob_77",
            "name": "Bob",
            "policy_id": "P-102",
        },
    },
    "policies": {
        "P-101": {
            "policy_id": "P-101",
            "user_id": "alice_99",
            "coverage_type": "Comprehensive",
            "deductible": 500,
            "vehicle_make": "Tesla",
            "vehicle_model": "Model 3",
            "vehicle_year": 2024,
            "vehicle_vin": "5YJ3E1EA1PF000001",
        },
        "P-102": {
            "policy_id": "P-102",
            "user_id": "bob_77",
            "coverage_type": "Third-Party",
            "deductible": 0,
            "vehicle_make": "Yamaha",
            "vehicle_model": "R1",
            "vehicle_year": 2018,
            "vehicle_vin": "JYARN33E18A000002",
        },
    },
}

# ────────────────────────────────────────────────────────────────────
# Task definitions (mirrors openenv.yaml)
# ────────────────────────────────────────────────────────────────────
_TASKS: Dict[str, Dict[str, Any]] = {
    "coverage_check": {
        "difficulty": "easy",
        "max_steps": 5,
        "system_prompt": (
            "You are an InsureLink customer-service agent. "
            "A customer named Alice (policy P-101) is calling to "
            "find out her deductible. Retrieve the policy and answer clearly."
        ),
        "initial_customer_message": (
            "Hi, I'm Alice (alice_99). My policy ID is P-101. "
            "What is my deductible?"
        ),
        "required_tools": ["get_policy_details"],
        "target_policy_id": "P-101",
    },
    "policy_update": {
        "difficulty": "medium",
        "max_steps": 5,
        "system_prompt": (
            "You are an InsureLink customer-service agent. "
            "A customer named Bob (policy P-102) wants to update "
            "the VIN on his vehicle record. Validate the request "
            "and perform the update."
        ),
        "initial_customer_message": (
            "Hi, I'm Bob (bob_77). My policy ID is P-102. "
            "I bought a new bike, VIN is 'MOTO-999'. "
            "Update my policy."
        ),
        "required_tools": ["get_policy_details", "update_vehicle"],
        "target_policy_id": "P-102",
    },
    "claim_arbitration": {
        "difficulty": "hard",
        "max_steps": 5,
        "system_prompt": (
            "You are an InsureLink claims agent. A customer named "
            "Bob (policy P-102) is filing a claim for damage to his "
            "own vehicle. You must retrieve his policy, assess eligibility, "
            "calculate the payout, and explain the decision clearly."
        ),
        "initial_customer_message": (
            "I'm Bob (bob_77). My policy ID is P-102. "
            "I crashed my bike and it costs $2000 to fix. "
            "Can I get a payout?"
        ),
        "required_tools": [
            "get_policy_details",
            "calculate_claim_payout",
        ],
        "target_policy_id": "P-102",
    },
}

AVAILABLE_TOOLS = [
    "get_policy_details",
    "update_vehicle",
    "calculate_claim_payout",
]


# ────────────────────────────────────────────────────────────────────
# InsuranceEnv
# ────────────────────────────────────────────────────────────────────
class InsuranceEnv:
    """Async insurance customer-service environment (OpenEnv spec)."""

    def __init__(self) -> None:
        self._db: Dict[str, Any] = {}
        self._task_id: Optional[str] = None
        self._task_cfg: Dict[str, Any] = {}
        self._chat_history: List[Dict[str, str]] = []
        self._step_count: int = 0
        self._max_steps: int = 10
        self._done: bool = False
        self._tools_called: List[str] = []
        self._action_history: List[Dict[str, Any]] = []
        self._last_tool_result: Optional[Any] = None
        self._cumulative_reward: float = 0.0

    # ── lifecycle ────────────────────────────────────────────────
    async def reset(self, task_id: str = "coverage_check") -> StepResult:
        """Reset the environment for a new episode.

        Args:
            task_id: One of ``coverage_check``, ``policy_update``,
                     ``claim_arbitration``.

        Returns:
            A :class:`StepResult` with the initial observation.
        """
        if task_id not in _TASKS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Choose from: {list(_TASKS.keys())}"
            )

        self._db = copy.deepcopy(_INITIAL_DB)
        self._task_id = task_id
        self._task_cfg = _TASKS[task_id]
        self._step_count = 0
        self._max_steps = self._task_cfg["max_steps"]
        self._done = False
        self._tools_called = []
        self._action_history = []
        self._last_tool_result = None
        self._cumulative_reward = 0.0

        # Seed chat with system prompt + first customer message
        self._chat_history = [
            {"role": "system", "content": self._task_cfg["system_prompt"]},
            {"role": "user", "content": self._task_cfg["initial_customer_message"]},
        ]

        obs = InsuranceObservation(
            chat_history=list(self._chat_history),
            tool_result=None,
            available_tools=AVAILABLE_TOOLS,
        )
        return StepResult(observation=obs, reward=0.0, done=False)

    async def step(self, action: InsuranceAction) -> StepResult:
        """Execute one agent turn.

        Args:
            action: The agent's chosen action (tool call and/or message).

        Returns:
            A :class:`StepResult` with updated observation, reward, and
            done flag.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self._step_count += 1
        tool_result: Optional[Any] = None
        tool_reward = 0.0
        error: Optional[str] = None

        # ── record action history ────────────────────────────────
        self._action_history.append({
            "step": self._step_count,
            "tool_name": action.tool_name,
            "tool_args": action.tool_args,
            "message": action.message,
        })

        # ── execute tool (if any) ────────────────────────────────
        if action.tool_name:
            if action.tool_name not in AVAILABLE_TOOLS:
                error = f"Unknown tool: {action.tool_name}"
                tool_result = {"error": error}
            else:
                try:
                    tool_result = self._dispatch_tool(
                        action.tool_name, action.tool_args
                    )
                    self._tools_called.append(action.tool_name)
                    tool_reward = self._compute_tool_reward(action.tool_name)
                except Exception as exc:
                    error = str(exc)
                    tool_result = {"error": error}

        self._last_tool_result = tool_result

        # ── append agent message to chat ─────────────────────────
        if action.message:
            self._chat_history.append(
                {"role": "assistant", "content": action.message}
            )

        # ── check termination ────────────────────────────────────
        done = self._step_count >= self._max_steps
        completion_reward = 0.0

        if done:
            completion_reward = self._compute_completion_reward()

        # ── total reward for this step ───────────────────────────
        comm_reward = self._compute_communication_reward(action.message)
        step_reward = round(tool_reward + completion_reward + comm_reward, 4)
        self._cumulative_reward += step_reward
        self._done = done

        obs = InsuranceObservation(
            chat_history=list(self._chat_history),
            tool_result=tool_result,
            available_tools=AVAILABLE_TOOLS,
        )
        return StepResult(
            observation=obs,
            reward=step_reward,
            done=done,
            info={
                "step": self._step_count,
                "error": error,
                "cumulative_reward": round(self._cumulative_reward, 4),
            },
        )

    def state(self) -> Dict[str, Any]:
        """Return a snapshot of the full environment state (for debugging/grading)."""
        return {
            "task_id": self._task_id,
            "step": self._step_count,
            "max_steps": self._max_steps,
            "done": self._done,
            "tools_called": list(self._tools_called),
            "action_history": list(self._action_history),
            "cumulative_reward": round(self._cumulative_reward, 4),
            "policies": copy.deepcopy(self._db.get("policies", {})),
            "users": copy.deepcopy(self._db.get("users", {})),
            "chat_history": list(self._chat_history),
        }

    async def close(self) -> None:
        """Clean up resources (no-op for in-memory env)."""
        self._db = {}
        self._chat_history = []
        self._action_history = []
        self._done = True

    # ── class-level factory (compatible with OpenEnv loaders) ────
    @classmethod
    async def from_docker_image(cls, image_name: Optional[str] = None) -> "InsuranceEnv":
        """Factory that mirrors the OpenEnv ``from_docker_image`` API.

        Since InsureLink runs entirely in-process, the *image_name*
        parameter is accepted but ignored.
        """
        return cls()

    # ────────────────────────────────────────────────────────────
    # Tool implementations
    # ────────────────────────────────────────────────────────────
    def _resolve_policy_id(self, policy_id: str) -> str:
        """Resolve a policy_id OR user_id to a valid policy_id.

        The LLM may pass a user_id (e.g. 'alice_99') instead of a
        policy_id (e.g. 'P-101'). This helper transparently resolves
        user_ids by looking up the user record.
        """
        # Direct match on policies table
        if policy_id in self._db["policies"]:
            return policy_id

        # Try resolving as a user_id
        user = self._db["users"].get(policy_id)
        if user and "policy_id" in user:
            return user["policy_id"]

        # Try case-insensitive / name-based lookup
        for uid, udata in self._db["users"].items():
            if (policy_id.lower() == uid.lower()
                    or policy_id.lower() == udata.get("name", "").lower()):
                return udata["policy_id"]

        raise ValueError(
            f"Policy or user '{policy_id}' not found. "
            f"Valid policy IDs: {list(self._db['policies'].keys())}. "
            f"Valid user IDs: {list(self._db['users'].keys())}."
        )

    def _dispatch_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        handler = {
            "get_policy_details": self._tool_get_policy_details,
            "update_vehicle": self._tool_update_vehicle,
            "calculate_claim_payout": self._tool_calculate_claim_payout,
        }[tool_name]
        return handler(**args)

    # ── get_policy_details ───────────────────────────────────────
    def _tool_get_policy_details(self, policy_id: str) -> Dict[str, Any]:
        """Return coverage type and deductible for a given policy."""
        policy_id = self._resolve_policy_id(policy_id)
        policy = self._db["policies"][policy_id]

        return {
            "policy_id": policy["policy_id"],
            "coverage_type": policy["coverage_type"],
            "deductible": policy["deductible"],
            "vehicle_make": policy["vehicle_make"],
            "vehicle_model": policy["vehicle_model"],
            "vehicle_year": policy["vehicle_year"],
            "vehicle_vin": policy["vehicle_vin"],
        }

    # ── update_vehicle ───────────────────────────────────────────
    def _tool_update_vehicle(
        self, policy_id: str, vin: str
    ) -> Dict[str, Any]:
        """Update the VIN on a policy's vehicle record."""
        policy_id = self._resolve_policy_id(policy_id)
        policy = self._db["policies"][policy_id]

        old_vin = policy["vehicle_vin"]
        policy["vehicle_vin"] = vin

        return {
            "status": "updated",
            "policy_id": policy_id,
            "old_vin": old_vin,
            "new_vin": vin,
        }

    # ── calculate_claim_payout ───────────────────────────────────
    def _tool_calculate_claim_payout(
        self,
        policy_id: str,
        repair_cost: float,
    ) -> Dict[str, Any]:
        """Calculate the claim payout based on policy type.

        Comprehensive: payout = repair_cost - deductible
        Third-Party:   payout = 0 (does not cover own vehicle damage)
        """
        policy_id = self._resolve_policy_id(policy_id)
        policy = self._db["policies"][policy_id]

        coverage_type = policy["coverage_type"]
        deductible = policy["deductible"]

        if coverage_type == "Third-Party":
            return {
                "eligible": False,
                "payout": 0,
                "reason": "Third-party does not cover own vehicle damage",
            }

        # Comprehensive
        payout = max(repair_cost - deductible, 0)
        return {
            "eligible": True,
            "repair_cost": repair_cost,
            "deductible": deductible,
            "payout": round(payout, 2),
            "reason": f"Comprehensive coverage: ${repair_cost} - ${deductible} deductible = ${payout:.2f} payout",
        }

    # ────────────────────────────────────────────────────────────
    # Reward helpers
    # ────────────────────────────────────────────────────────────
    def _compute_tool_reward(self, tool_name: str) -> float:
        """Reward for calling a required tool (weight: 0.4 total)."""
        required = self._task_cfg.get("required_tools", [])
        if tool_name in required:
            per_tool = 0.4 / max(len(required), 1)
            return round(per_tool, 4)
        return 0.0

    def _compute_completion_reward(self) -> float:
        """Reward for how many required tools were actually called (0.4)."""
        required = set(self._task_cfg.get("required_tools", []))
        if not required:
            return 0.4
        called = set(self._tools_called)
        fraction = len(required & called) / len(required)
        return round(0.4 * fraction, 4)

    def _compute_communication_reward(
        self, message: Optional[str]
    ) -> float:
        """Small reward for providing a substantive customer message (0.2 max)."""
        if not message:
            return 0.0
        length = len(message.strip())
        if length == 0:
            return 0.0
        max_per_step = 0.2 / self._max_steps
        return round(min(length / 200.0, 1.0) * max_per_step, 4)
