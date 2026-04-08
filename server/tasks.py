"""
InsureLink-v1 — Deterministic Graders
=======================================
Each grading function returns a float in [0.0, 1.0].

    grade_coverage_check   (Easy)   — Did the agent report Alice's $500 deductible?
    grade_policy_update    (Medium) — Was Bob's VIN actually updated in the DB?
    grade_claim_arbitration (Hard)  — Did the agent correctly deny Bob's own-vehicle claim?
"""

from __future__ import annotations

from typing import Any, Dict, List


# ── Easy: Coverage Check ─────────────────────────────────────────────
def grade_coverage_check(state: Dict[str, Any], agent_output: str) -> float:
    """
    Easy Task — Coverage Check.

    Alice has Comprehensive coverage with a $500 deductible (policy P-101).
    The agent gets 1.0 only if their response mentions the $500 deductible.
    Partial credit (0.5) if they at least called get_policy_details.
    """
    score = 0.0

    # Partial credit for calling the right tool
    tools_called = state.get("tools_called", [])
    if "get_policy_details" in tools_called:
        score = 0.5

    # Full credit for correctly communicating the deductible
    message = agent_output.lower()
    if "$500" in message or "500" in message:
        # Verify the agent is talking about the deductible, not random 500
        deductible_keywords = ["deductible", "500"]
        if all(kw in message for kw in deductible_keywords):
            score = 1.0

    return score


# ── Medium: Policy Update ────────────────────────────────────────────
def grade_policy_update(state: Dict[str, Any], action_history: List[Dict[str, Any]]) -> float:
    """
    Medium Task — Policy Update.

    Bob (policy P-102) wants to change his vehicle VIN to MOTO-999.
    1.0 if the internal dictionary was actually modified.
    0.5 if the agent called update_vehicle but with wrong data.
    """
    # Check if the DB was actually mutated
    policies = state.get("policies", {})
    bob_policy = policies.get("P-102", {})

    if bob_policy.get("vehicle_vin") == "MOTO-999":
        return 1.0

    # Partial credit: called the tool but VIN doesn't match
    tools_called = state.get("tools_called", [])
    if "update_vehicle" in tools_called:
        return 0.5

    return 0.0


# ── Hard: Claim Arbitration ──────────────────────────────────────────
def grade_claim_arbitration(state: Dict[str, Any], agent_output: str) -> float:
    """
    Hard Task — Claim Arbitration.

    Bob has Third-Party insurance. If he crashes HIS bike, the payout is $0.
    The agent gets 1.0 only if they explain that his policy doesn't cover
    his own repairs.

    Partial credit:
        0.25 — called get_policy_details
        0.50 — called calculate_claim_payout
        1.00 — correctly explained the denial
    """
    score = 0.0

    tools_called = state.get("tools_called", [])

    # 0.25 for looking up the policy
    if "get_policy_details" in tools_called:
        score = 0.25

    # 0.50 for running the claim calculation
    if "calculate_claim_payout" in tools_called:
        score = 0.50

    # 1.0 for correctly communicating the denial
    denied_keywords = [
        "not covered",
        "denied",
        "$0",
        "third-party",
        "third party",
        "liability only",
        "does not cover",
        "doesn't cover",
        "not eligible",
        "own vehicle damage",
        "payout is 0",
        "payout: $0",
        "payout of $0",
        "zero payout",
    ]
    message = agent_output.lower()
    if any(kw in message for kw in denied_keywords):
        score = 1.0

    return score


# ── Convenience: run all graders for a given task ────────────────────
def grade_task(
    task_id: str,
    state: Dict[str, Any],
    agent_output: str = "",
) -> float:
    """Dispatch to the correct grader based on task_id.

    Args:
        task_id: One of coverage_check, policy_update, claim_arbitration.
        state: The dict returned by ``env.state()``.
        agent_output: The final agent message (concatenated assistant turns).

    Returns:
        Score in [0.0, 1.0].
    """
    action_history = state.get("action_history", [])

    if task_id == "coverage_check":
        return grade_coverage_check(state, agent_output)
    elif task_id == "policy_update":
        return grade_policy_update(state, action_history)
    elif task_id == "claim_arbitration":
        return grade_claim_arbitration(state, agent_output)
    else:
        raise ValueError(f"Unknown task_id: {task_id}")
