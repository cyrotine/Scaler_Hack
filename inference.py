"""
InsureLink-v1 — Inference Script (Dry Run)
============================================
Runs all three InsureLink tasks sequentially against an LLM:

    1. coverage_check    (Easy)   — Alice asks for her deductible
    2. policy_update     (Medium) — Bob updates his VIN to MOTO-999
    3. claim_arbitration (Hard)   — Bob tries to claim $2000 on Third-Party

MANDATORY env vars:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT (no exceptions):
    [START] task=<task> env=insurelink_v1 model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<null|msg>
    [END]   success=<true|false> steps=<n> score=<float> rewards=<r1,r2,...>
"""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from server.models import InsuranceAction
from server.env import InsuranceEnv
from server.tasks import grade_task

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration (read from environment)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
IMAGE_NAME = os.getenv("IMAGE_NAME")  # accepted but unused (in-memory env)

BENCHMARK = "insurelink_v1"
MAX_STEPS = 5
TEMPERATURE = 0.4
MAX_TOKENS = 512

# All tasks to run in a single dry-run session
TASKS = ["coverage_check", "policy_update", "claim_arbitration"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# System prompt — tells the LLM exactly what it is and how to respond
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SYSTEM_PROMPT = textwrap.dedent("""\
You are an InsureLink Agent — a professional insurance customer-service
representative. You must resolve the customer's query using the tools
below and provide a final answer.

CUSTOMER DATABASE (use these policy IDs in all tool calls):
  • alice_99 → Policy ID: P-101 (Comprehensive, $500 deductible, 2024 Tesla Model 3)
  • bob_77   → Policy ID: P-102 (Third-Party, $0 deductible, 2018 Yamaha R1)

AVAILABLE TOOLS:

1. get_policy_details(policy_id: str)
   → Returns coverage type, deductible, and vehicle information.
   Example: {"tool_name": "get_policy_details", "tool_args": {"policy_id": "P-101"}, "message": "Let me look up your policy."}

2. update_vehicle(policy_id: str, vin: str)
   → Updates the VIN on the customer's policy record.
   Example: {"tool_name": "update_vehicle", "tool_args": {"policy_id": "P-102", "vin": "NEW_VIN"}, "message": "Updating your vehicle now."}

3. calculate_claim_payout(policy_id: str, repair_cost: float)
   → Calculates the claim payout based on policy type.
     • Comprehensive: payout = repair_cost − deductible
     • Third-Party: payout = $0 (Third-party does not cover own vehicle damage)
   Example: {"tool_name": "calculate_claim_payout", "tool_args": {"policy_id": "P-102", "repair_cost": 2000}, "message": "Calculating your claim."}

RESPONSE FORMAT — You MUST reply with EXACTLY one JSON object per turn (no markdown fences, no extra text):
{
  "tool_name": "<tool_name or null>",
  "tool_args": { ... },
  "message": "<your professional reply to the customer>"
}

RULES:
- ALWAYS use the policy ID (P-101 or P-102), NOT the user ID, in tool_args.
- Call get_policy_details FIRST before any other tool.
- After getting tool results, provide a clear final answer with specific numbers.
- When done, set "tool_name" to null and give your final answer in "message".
- Be concise, professional, and always cite exact dollar amounts.
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mandatory stdout logging
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} "
        f"error={error or 'null'}",
        flush=True,
    )


def log_end(
    success: bool, steps: int, score: float, rewards: List[float]
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM interaction helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_openai_messages(
    chat_history: List[Dict[str, str]],
    tool_result: Optional[Any],
) -> List[Dict[str, str]]:
    """Convert environment chat history into OpenAI-compatible messages."""
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    for turn in chat_history:
        if turn["role"] == "system":
            continue  # skip env system prompt; we use our own
        messages.append({"role": turn["role"], "content": turn["content"]})

    # Feed the tool result back so the model can reason about it
    if tool_result is not None:
        messages.append({
            "role": "user",
            "content": f"[Tool Result]\n{json.dumps(tool_result, indent=2)}",
        })

    return messages


def _parse_agent_response(raw: str) -> InsuranceAction:
    """Parse the model's JSON response into an InsuranceAction.

    Falls back to a chat-only action if JSON parsing fails.
    """
    text = raw.strip()

    # Strip markdown code fences if the model wraps its response
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
        return InsuranceAction(
            tool_name=data.get("tool_name"),
            tool_args=data.get("tool_args", {}),
            message=data.get("message", ""),
        )
    except (json.JSONDecodeError, Exception):
        # Treat entire response as a plain chat message
        return InsuranceAction(tool_name=None, tool_args={}, message=raw)


def get_agent_action(
    client: OpenAI,
    chat_history: List[Dict[str, str]],
    tool_result: Optional[Any],
) -> InsuranceAction:
    """Query the LLM and return a parsed InsuranceAction."""
    messages = _build_openai_messages(chat_history, tool_result)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return _parse_agent_response(text)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return InsuranceAction(
            tool_name=None,
            tool_args={},
            message="I apologise — I encountered a technical issue. "
                    "Could you please repeat your request?",
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Human-readable display helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TASK_LABELS = {
    "coverage_check": ("📋 Coverage Check", "Easy", "Alice wants to know her deductible"),
    "policy_update": ("🔧 Policy Update", "Medium", "Bob wants to update his vehicle VIN"),
    "claim_arbitration": ("⚖️  Claim Arbitration", "Hard", "Bob files a damage claim on Third-Party policy"),
}


def print_banner(task_id: str) -> None:
    label, difficulty, desc = TASK_LABELS.get(task_id, (task_id, "?", ""))
    print(flush=True)
    print(f"┌{'─' * 58}┐", flush=True)
    print(f"│  {label:<40} [{difficulty}]     │", flush=True)
    print(f"│  {desc:<54} │", flush=True)
    print(f"└{'─' * 58}┘", flush=True)
    print(flush=True)


def print_customer(message: str) -> None:
    print(f"  🧑 Customer: {message}", flush=True)
    print(flush=True)


def print_tool_call(tool_name: str, tool_args: Dict[str, Any]) -> None:
    args_str = ", ".join(f"{k}={v!r}" for k, v in tool_args.items())
    print(f"  🔧 Tool Call: {tool_name}({args_str})", flush=True)


def print_tool_result(result: Any) -> None:
    if isinstance(result, dict):
        for k, v in result.items():
            print(f"     ├─ {k}: {v}", flush=True)
    else:
        print(f"     └─ {result}", flush=True)
    print(flush=True)


def print_agent(message: Optional[str]) -> None:
    if message:
        print(f"  🤖 Agent: {message}", flush=True)
        print(flush=True)


def print_score_card(task_id: str, score: float, success: bool, steps: int) -> None:
    label, difficulty, _ = TASK_LABELS.get(task_id, (task_id, "?", ""))
    icon = "✅" if success else "❌"
    bar_filled = int(score * 20)
    bar = "█" * bar_filled + "░" * (20 - bar_filled)
    print(f"  {icon} Score: [{bar}] {score:.0%}  ({steps} steps)", flush=True)
    print(flush=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run a single task episode
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def run_task(
    client: OpenAI,
    env: InsuranceEnv,
    task_id: str,
) -> float:
    """Run one complete episode for *task_id*. Returns the score."""

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    print_banner(task_id)

    try:
        # ── Reset environment ────────────────────────────────────
        result = await env.reset(task_id=task_id)
        obs = result.observation

        # Show the initial customer message
        customer_msg = obs.chat_history[-1]["content"]
        print_customer(customer_msg)

        # ── Agent loop (max 5 steps) ─────────────────────────────
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # Ask the LLM what to do
            action = get_agent_action(
                client,
                obs.chat_history,
                obs.tool_result,
            )

            # Execute in the environment
            result = await env.step(action)
            obs = result.observation

            reward = result.reward
            done = result.done
            error = result.info.get("error")

            rewards.append(reward)
            steps_taken = step

            # ── Show what happened in human-readable form ────────
            if action.tool_name:
                print_tool_call(action.tool_name, action.tool_args)
                if error:
                    print(f"     ⚠️  Error: {error}", flush=True)
                    print(flush=True)
                elif obs.tool_result:
                    print_tool_result(obs.tool_result)

            if action.message:
                print_agent(action.message)

            # ── Mandatory machine-format log ─────────────────────
            action_label = action.tool_name or "message"
            if action.tool_name and action.tool_args:
                args_short = ",".join(
                    f"{k}={v}" for k, v in action.tool_args.items()
                )
                action_label = f"{action.tool_name}({args_short})"

            log_step(
                step=step,
                action=action_label,
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        # ── Deterministic grading ────────────────────────────────
        env_state = env.state()

        # Concatenate all assistant messages for the grader
        agent_output = " ".join(
            turn["content"]
            for turn in env_state.get("chat_history", [])
            if turn["role"] == "assistant"
        )

        score = grade_task(task_id, env_state, agent_output)
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.5

        print_score_card(task_id, score, success, steps_taken)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )

    return score


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main — run all three tasks sequentially
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(flush=True)
    print("╔════════════════════════════════════════════════════════════╗", flush=True)
    print("║           🏦  InsureLink-v1  —  Dry Run                  ║", flush=True)
    print("╠════════════════════════════════════════════════════════════╣", flush=True)
    print(f"║  Model    : {MODEL_NAME:<45} ║", flush=True)
    print(f"║  Endpoint : {API_BASE_URL:<45} ║", flush=True)
    print(f"║  Max steps: {MAX_STEPS:<45} ║", flush=True)
    print("╚════════════════════════════════════════════════════════════╝", flush=True)

    scores: Dict[str, float] = {}
    for task_id in TASKS:
        env = InsuranceEnv()
        task_score = await run_task(client, env, task_id)
        scores[task_id] = task_score

    # ── Final Report Card ────────────────────────────────────────
    print(flush=True)
    print("╔════════════════════════════════════════════════════════════╗", flush=True)
    print("║                   📊  FINAL REPORT CARD                  ║", flush=True)
    print("╠════════════════════════════════════════════════════════════╣", flush=True)
    for tid, sc in scores.items():
        label, diff, _ = TASK_LABELS.get(tid, (tid, "?", ""))
        icon = "✅" if sc >= 0.5 else "❌"
        print(f"║  {icon} {label:<30} [{diff:<6}]  {sc:.0%}       ║", flush=True)
    avg = sum(scores.values()) / len(scores) if scores else 0
    print("╠════════════════════════════════════════════════════════════╣", flush=True)
    print(f"║  🏆 Overall Score: {avg:.0%}                                     ║", flush=True)
    all_pass = all(s >= 0.5 for s in scores.values())
    verdict = "ALL TASKS PASSED ✅" if all_pass else "SOME TASKS FAILED ❌"
    print(f"║  {verdict:<55} ║", flush=True)
    print("╚════════════════════════════════════════════════════════════╝", flush=True)
    print(flush=True)


if __name__ == "__main__":
    asyncio.run(main())

