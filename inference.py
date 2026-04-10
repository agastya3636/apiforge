#!/usr/bin/env python3
"""
Deterministic baseline inference script for the API Debugging Environment.

Environment variables
---------------------
API_BASE_URL  : Base URL of the OpenAI-compatible API (default: https://api.openai.com/v1)
MODEL_NAME    : Model identifier to use             (default: gpt-4o-mini)
HF_TOKEN      : API key / Hugging Face token

Output format (DO NOT CHANGE)
------------------------------
[START]
task: <task_name>

[STEP]
action: <action>
reward: <reward>

[END]
final_score: <score>
"""

import os
import sys

from openai import OpenAI

from env.environment import APIDebugEnv
from env.models import VALID_ACTIONS
from env.tasks import TASK_NAMES

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

SYSTEM_PROMPT = f"""You are a backend engineer debugging API issues.

You will receive the current system state and must choose exactly ONE action.

Available actions:
{chr(10).join(f"  - {a}" for a in VALID_ACTIONS)}

Rules:
- Respond with ONLY the action name — no explanation, no punctuation, no extra text.
- The action must be one of the listed strings above, spelled exactly.
- If unsure, respond with: do_nothing
"""


def build_user_prompt(obs: dict, task_name: str) -> str:
    hints_str = "\n".join(f"  • {h}" for h in obs["hints"]) if obs["hints"] else "  (none)"
    return (
        f"Task: {task_name}\n"
        f"Status Code : {obs['status_code']}\n"
        f"Error       : {obs['error'] or 'None'}\n"
        f"Latency (ms): {obs['latency_ms']}\n"
        f"Hints:\n{hints_str}\n\n"
        f"Logs:\n{obs['logs']}\n"
        f"What action do you take?"
    )


def get_action(client: OpenAI, obs: dict, task_name: str) -> str:
    """Query the LLM and return a sanitized, valid action string."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(obs, task_name)},
        ],
        max_tokens=20,
        temperature=0,  # deterministic
        seed=42,
    )
    raw = response.choices[0].message.content.strip().lower()

    # Exact match first
    if raw in VALID_ACTIONS:
        return raw

    # Substring match (handles extra punctuation / whitespace)
    for action in VALID_ACTIONS:
        if action in raw:
            return action

    return "do_nothing"


def run_task(client: OpenAI, env: APIDebugEnv, task_name: str) -> float:
    print("[START]")
    print(f"task: {task_name}")
    print()
    sys.stdout.flush()

    obs = env.reset(task_name)
    done = False

    while not done:
        obs_dict = obs.model_dump()
        action = get_action(client, obs_dict, task_name)

        result = env.step(action)
        obs = result.observation
        done = result.done

        print("[STEP]")
        print(f"action: {action}")
        print(f"reward: {result.reward}")
        print()
        sys.stdout.flush()

    final_score = env.final_score()

    print("[END]")
    print(f"final_score: {final_score}")
    print()
    sys.stdout.flush()

    return final_score


def main() -> None:
    api_key = HF_TOKEN or os.environ.get("OPENAI_API_KEY", "sk-placeholder")
    client = OpenAI(api_key=api_key, base_url=API_BASE_URL)
    env = APIDebugEnv(seed=42)

    scores: dict[str, float] = {}
    for task_name in TASK_NAMES:
        scores[task_name] = run_task(client, env, task_name)

    # Summary
    avg = round(sum(scores.values()) / len(scores), 4)
    print("=== SUMMARY ===")
    for name, score in scores.items():
        print(f"  {name}: {score}")
    print(f"  average: {avg}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
