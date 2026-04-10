#!/usr/bin/env python3
"""
Gradio dashboard for the API Debugging Environment.

Usage:
    python dashboard.py
    python dashboard.py --share   # generate public URL

Features:
  - Task selector with difficulty badge
  - Noise level and penalty sliders
  - One-click action buttons with colour-coded rewards
  - Live observation panel (logs, status, error, latency, hints)
  - Step-reward bar chart (matplotlib)
  - Cumulative score and episode status
"""

import argparse
import sys

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.use("Agg")

from env.curriculum import CURRICULUM_ORDER, DIFFICULTY_MAP
from env.environment import APIDebugEnv
from env.models import VALID_ACTIONS
from env.tasks import TASK_NAMES

# ---------------------------------------------------------------------------
# Global episode state (one env per Gradio session)
# ---------------------------------------------------------------------------
_env = APIDebugEnv(seed=42)
_step_rewards: list[float] = []
_episode_done: bool = False

DIFFICULTY_EMOJI = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}
TASK_CHOICES = [
    f"{DIFFICULTY_EMOJI[DIFFICULTY_MAP[t]]} {t}" for t in TASK_NAMES
]


def _task_name_from_choice(choice: str) -> str:
    """Strip the emoji prefix from the dropdown label."""
    return choice.split(" ", 1)[1]


# ---------------------------------------------------------------------------
# Core callbacks
# ---------------------------------------------------------------------------

def reset_task(task_choice: str, noise_level: float, step_penalty: float, destructive_penalty: float):
    global _env, _step_rewards, _episode_done

    task_name = _task_name_from_choice(task_choice)
    difficulty = DIFFICULTY_MAP[task_name]

    _env = APIDebugEnv(
        seed=42,
        noise_level=noise_level,
        step_penalty=step_penalty,
        destructive_penalty=destructive_penalty,
    )
    obs = _env.reset(task_name)
    _step_rewards = []
    _episode_done = False

    obs_text = _format_obs(obs)
    status_text = f"Episode started — Task: {task_name} [{difficulty.upper()}]"
    chart = _make_chart([])

    return obs_text, 0.0, status_text, chart, _make_action_log("")


def take_action(action: str):
    global _step_rewards, _episode_done

    if _episode_done:
        return (
            _format_obs(_env.state()),
            _env.total_reward,
            "Episode already done. Press Reset to start a new episode.",
            _make_chart(_step_rewards),
            _make_action_log(f"⚠️  Episode done — ignored action: {action}"),
        )

    if _env.current_task is None:
        return (
            "No task loaded. Select a task and press Reset.",
            0.0,
            "Not started.",
            _make_chart([]),
            _make_action_log(""),
        )

    result = _env.step(action)
    _step_rewards.append(result.reward)
    _episode_done = result.done

    obs_text = _format_obs(result.observation)
    cumulative = _env.total_reward

    if result.done:
        final = _env.final_score()
        grade_str = _grade_label(final)
        status_text = f"✅ Episode complete — Final Score: {final:.4f}  {grade_str}"
    else:
        status_text = f"Step {_env._steps} — reward this step: {result.reward:+.4f}"

    log_line = f"Step {len(_step_rewards)}: action={action}  reward={result.reward:+.4f}"
    chart = _make_chart(_step_rewards)

    return obs_text, cumulative, status_text, chart, _make_action_log(log_line)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_obs(obs) -> str:
    status_emoji = "✅" if obs.status_code == 200 else "❌"
    hints = "\n  ".join(obs.hints) if obs.hints else "(none)"
    return (
        f"{status_emoji} Status Code : {obs.status_code}\n"
        f"   Error      : {obs.error or 'None'}\n"
        f"   Latency    : {obs.latency_ms} ms\n\n"
        f"💡 Hints:\n  {hints}\n\n"
        f"📋 Logs:\n{obs.logs}"
    )


def _grade_label(score: float) -> str:
    if score >= 1.0:
        return "🏆 Perfect"
    if score >= 0.7:
        return "👍 Good"
    if score >= 0.4:
        return "⚠️  Partial"
    return "❌ Poor"


def _make_action_log(last_line: str) -> str:
    return last_line


def _make_chart(rewards: list[float]):
    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    if rewards:
        colors = ["#a6e3a1" if r > 0 else "#f38ba8" if r < 0 else "#6c7086"
                  for r in rewards]
        ax.bar(range(1, len(rewards) + 1), rewards, color=colors, edgecolor="#313244")
        ax.set_ylim(
            min(-0.15, min(rewards) - 0.05),
            max(1.1, max(rewards) + 0.1),
        )
        ax.axhline(0, color="#6c7086", linewidth=0.8, linestyle="--")

        cumulative = []
        total = 0.0
        for r in rewards:
            total += r
            cumulative.append(round(total, 4))
        ax.plot(
            range(1, len(cumulative) + 1), cumulative,
            color="#89b4fa", linewidth=1.8, marker="o", markersize=4, label="cumulative",
        )
        ax.legend(facecolor="#313244", edgecolor="none", labelcolor="#cdd6f4", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No steps yet", ha="center", va="center",
                transform=ax.transAxes, color="#6c7086", fontsize=11)
        ax.set_ylim(0, 1)

    ax.set_xlabel("Step", color="#cdd6f4", fontsize=9)
    ax.set_ylabel("Reward", color="#cdd6f4", fontsize=9)
    ax.set_title("Step Rewards + Cumulative Score", color="#cdd6f4", fontsize=10)
    ax.tick_params(colors="#cdd6f4")
    for spine in ax.spines.values():
        spine.set_edgecolor("#313244")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="API Debug Env",
        theme=gr.themes.Soft(),
        css="""
        .action-btn { font-size: 0.82rem !important; min-width: 130px; }
        .obs-box textarea { font-family: monospace !important; font-size: 0.82rem; }
        """,
    ) as demo:

        gr.Markdown(
            "# 🔧 API Debugging Environment\n"
            "Select a task, configure noise/penalties, hit **Reset**, then click action buttons to debug the API."
        )

        # ── Configuration row ──────────────────────────────────────────────
        with gr.Row():
            task_dd = gr.Dropdown(
                choices=TASK_CHOICES,
                value=TASK_CHOICES[0],
                label="Task",
                scale=2,
            )
            noise_sl = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Noise Level")
            step_pen_sl = gr.Slider(0.0, 0.2, value=0.0, step=0.01, label="Step Penalty")
            dest_pen_sl = gr.Slider(0.0, 0.3, value=0.0, step=0.01, label="Destructive Penalty")
            reset_btn = gr.Button("🔄 Reset", variant="primary", scale=1)

        # ── Main content area ───────────────────────────────────────────────
        with gr.Row():
            # Left — observation
            with gr.Column(scale=3):
                obs_box = gr.Textbox(
                    label="Current Observation",
                    lines=22,
                    interactive=False,
                    elem_classes=["obs-box"],
                    value="Press Reset to start an episode.",
                )

            # Right — actions + score
            with gr.Column(scale=2):
                gr.Markdown("### Actions")
                action_buttons = []
                rows = [VALID_ACTIONS[i:i+2] for i in range(0, len(VALID_ACTIONS), 2)]
                for row_actions in rows:
                    with gr.Row():
                        for act in row_actions:
                            btn = gr.Button(act, elem_classes=["action-btn"])
                            action_buttons.append((act, btn))

                gr.Markdown("---")
                score_box = gr.Number(label="Cumulative Score", value=0.0, precision=4)
                status_box = gr.Textbox(
                    label="Episode Status", value="Not started.", interactive=False, lines=2
                )
                last_action_box = gr.Textbox(
                    label="Last Action Log", interactive=False, lines=1
                )

        # ── Reward chart ───────────────────────────────────────────────────
        with gr.Row():
            reward_chart = gr.Plot(label="Step Rewards & Cumulative Score", scale=1)

        # ── Wiring ─────────────────────────────────────────────────────────
        all_outputs = [obs_box, score_box, status_box, reward_chart, last_action_box]

        reset_btn.click(
            fn=reset_task,
            inputs=[task_dd, noise_sl, step_pen_sl, dest_pen_sl],
            outputs=all_outputs,
        )

        for act, btn in action_buttons:
            btn.click(
                fn=lambda a=act: take_action(a),
                inputs=[],
                outputs=all_outputs,
            )

        # ── Curriculum sidebar (collapsible) ──────────────────────────────
        with gr.Accordion("📚 Task Curriculum Order", open=False):
            curriculum_md = "\n".join(
                f"| {i+1} | {DIFFICULTY_EMOJI[DIFFICULTY_MAP[t]]} `{t}` | {DIFFICULTY_MAP[t].capitalize()} |"
                for i, t in enumerate(CURRICULUM_ORDER)
            )
            gr.Markdown(
                "| # | Task | Difficulty |\n|---|---|---|\n" + curriculum_md
            )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API Debug Env Dashboard")
    parser.add_argument("--share", action="store_true", help="Generate public Gradio URL")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    args = parser.parse_args()

    ui = build_ui()
    ui.launch(share=args.share, server_port=args.port)
