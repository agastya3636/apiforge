from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict

if TYPE_CHECKING:
    from env.environment import APIDebugEnv


def grade_easy(env: "APIDebugEnv") -> float:
    """
    Grader for timeout_debug (easy).

    Scoring:
        1.0  — increase_timeout applied   (status 200, full fix)
        0.5  — retry_request only         (partial: surfaced issue, not resolved)
        0.0  — wrong or no action
    """
    obs = env.state()
    if obs.status_code == 200:
        return 1.0
    # partial_fix flag is set when retry was used but root cause not addressed
    if env.partial_fix:
        return 0.5
    return 0.0


def grade_medium(env: "APIDebugEnv") -> float:
    """
    Grader for auth_failure, rate_limit_429, db_timeout, ssl_cert_expiry (medium).

    Scoring (stepwise, cumulative):
        0.3  — check_logs performed (discovery step)
        0.6  — fix applied without prior log check (skipped investigation)
        1.0  — check_logs → fix applied (full investigation + fix)
    """
    return round(env.total_reward, 4)


def grade_hard(env: "APIDebugEnv") -> float:
    """
    Grader for deployment_500 and cascading_failure (hard).

    Scoring (stepwise):
        0.3  — check_logs (identified timeline)
        +0.3 — analyze_deployment (confirmed root cause)
        +0.4 — rollback_version (service restored)
        ──────────────────────────────────────────
        1.0  total for full correct path
    """
    return round(env.total_reward, 4)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GRADERS: Dict[str, Callable[["APIDebugEnv"], float]] = {
    "timeout_debug":    grade_easy,
    "auth_failure":     grade_medium,
    "rate_limit_429":   grade_medium,
    "db_timeout":       grade_medium,
    "ssl_cert_expiry":  grade_medium,
    "deployment_500":   grade_hard,
    "cascading_failure": grade_hard,
}


def grade(env: "APIDebugEnv") -> float:
    """Return the final score for the current episode."""
    grader = GRADERS.get(env.current_task)
    if grader is None:
        raise ValueError(f"No grader registered for task: '{env.current_task}'")
    return grader(env)
