"""
Curriculum Learning Manager for the API Debugging Environment.

Tasks are ordered by difficulty. The manager tracks per-task performance and
automatically unlocks the next task when the agent masters the current one
(rolling average score >= mastery_threshold over the last N episodes).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

# Ordered from easiest to hardest
CURRICULUM_ORDER: List[str] = [
    # --- Easy ---
    "timeout_debug",
    # --- Medium ---
    "auth_failure",
    "rate_limit_429",
    "db_timeout",
    "ssl_cert_expiry",
    # --- Hard ---
    "deployment_500",
    "cascading_failure",
]

DIFFICULTY_MAP: Dict[str, str] = {
    "timeout_debug": "easy",
    "auth_failure": "medium",
    "rate_limit_429": "medium",
    "db_timeout": "medium",
    "ssl_cert_expiry": "medium",
    "deployment_500": "hard",
    "cascading_failure": "hard",
}


class CurriculumManager:
    """
    Manages task unlocking based on rolling average performance.

    Parameters
    ----------
    mastery_threshold : float
        Score (0–1) required to unlock the next task. Default 0.8.
    window : int
        Number of recent episodes used to compute the rolling average. Default 3.
    """

    def __init__(self, mastery_threshold: float = 0.8, window: int = 3) -> None:
        self.mastery_threshold = mastery_threshold
        self.window = window
        self._history: Dict[str, List[float]] = defaultdict(list)
        self._unlocked_idx: int = 0  # index into CURRICULUM_ORDER

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, task_name: str, score: float) -> bool:
        """
        Record a finished episode score for *task_name*.

        Returns True if this score caused a new task to be unlocked.
        """
        self._history[task_name].append(round(score, 4))
        return self._try_unlock()

    def current_task(self) -> str:
        """Return the task the agent should work on next."""
        return CURRICULUM_ORDER[self._unlocked_idx]

    def unlocked_tasks(self) -> List[str]:
        """Return all tasks that have been unlocked so far."""
        return CURRICULUM_ORDER[: self._unlocked_idx + 1]

    def is_unlocked(self, task_name: str) -> bool:
        return task_name in self.unlocked_tasks()

    def mastery(self, task_name: str) -> float:
        """Rolling average score for *task_name* (0.0 if no history)."""
        scores = self._history.get(task_name, [])
        if not scores:
            return 0.0
        recent = scores[-self.window :]
        return round(sum(recent) / len(recent), 4)

    def progress(self) -> dict:
        """Return a summary dict suitable for logging / display."""
        return {
            "current_task": self.current_task(),
            "unlocked": self.unlocked_tasks(),
            "mastery_scores": {t: self.mastery(t) for t in CURRICULUM_ORDER},
            "history": dict(self._history),
        }

    def reset(self) -> None:
        """Clear all history and restart from the first task."""
        self._history.clear()
        self._unlocked_idx = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _try_unlock(self) -> bool:
        """Unlock the next task if the current one is mastered over >= window episodes."""
        if self._unlocked_idx >= len(CURRICULUM_ORDER) - 1:
            return False  # already at final task
        current = CURRICULUM_ORDER[self._unlocked_idx]
        scores = self._history.get(current, [])
        # Require a full window of episodes before considering unlock
        if len(scores) >= self.window and self.mastery(current) >= self.mastery_threshold:
            self._unlocked_idx += 1
            return True
        return False
