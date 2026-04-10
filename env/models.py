from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

VALID_ACTIONS: List[str] = [
    "check_logs",
    "retry_request",
    "fix_auth_token",
    "increase_timeout",
    "restart_service",
    "rollback_version",
    "analyze_deployment",
    "implement_backoff",
    "renew_certificate",
    "do_nothing",
]

# Actions that can harm the system when applied incorrectly
DESTRUCTIVE_ACTIONS: List[str] = [
    "restart_service",
    "rollback_version",
]


class Observation(BaseModel):
    logs: str
    status_code: int
    latency_ms: int
    error: str
    hints: List[str] = Field(default_factory=list)


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class TaskConfig(BaseModel):
    name: str
    difficulty: str  # easy | medium | hard
    description: str
    max_steps: int
