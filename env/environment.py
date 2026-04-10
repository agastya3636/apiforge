from __future__ import annotations

from typing import Optional, Tuple

from env.grader import grade
from env.models import DESTRUCTIVE_ACTIONS, VALID_ACTIONS, Observation, StepResult
from env.noise import NoiseInjector
from env.tasks import TASKS


class APIDebugEnv:
    """
    OpenEnv-compatible API Debugging Environment.

    Seven tasks across three difficulties:
        Easy   — timeout_debug
        Medium — auth_failure, rate_limit_429, db_timeout, ssl_cert_expiry
        Hard   — deployment_500, cascading_failure

    Advanced features
    -----------------
    noise_level : float [0, 1]
        Inject red-herring logs, misleading hints, and latency jitter.
    step_penalty : float
        Deducted from reward when an action returns 0 reward (wasted step).
    destructive_penalty : float
        Extra deduction when a destructive action (restart/rollback) is used
        incorrectly (on top of step_penalty).

    Interface
    ---------
    reset(task_name) -> Observation
    step(action)     -> StepResult
    state()          -> Observation
    final_score()    -> float
    """

    def __init__(
        self,
        seed: int = 42,
        noise_level: float = 0.0,
        step_penalty: float = 0.0,
        destructive_penalty: float = 0.0,
    ) -> None:
        self.seed = seed
        self.noise_level = noise_level
        self.step_penalty = step_penalty
        self.destructive_penalty = destructive_penalty

        self.current_task: Optional[str] = None
        self.total_reward: float = 0.0
        self.partial_fix: bool = False

        self._stage: int = 0
        self._done: bool = False
        self._steps: int = 0
        self._current_obs: Optional[Observation] = None
        self._task_data: Optional[dict] = None
        self._noise = NoiseInjector(noise_level=noise_level, seed=seed)

    # ------------------------------------------------------------------
    # OpenEnv public interface
    # ------------------------------------------------------------------

    def reset(self, task_name: str) -> Observation:
        """Initialize the environment for the given task and return the initial observation."""
        if task_name not in TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. Available: {list(TASKS.keys())}"
            )
        self.current_task = task_name
        self._task_data = TASKS[task_name]
        self._stage = 0
        self._done = False
        self._steps = 0
        self.total_reward = 0.0
        self.partial_fix = False
        self._noise.reset(self.seed)
        self._current_obs = self._noise.inject(
            self._task_data["initial_observation"].model_copy(deep=True)
        )
        return self._current_obs

    def step(self, action: str) -> StepResult:
        """
        Execute an action string and return a StepResult.

        Penalties are applied when:
            - An action produces zero reward (step_penalty)
            - A destructive action is misapplied (destructive_penalty added on top)
        """
        if self._done:
            raise RuntimeError("Episode has ended. Call reset() to start a new episode.")
        if self._current_obs is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        action = action.strip().lower()

        if action not in VALID_ACTIONS:
            return StepResult(
                observation=self._current_obs,
                reward=0.0,
                done=False,
                info={
                    "error": f"Invalid action '{action}'.",
                    "valid_actions": VALID_ACTIONS,
                },
            )

        self._steps += 1
        reward, done = self._dispatch(action)

        # --- Penalty shaping ---
        if reward == 0.0:
            reward -= self.step_penalty
            if action in DESTRUCTIVE_ACTIONS:
                reward -= self.destructive_penalty
            reward = round(reward, 4)

        self.total_reward = round(self.total_reward + reward, 4)

        max_steps = self._task_data["config"].max_steps
        if self._steps >= max_steps and not done:
            done = True

        self._done = done

        # Apply noise to outgoing observation
        self._current_obs = self._noise.inject(self._current_obs)

        return StepResult(
            observation=self._current_obs,
            reward=reward,
            done=done,
            info={
                "task": self.current_task,
                "stage": self._stage,
                "steps": self._steps,
                "cumulative_reward": self.total_reward,
            },
        )

    def state(self) -> Observation:
        """Return the current observation without advancing the episode."""
        if self._current_obs is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._current_obs

    def final_score(self) -> float:
        """Return the graded final score for the current episode."""
        return grade(self)

    # ------------------------------------------------------------------
    # Action dispatch router
    # ------------------------------------------------------------------

    def _dispatch(self, action: str) -> Tuple[float, bool]:
        handlers = {
            "timeout_debug":     self._step_easy_timeout,
            "auth_failure":      self._step_medium_auth,
            "rate_limit_429":    self._step_medium_rate_limit,
            "db_timeout":        self._step_medium_db_timeout,
            "ssl_cert_expiry":   self._step_medium_ssl,
            "deployment_500":    self._step_hard_deployment,
            "cascading_failure": self._step_hard_cascade,
        }
        handler = handlers.get(self.current_task)
        return handler(action) if handler else (0.0, False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _stage_obs(self, stage: int) -> Observation:
        return self._task_data["stage_observations"][stage].model_copy(deep=True)

    def _append_log(self, line: str) -> Observation:
        d = self._current_obs.model_dump()
        d["logs"] = d["logs"] + line
        return Observation(**d)

    def _wrong_action(self, action: str, hint: str, status: int, error: str, latency: int) -> Tuple[float, bool]:
        self._current_obs = self._append_log(
            f"[WARN]  Action '{action}' had no effect — {hint}\n"
        )
        self._current_obs = Observation(
            **{**self._current_obs.model_dump(), "hints": [hint]},
        )
        return 0.0, False

    # ------------------------------------------------------------------
    # Easy — timeout_debug
    # ------------------------------------------------------------------

    def _step_easy_timeout(self, action: str) -> Tuple[float, bool]:
        if action == "increase_timeout":
            self._current_obs = Observation(
                logs=(
                    self._current_obs.logs
                    + "2024-01-15 14:24:00 [INFO]  Gateway timeout raised to 60000ms\n"
                    "2024-01-15 14:24:01 [INFO]  Request retried with updated config\n"
                    "2024-01-15 14:24:01 [INFO]  Upstream responded in 2100ms\n"
                    "2024-01-15 14:24:01 [INFO]  Request succeeded: 200 OK\n"
                ),
                status_code=200, latency_ms=2100, error="",
                hints=["Timeout threshold raised. Service is healthy."],
            )
            return 1.0, True

        if action == "retry_request":
            self.partial_fix = True
            self._current_obs = Observation(
                logs=(
                    self._current_obs.logs
                    + "2024-01-15 14:24:05 [INFO]  Manual retry triggered\n"
                    "2024-01-15 14:24:35 [ERROR] Retry also timed out after 30000ms\n"
                    "2024-01-15 14:24:35 [WARN]  Root cause (timeout threshold) not addressed\n"
                ),
                status_code=504, latency_ms=30000, error="TimeoutError",
                hints=["Retry did not resolve root cause. Consider increasing the timeout."],
            )
            return 0.5, True

        self._current_obs = self._append_log(
            f"2024-01-15 14:24:10 [WARN]  '{action}' had no effect on the timeout.\n"
        )
        self._current_obs = Observation(
            **{**self._current_obs.model_dump(),
               "hints": ["Timeout persists. Try 'increase_timeout' or 'retry_request'."]},
        )
        return 0.0, False

    # ------------------------------------------------------------------
    # Medium — auth_failure
    # ------------------------------------------------------------------

    def _step_medium_auth(self, action: str) -> Tuple[float, bool]:
        if self._stage == 0:
            if action == "check_logs":
                self._stage = 1
                self._current_obs = self._stage_obs(1)
                return 0.3, False
            if action == "fix_auth_token":
                self._current_obs = Observation(
                    logs=(self._current_obs.logs
                          + "2024-01-15 09:16:00 [INFO]  Auth token refreshed\n"
                          "2024-01-15 09:16:01 [INFO]  Authentication successful — 200 OK\n"),
                    status_code=200, latency_ms=50, error="",
                    hints=["Token refreshed (investigation skipped). Service restored."],
                )
                return 0.6, True
            self._current_obs = self._append_log(
                f"2024-01-15 09:15:30 [WARN]  '{action}' did not address the auth failure.\n"
            )
            self._current_obs = Observation(
                **{**self._current_obs.model_dump(),
                   "hints": ["Auth still failing. Try 'check_logs' or 'fix_auth_token'."]},
            )
            return 0.0, False

        if self._stage == 1:
            if action == "fix_auth_token":
                self._current_obs = Observation(
                    logs=(self._current_obs.logs
                          + "2024-01-15 09:16:00 [INFO]  Token refreshed from secrets manager\n"
                          "2024-01-15 09:16:00 [INFO]  Env cache invalidated\n"
                          "2024-01-15 09:16:01 [INFO]  Authentication successful — 200 OK\n"),
                    status_code=200, latency_ms=50, error="",
                    hints=["Full fix applied: token refreshed + cache cleared."],
                )
                return 0.7, True
            self._current_obs = self._append_log(
                f"2024-01-15 09:15:45 [WARN]  '{action}' did not fix the stale token.\n"
            )
            self._current_obs = Observation(
                **{**self._current_obs.model_dump(),
                   "hints": ["Stale token confirmed. Apply 'fix_auth_token'."]},
            )
            return 0.0, False

        return 0.0, False

    # ------------------------------------------------------------------
    # Medium — rate_limit_429
    # ------------------------------------------------------------------

    def _step_medium_rate_limit(self, action: str) -> Tuple[float, bool]:
        if self._stage == 0:
            if action == "check_logs":
                self._stage = 1
                self._current_obs = self._stage_obs(1)
                return 0.3, False
            if action == "implement_backoff":
                self._current_obs = Observation(
                    logs=(self._current_obs.logs
                          + "2024-01-15 11:31:00 [INFO]  Exponential backoff applied to client app-42\n"
                          "2024-01-15 11:31:01 [INFO]  Request rate normalized to 800 req/min\n"
                          "2024-01-15 11:31:01 [INFO]  Request succeeded: 200 OK\n"),
                    status_code=200, latency_ms=80, error="",
                    hints=["Backoff applied (investigation skipped). Rate limit resolved."],
                )
                return 0.6, True
            self._current_obs = self._append_log(
                f"2024-01-15 11:30:15 [WARN]  '{action}' did not address rate limiting.\n"
            )
            self._current_obs = Observation(
                **{**self._current_obs.model_dump(),
                   "hints": ["429 persists. Try 'check_logs' or 'implement_backoff'."]},
            )
            return 0.0, False

        if self._stage == 1:
            if action == "implement_backoff":
                self._current_obs = Observation(
                    logs=(self._current_obs.logs
                          + "2024-01-15 11:31:00 [INFO]  Exponential backoff configured for app-42\n"
                          "2024-01-15 11:31:00 [INFO]  Burst window throttled to 500 req/min\n"
                          "2024-01-15 11:31:01 [INFO]  Request succeeded: 200 OK\n"),
                    status_code=200, latency_ms=80, error="",
                    hints=["Full fix: burst pattern resolved with backoff. Quota restored."],
                )
                return 0.7, True
            self._current_obs = self._append_log(
                f"2024-01-15 11:30:30 [WARN]  '{action}' did not fix the burst pattern.\n"
            )
            self._current_obs = Observation(
                **{**self._current_obs.model_dump(),
                   "hints": ["Burst pattern identified. Apply 'implement_backoff'."]},
            )
            return 0.0, False

        return 0.0, False

    # ------------------------------------------------------------------
    # Medium — db_timeout
    # ------------------------------------------------------------------

    def _step_medium_db_timeout(self, action: str) -> Tuple[float, bool]:
        if self._stage == 0:
            if action == "check_logs":
                self._stage = 1
                self._current_obs = self._stage_obs(1)
                return 0.4, False
            if action == "restart_service":
                self._current_obs = Observation(
                    logs=(self._current_obs.logs
                          + "2024-01-15 13:06:00 [INFO]  Service restarted — connection pool reset\n"
                          "2024-01-15 13:06:01 [INFO]  All DB connections reclaimed\n"
                          "2024-01-15 13:06:01 [INFO]  Request succeeded: 200 OK\n"),
                    status_code=200, latency_ms=210, error="",
                    hints=["Restart reset pool (investigation skipped). Service restored."],
                )
                return 0.5, True
            self._current_obs = self._append_log(
                f"2024-01-15 13:05:50 [WARN]  '{action}' did not fix the DB pool exhaustion.\n"
            )
            self._current_obs = Observation(
                **{**self._current_obs.model_dump(),
                   "hints": ["DB pool still exhausted. Try 'check_logs' or 'restart_service'."]},
            )
            return 0.0, False

        if self._stage == 1:
            if action == "restart_service":
                self._current_obs = Observation(
                    logs=(self._current_obs.logs
                          + "2024-01-15 13:06:00 [INFO]  Service restarted — pool reset, blocking query killed\n"
                          "2024-01-15 13:06:01 [INFO]  Pool stats — active: 0/50, idle: 50\n"
                          "2024-01-15 13:06:01 [INFO]  Request succeeded: 200 OK\n"),
                    status_code=200, latency_ms=210, error="",
                    hints=["Full fix: pool reset, blocking query cleared. All connections healthy."],
                )
                return 0.6, True
            self._current_obs = self._append_log(
                f"2024-01-15 13:06:05 [WARN]  '{action}' did not reset the connection pool.\n"
            )
            self._current_obs = Observation(
                **{**self._current_obs.model_dump(),
                   "hints": ["Blocking query identified. Use 'restart_service' to reset the pool."]},
            )
            return 0.0, False

        return 0.0, False

    # ------------------------------------------------------------------
    # Medium — ssl_cert_expiry
    # ------------------------------------------------------------------

    def _step_medium_ssl(self, action: str) -> Tuple[float, bool]:
        if self._stage == 0:
            if action == "check_logs":
                self._stage = 1
                self._current_obs = self._stage_obs(1)
                return 0.3, False
            if action == "renew_certificate":
                self._current_obs = Observation(
                    logs=(self._current_obs.logs
                          + "2024-01-15 00:02:00 [INFO]  TLS certificate renewed (valid 90 days)\n"
                          "2024-01-15 00:02:01 [INFO]  TLS handshake successful\n"
                          "2024-01-15 00:02:01 [INFO]  Request succeeded: 200 OK\n"),
                    status_code=200, latency_ms=55, error="",
                    hints=["Certificate renewed (investigation skipped). HTTPS restored."],
                )
                return 0.6, True
            self._current_obs = self._append_log(
                f"2024-01-15 00:01:30 [WARN]  '{action}' did not resolve the SSL error.\n"
            )
            self._current_obs = Observation(
                **{**self._current_obs.model_dump(),
                   "hints": ["SSL handshake still failing. Try 'check_logs' or 'renew_certificate'."]},
            )
            return 0.0, False

        if self._stage == 1:
            if action == "renew_certificate":
                self._current_obs = Observation(
                    logs=(self._current_obs.logs
                          + "2024-01-15 00:02:00 [INFO]  Certificate renewed via ACME — valid 90 days\n"
                          "2024-01-15 00:02:00 [INFO]  Auto-renewal re-enabled to prevent recurrence\n"
                          "2024-01-15 00:02:01 [INFO]  TLS handshake successful — 200 OK\n"),
                    status_code=200, latency_ms=55, error="",
                    hints=["Full fix: cert renewed + auto-renewal re-enabled. HTTPS fully restored."],
                )
                return 0.7, True
            self._current_obs = self._append_log(
                f"2024-01-15 00:01:50 [WARN]  '{action}' did not renew the certificate.\n"
            )
            self._current_obs = Observation(
                **{**self._current_obs.model_dump(),
                   "hints": ["Cert expiry confirmed. Apply 'renew_certificate'."]},
            )
            return 0.0, False

        return 0.0, False

    # ------------------------------------------------------------------
    # Hard — deployment_500
    # ------------------------------------------------------------------

    def _step_hard_deployment(self, action: str) -> Tuple[float, bool]:
        if self._stage == 0:
            if action == "check_logs":
                self._stage = 1
                self._current_obs = self._stage_obs(1)
                return 0.3, False
            self._current_obs = self._append_log(
                f"2024-01-15 16:43:00 [WARN]  '{action}' did not surface root cause.\n"
            )
            self._current_obs = Observation(
                **{**self._current_obs.model_dump(),
                   "hints": ["Widespread 500 errors. Start by checking the logs."]},
            )
            return 0.0, False

        if self._stage == 1:
            if action == "analyze_deployment":
                self._stage = 2
                self._current_obs = self._stage_obs(2)
                return 0.3, False
            if action == "restart_service":
                self._current_obs = self._append_log(
                    "2024-01-15 16:44:00 [INFO]  Service restarted\n"
                    "2024-01-15 16:44:05 [ERROR] Same NullPointerException after restart — faulty code still live\n"
                )
                self._current_obs = Observation(
                    **{**self._current_obs.model_dump(),
                       "hints": ["Restart reloaded faulty v2.4.2. Analyze deployment for root cause."]},
                )
                return 0.0, False
            self._current_obs = self._append_log(
                f"2024-01-15 16:43:30 [WARN]  '{action}' insufficient. Use 'analyze_deployment'.\n"
            )
            self._current_obs = Observation(
                **{**self._current_obs.model_dump(),
                   "hints": ["Errors correlate with v2.4.2 deploy. Analyze the deployment."]},
            )
            return 0.0, False

        if self._stage == 2:
            if action == "rollback_version":
                self._current_obs = Observation(
                    logs=(self._current_obs.logs
                          + "2024-01-15 16:45:00 [INFO]  Rollback: v2.4.2 → v2.4.1\n"
                          "2024-01-15 16:45:25 [INFO]  Rollback completed\n"
                          "2024-01-15 16:45:26 [INFO]  Health check: 200 OK — error rate 0%\n"),
                    status_code=200, latency_ms=110, error="",
                    hints=["Rollback successful. Service fully restored on v2.4.1."],
                )
                return 0.4, True
            if action == "restart_service":
                self._current_obs = self._append_log(
                    "2024-01-15 16:45:00 [INFO]  Restarted (still on v2.4.2)\n"
                    "2024-01-15 16:45:05 [ERROR] NullPointerException persists\n"
                )
                self._current_obs = Observation(
                    **{**self._current_obs.model_dump(),
                       "hints": ["Restart reloaded faulty code. Use 'rollback_version'."]},
                )
                return 0.0, False
            self._current_obs = self._append_log(
                f"2024-01-15 16:44:30 [WARN]  '{action}' cannot fix the deployment bug.\n"
            )
            self._current_obs = Observation(
                **{**self._current_obs.model_dump(),
                   "hints": ["Root cause confirmed. Use 'rollback_version' to restore v2.4.1."]},
            )
            return 0.0, False

        return 0.0, False

    # ------------------------------------------------------------------
    # Hard — cascading_failure
    # ------------------------------------------------------------------

    def _step_hard_cascade(self, action: str) -> Tuple[float, bool]:
        if self._stage == 0:
            if action == "check_logs":
                self._stage = 1
                self._current_obs = self._stage_obs(1)
                return 0.3, False
            self._current_obs = self._append_log(
                f"2024-01-15 18:10:20 [WARN]  '{action}' did not surface cascade root cause.\n"
            )
            self._current_obs = Observation(
                **{**self._current_obs.model_dump(),
                   "hints": ["Multiple services returning 503. Check logs first."]},
            )
            return 0.0, False

        if self._stage == 1:
            if action == "analyze_deployment":
                self._stage = 2
                self._current_obs = self._stage_obs(2)
                return 0.3, False
            if action == "restart_service":
                # restart of stock-svc fails because env var still missing
                self._current_obs = self._append_log(
                    "2024-01-15 18:11:00 [INFO]  stock-svc restart attempted\n"
                    "2024-01-15 18:11:01 [ERROR] stock-svc still failing: REDIS_URL not set\n"
                    "2024-01-15 18:11:01 [WARN]  Restart loop continues — cascade unresolved\n"
                )
                self._current_obs = Observation(
                    **{**self._current_obs.model_dump(),
                       "hints": ["Restart failed — underlying deployment issue not fixed. Analyze the deployment."]},
                )
                return 0.0, False
            self._current_obs = self._append_log(
                f"2024-01-15 18:10:40 [WARN]  '{action}' insufficient. Use 'analyze_deployment'.\n"
            )
            self._current_obs = Observation(
                **{**self._current_obs.model_dump(),
                   "hints": ["stock-svc deploy correlation found. Analyze the deployment."]},
            )
            return 0.0, False

        if self._stage == 2:
            if action == "rollback_version":
                self._current_obs = Observation(
                    logs=(self._current_obs.logs
                          + "2024-01-15 18:12:00 [INFO]  stock-svc rollback: v1.2.3 → v1.2.2\n"
                          "2024-01-15 18:12:20 [INFO]  stock-svc healthy — registered with service discovery\n"
                          "2024-01-15 18:12:21 [INFO]  inventory-svc reconnected to stock-svc\n"
                          "2024-01-15 18:12:22 [INFO]  api-gateway health check: 200 OK\n"
                          "2024-01-15 18:12:22 [INFO]  Cascade fully resolved — error rate 0%\n"),
                    status_code=200, latency_ms=180, error="",
                    hints=["stock-svc v1.2.2 restored. Full cascade resolved."],
                )
                return 0.4, True
            if action == "restart_service":
                self._current_obs = self._append_log(
                    "2024-01-15 18:12:00 [ERROR] stock-svc restart failed again: REDIS_URL still absent\n"
                )
                self._current_obs = Observation(
                    **{**self._current_obs.model_dump(),
                       "hints": ["Restart still fails. Use 'rollback_version' to restore v1.2.2."]},
                )
                return 0.0, False
            self._current_obs = self._append_log(
                f"2024-01-15 18:11:30 [WARN]  '{action}' cannot resolve the cascade.\n"
            )
            self._current_obs = Observation(
                **{**self._current_obs.model_dump(),
                   "hints": ["Root cause confirmed. Use 'rollback_version' to restore stock-svc v1.2.2."]},
            )
            return 0.0, False

        return 0.0, False
