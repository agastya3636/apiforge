"""
Observation noise injection for the API Debugging Environment.

Controlled by noise_level in [0.0, 1.0]:
  0.0  — deterministic, no noise (default)
  0.5  — moderate: red-herring log lines, latency jitter, occasional misleading hints
  1.0  — maximum: heavy red herrings, significant jitter, frequent misleading hints
"""

from __future__ import annotations

import random
from typing import List

from env.models import Observation

# ---------------------------------------------------------------------------
# Red-herring log lines injected before or after real logs
# ---------------------------------------------------------------------------
RED_HERRING_LOGS: List[str] = [
    "2024-01-15 14:20:00 [WARN]  Memory usage at 78% — within acceptable range\n",
    "2024-01-15 14:20:01 [INFO]  Scheduled cache invalidation completed (0 keys)\n",
    "2024-01-15 14:20:02 [DEBUG] Health-check ping received from load balancer\n",
    "2024-01-15 14:20:03 [INFO]  Background job queue depth: 14 (normal)\n",
    "2024-01-15 14:20:04 [WARN]  Brief CPU spike (2 s) — non-critical, returned to baseline\n",
    "2024-01-15 14:20:05 [INFO]  TLS session resumed for client 10.0.0.45\n",
    "2024-01-15 14:20:06 [DEBUG] Config hot-reload triggered — no changes detected\n",
    "2024-01-15 14:20:07 [INFO]  Metrics scraped by Prometheus\n",
]

# ---------------------------------------------------------------------------
# Misleading hints injected alongside real hints
# ---------------------------------------------------------------------------
MISLEADING_HINTS: List[str] = [
    "High memory usage may be causing instability.",
    "Cache invalidation could be interfering with request handling.",
    "Background jobs may be saturating I/O bandwidth.",
    "CPU spike could indicate a compute bottleneck.",
    "TLS renegotiation overhead may be increasing latency.",
]


class NoiseInjector:
    """
    Injects configurable noise into Observation objects.

    Parameters
    ----------
    noise_level : float
        Noise intensity in [0.0, 1.0]. 0.0 = no noise.
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(self, noise_level: float = 0.0, seed: int = 42) -> None:
        if not 0.0 <= noise_level <= 1.0:
            raise ValueError("noise_level must be in [0.0, 1.0]")
        self.noise_level = noise_level
        self._rng = random.Random(seed)

    def inject(self, obs: Observation) -> Observation:
        """Return a (possibly noisy) copy of the observation."""
        if self.noise_level == 0.0:
            return obs

        obs_dict = obs.model_dump()

        # --- Red-herring log lines -------------------------------------------
        if self._rng.random() < self.noise_level:
            n_lines = self._rng.randint(1, max(1, int(self.noise_level * 3)))
            chosen = self._rng.sample(RED_HERRING_LOGS, min(n_lines, len(RED_HERRING_LOGS)))
            obs_dict["logs"] = "".join(chosen) + obs_dict["logs"]

        # --- Misleading hints ------------------------------------------------
        if self._rng.random() < self.noise_level * 0.6:
            misleading = self._rng.choice(MISLEADING_HINTS)
            obs_dict["hints"] = [misleading] + obs_dict["hints"]

        # --- Latency jitter (±25 %) -----------------------------------------
        if self._rng.random() < self.noise_level:
            jitter_pct = self._rng.uniform(-0.25, 0.25) * self.noise_level
            obs_dict["latency_ms"] = max(
                1, int(obs_dict["latency_ms"] * (1 + jitter_pct))
            )

        return Observation(**obs_dict)

    def reset(self, seed: int | None = None) -> None:
        """Re-seed the RNG (call before each episode for reproducibility)."""
        self._rng = random.Random(seed if seed is not None else self._rng.randint(0, 2**31))
