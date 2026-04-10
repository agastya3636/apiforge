from __future__ import annotations

from typing import Dict

from env.models import Observation, TaskConfig

# ---------------------------------------------------------------------------
# Task registry
# Each task entry contains:
#   config              — TaskConfig metadata
#   initial_observation — Observation shown at episode start
#   stage_observations  — Observations revealed after multi-step actions
# ---------------------------------------------------------------------------

TASKS: Dict[str, dict] = {
    # -----------------------------------------------------------------------
    # EASY — 504 Timeout
    # -----------------------------------------------------------------------
    "timeout_debug": {
        "config": TaskConfig(
            name="timeout_debug",
            difficulty="easy",
            description=(
                "An API endpoint is returning 504 Gateway Timeout errors. "
                "Diagnose and fix the timeout issue to restore service."
            ),
            max_steps=5,
        ),
        "initial_observation": Observation(
            logs=(
                "2024-01-15 14:23:01 [INFO]  Request received: GET /api/v1/users\n"
                "2024-01-15 14:23:01 [INFO]  Forwarding to upstream service (host: svc-users:8080)\n"
                "2024-01-15 14:23:31 [ERROR] Connection timed out after 30000ms\n"
                "2024-01-15 14:23:31 [ERROR] TimeoutError: upstream service did not respond\n"
                "2024-01-15 14:23:31 [ERROR] Request failed with status 504\n"
                "2024-01-15 14:23:31 [WARN]  Retry attempt 1/3 — timed out\n"
                "2024-01-15 14:23:31 [WARN]  Retry attempt 2/3 — timed out\n"
                "2024-01-15 14:23:31 [WARN]  Retry attempt 3/3 — timed out\n"
            ),
            status_code=504,
            latency_ms=30000,
            error="TimeoutError",
            hints=[],
        ),
        "stage_observations": {},  # single-step task — no intermediate stages
    },

    # -----------------------------------------------------------------------
    # MEDIUM — 401 Auth Failure
    # -----------------------------------------------------------------------
    "auth_failure": {
        "config": TaskConfig(
            name="auth_failure",
            difficulty="medium",
            description=(
                "API requests are being rejected with 401 Unauthorized. "
                "Investigate the authentication failure and restore access."
            ),
            max_steps=6,
        ),
        "initial_observation": Observation(
            logs=(
                "2024-01-15 09:15:22 [INFO]  Request received: POST /api/v1/data\n"
                "2024-01-15 09:15:22 [INFO]  Extracting Bearer token from Authorization header\n"
                "2024-01-15 09:15:22 [INFO]  Validating JWT with public key...\n"
                "2024-01-15 09:15:22 [ERROR] Token validation failed: JWT signature mismatch\n"
                "2024-01-15 09:15:22 [ERROR] AuthenticationError: Invalid or expired token\n"
                "2024-01-15 09:15:22 [ERROR] Request rejected with status 401\n"
            ),
            status_code=401,
            latency_ms=45,
            error="AuthenticationError",
            hints=[],
        ),
        "stage_observations": {
            # Revealed after check_logs
            1: Observation(
                logs=(
                    "2024-01-15 09:15:22 [INFO]  Request received: POST /api/v1/data\n"
                    "2024-01-15 09:15:22 [ERROR] Token validation failed: JWT signature mismatch\n"
                    "2024-01-15 09:15:22 [ERROR] AuthenticationError: Invalid or expired token\n"
                    "2024-01-15 09:15:22 [ERROR] Request rejected with status 401\n"
                    "--- Extended Log Analysis ---\n"
                    "2024-01-15 09:00:00 [INFO]  API_SECRET_KEY rotated by deployment pipeline\n"
                    "2024-01-15 09:00:01 [WARN]  Service reloaded but env cache was not invalidated\n"
                    "2024-01-15 09:00:02 [WARN]  Service still signing tokens with old key\n"
                    "2024-01-15 09:01:00 [ERROR] All subsequent auth requests failing since key rotation\n"
                ),
                status_code=401,
                latency_ms=45,
                error="AuthenticationError",
                hints=[
                    "API_SECRET_KEY was rotated at 09:00:00.",
                    "Service env cache has stale token. Fix the auth token to restore access.",
                ],
            ),
        },
    },

    # -----------------------------------------------------------------------
    # HARD — Deployment-induced 500
    # -----------------------------------------------------------------------
    "deployment_500": {
        "config": TaskConfig(
            name="deployment_500",
            difficulty="hard",
            description=(
                "A recent deployment has caused widespread 500 Internal Server Errors "
                "across all endpoints. Identify the faulty release and roll it back."
            ),
            max_steps=8,
        ),
        "initial_observation": Observation(
            logs=(
                "2024-01-15 16:42:11 [INFO]  Request received: POST /api/v1/process\n"
                "2024-01-15 16:42:11 [INFO]  Loading DataProcessor configuration...\n"
                "2024-01-15 16:42:11 [ERROR] NullPointerException in DataProcessor.process() at line 247\n"
                "2024-01-15 16:42:11 [ERROR] Caused by: Missing required field 'schema_version' in config\n"
                "2024-01-15 16:42:11 [ERROR] InternalServerError: Unhandled exception in request handler\n"
                "2024-01-15 16:42:11 [ERROR] Request failed with status 500\n"
                "2024-01-15 16:42:12 [ERROR] Same error on: GET /api/v1/health\n"
                "2024-01-15 16:42:13 [ERROR] Same error on: GET /api/v1/metrics\n"
                "2024-01-15 16:42:14 [ERROR] Error rate: 100% — all endpoints affected\n"
            ),
            status_code=500,
            latency_ms=120,
            error="InternalServerError",
            hints=[],
        ),
        "stage_observations": {
            # Revealed after check_logs
            1: Observation(
                logs=(
                    "2024-01-15 16:42:11 [ERROR] NullPointerException in DataProcessor.process() at line 247\n"
                    "2024-01-15 16:42:11 [ERROR] Caused by: Missing required field 'schema_version'\n"
                    "2024-01-15 16:42:11 [ERROR] InternalServerError: Request failed with status 500\n"
                    "--- Extended Log Analysis ---\n"
                    "2024-01-15 16:30:00 [INFO]  Deployment pipeline triggered: v2.4.1 → v2.4.2\n"
                    "2024-01-15 16:35:00 [INFO]  Deployment completed: v2.4.2 is now live\n"
                    "2024-01-15 16:35:01 [ERROR] First NullPointerException reported 1 s after deploy\n"
                    "2024-01-15 16:42:00 [ERROR] Error rate: 100% across all endpoints\n"
                    "2024-01-15 16:42:00 [INFO]  Previous stable version: v2.4.1\n"
                ),
                status_code=500,
                latency_ms=120,
                error="InternalServerError",
                hints=[
                    "Errors began immediately after deploying v2.4.2 at 16:35:00.",
                    "Analyze the deployment to confirm root cause.",
                ],
            ),
            # Revealed after analyze_deployment
            2: Observation(
                logs=(
                    "2024-01-15 16:42:11 [ERROR] NullPointerException in DataProcessor.process() at line 247\n"
                    "2024-01-15 16:42:11 [ERROR] Caused by: Missing required field 'schema_version'\n"
                    "--- Deployment Analysis Report ---\n"
                    "2024-01-15 16:30:00 [INFO]  v2.4.2 diff: DataProcessor config schema updated\n"
                    "2024-01-15 16:30:00 [INFO]  New mandatory field 'schema_version' added to config model\n"
                    "2024-01-15 16:30:00 [WARN]  Backward compatibility NOT maintained\n"
                    "2024-01-15 16:30:00 [WARN]  All existing config files missing 'schema_version' field\n"
                    "2024-01-15 16:30:00 [WARN]  Migration script was NOT included in this release\n"
                    ">>> ROOT CAUSE: v2.4.2 introduced a breaking schema change without migration\n"
                    ">>> RECOMMENDED ACTION: Rollback to v2.4.1 immediately\n"
                ),
                status_code=500,
                latency_ms=120,
                error="InternalServerError",
                hints=[
                    "v2.4.2 broke backward compatibility — missing migration script.",
                    "Rollback to v2.4.1 will immediately restore service.",
                ],
            ),
        },
    },
    # -----------------------------------------------------------------------
    # MEDIUM — 429 Rate Limit
    # -----------------------------------------------------------------------
    "rate_limit_429": {
        "config": TaskConfig(
            name="rate_limit_429",
            difficulty="medium",
            description=(
                "API is returning 429 Too Many Requests. The client is sending "
                "bursts without exponential backoff. Investigate and apply backoff."
            ),
            max_steps=6,
        ),
        "initial_observation": Observation(
            logs=(
                "2024-01-15 11:30:00 [INFO]  Request received: GET /api/v1/feed\n"
                "2024-01-15 11:30:00 [INFO]  Checking rate limit quota for client_id=app-42\n"
                "2024-01-15 11:30:00 [ERROR] Rate limit exceeded: 1200 req/min (limit: 1000)\n"
                "2024-01-15 11:30:00 [ERROR] RateLimitExceeded: Too many requests from client\n"
                "2024-01-15 11:30:00 [ERROR] Request rejected with status 429\n"
                "2024-01-15 11:30:00 [WARN]  Retry-After header sent: 60 seconds\n"
            ),
            status_code=429,
            latency_ms=5,
            error="RateLimitExceeded",
            hints=[],
        ),
        "stage_observations": {
            1: Observation(
                logs=(
                    "2024-01-15 11:30:00 [ERROR] Rate limit exceeded: 1200 req/min (limit: 1000)\n"
                    "2024-01-15 11:30:00 [ERROR] RateLimitExceeded: Too many requests\n"
                    "--- Extended Log Analysis ---\n"
                    "2024-01-15 11:29:00 [INFO]  Client app-42 sending 20 req/s in burst mode\n"
                    "2024-01-15 11:29:30 [WARN]  No backoff strategy detected in client requests\n"
                    "2024-01-15 11:29:45 [WARN]  Quota will reset in 60 s — bursting continues\n"
                    "2024-01-15 11:30:00 [ERROR] Quota exhausted after sustained burst\n"
                ),
                status_code=429,
                latency_ms=5,
                error="RateLimitExceeded",
                hints=[
                    "Client app-42 is bursting without backoff.",
                    "Implement exponential backoff to stay within quota.",
                ],
            ),
        },
    },

    # -----------------------------------------------------------------------
    # MEDIUM — Database Connection Timeout
    # -----------------------------------------------------------------------
    "db_timeout": {
        "config": TaskConfig(
            name="db_timeout",
            difficulty="medium",
            description=(
                "API returns 500 errors caused by an exhausted database connection pool. "
                "Diagnose the DB issue and restore connectivity."
            ),
            max_steps=6,
        ),
        "initial_observation": Observation(
            logs=(
                "2024-01-15 13:05:11 [INFO]  Request received: GET /api/v1/reports\n"
                "2024-01-15 13:05:11 [INFO]  Acquiring DB connection from pool...\n"
                "2024-01-15 13:05:41 [ERROR] Pool timeout: no connection available after 30000ms\n"
                "2024-01-15 13:05:41 [ERROR] DatabaseConnectionTimeout: connection pool exhausted\n"
                "2024-01-15 13:05:41 [ERROR] Pool stats — active: 50/50, idle: 0, waiting: 143\n"
                "2024-01-15 13:05:41 [ERROR] Request failed with status 500\n"
            ),
            status_code=500,
            latency_ms=30041,
            error="DatabaseConnectionTimeout",
            hints=[],
        ),
        "stage_observations": {
            1: Observation(
                logs=(
                    "2024-01-15 13:05:41 [ERROR] Pool timeout: no connection available after 30000ms\n"
                    "2024-01-15 13:05:41 [ERROR] Pool stats — active: 50/50, idle: 0, waiting: 143\n"
                    "--- Extended Log Analysis ---\n"
                    "2024-01-15 12:55:00 [WARN]  Long-running query detected (pid=1742, 8 min 20 s)\n"
                    "2024-01-15 12:58:00 [WARN]  Multiple queries blocking on pid=1742\n"
                    "2024-01-15 13:00:00 [ERROR] Connection pool saturation reached 90%\n"
                    "2024-01-15 13:03:00 [ERROR] Pool fully saturated — new requests queuing\n"
                    "2024-01-15 13:05:00 [ERROR] Queue overflow — rejecting connections\n"
                ),
                status_code=500,
                latency_ms=30041,
                error="DatabaseConnectionTimeout",
                hints=[
                    "Long-running query (pid=1742) is holding connections.",
                    "Restarting the service will reset the connection pool and clear blocked queries.",
                ],
            ),
        },
    },

    # -----------------------------------------------------------------------
    # MEDIUM — SSL Certificate Expiry
    # -----------------------------------------------------------------------
    "ssl_cert_expiry": {
        "config": TaskConfig(
            name="ssl_cert_expiry",
            difficulty="medium",
            description=(
                "API is returning 503 errors due to an expired TLS certificate. "
                "Diagnose the SSL handshake failure and renew the certificate."
            ),
            max_steps=6,
        ),
        "initial_observation": Observation(
            logs=(
                "2024-01-15 00:01:05 [INFO]  Request received: GET /api/v1/status\n"
                "2024-01-15 00:01:05 [INFO]  Initiating TLS handshake with upstream\n"
                "2024-01-15 00:01:05 [ERROR] SSL handshake failed: certificate expired\n"
                "2024-01-15 00:01:05 [ERROR] SSLHandshakeError: certificate validity ended 2024-01-14\n"
                "2024-01-15 00:01:05 [ERROR] Request failed with status 503\n"
                "2024-01-15 00:01:05 [WARN]  All HTTPS upstream calls will fail until cert is renewed\n"
            ),
            status_code=503,
            latency_ms=10,
            error="SSLHandshakeError",
            hints=[],
        ),
        "stage_observations": {
            1: Observation(
                logs=(
                    "2024-01-15 00:01:05 [ERROR] SSL handshake failed: certificate expired\n"
                    "2024-01-15 00:01:05 [ERROR] SSLHandshakeError: cert validity ended 2024-01-14\n"
                    "--- Extended Log Analysis ---\n"
                    "2024-01-14 00:00:00 [WARN]  TLS certificate expires in 24 hours — renew soon\n"
                    "2024-01-14 12:00:00 [WARN]  TLS certificate expires in 12 hours\n"
                    "2024-01-14 23:00:00 [WARN]  TLS certificate expires in 1 hour\n"
                    "2024-01-15 00:00:00 [ERROR] TLS certificate expired — auto-renew not configured\n"
                ),
                status_code=503,
                latency_ms=10,
                error="SSLHandshakeError",
                hints=[
                    "Certificate expired at midnight — auto-renewal was not configured.",
                    "Renew the certificate to restore HTTPS connectivity.",
                ],
            ),
        },
    },

    # -----------------------------------------------------------------------
    # HARD — Cascading Microservice Failure
    # -----------------------------------------------------------------------
    "cascading_failure": {
        "config": TaskConfig(
            name="cascading_failure",
            difficulty="hard",
            description=(
                "A deployment to stock-svc (Service C) caused it to crash on startup. "
                "The failure is cascading: svc-C → inventory-svc → api-gateway, "
                "all returning 503. Identify the root service and roll back."
            ),
            max_steps=9,
        ),
        "initial_observation": Observation(
            logs=(
                "2024-01-15 18:10:05 [INFO]  Request received: POST /api/v1/order\n"
                "2024-01-15 18:10:05 [INFO]  Calling inventory-svc: GET /inventory/check\n"
                "2024-01-15 18:10:06 [ERROR] inventory-svc returned 503 Service Unavailable\n"
                "2024-01-15 18:10:06 [INFO]  inventory-svc upstream: GET /stock-svc/available\n"
                "2024-01-15 18:10:07 [ERROR] stock-svc unreachable: Connection refused (port 8083)\n"
                "2024-01-15 18:10:07 [ERROR] Cascade: stock-svc → inventory-svc → api-gateway\n"
                "2024-01-15 18:10:07 [ERROR] Request failed with status 503\n"
                "2024-01-15 18:10:07 [ERROR] All dependent endpoints failing\n"
            ),
            status_code=503,
            latency_ms=2050,
            error="CascadingServiceFailure",
            hints=[],
        ),
        "stage_observations": {
            # Revealed after check_logs
            1: Observation(
                logs=(
                    "2024-01-15 18:10:07 [ERROR] Cascade: stock-svc → inventory-svc → api-gateway\n"
                    "--- Extended Log Analysis ---\n"
                    "2024-01-15 18:05:00 [INFO]  Deployment triggered: stock-svc v1.2.2 → v1.2.3\n"
                    "2024-01-15 18:06:00 [ERROR] stock-svc health check failed after deploy\n"
                    "2024-01-15 18:06:01 [WARN]  stock-svc removed from service registry\n"
                    "2024-01-15 18:07:00 [ERROR] inventory-svc losing upstream — starting to 503\n"
                    "2024-01-15 18:09:00 [ERROR] api-gateway propagating 503 to all clients\n"
                ),
                status_code=503,
                latency_ms=2050,
                error="CascadingServiceFailure",
                hints=[
                    "stock-svc failed health check immediately after deploying v1.2.3.",
                    "Analyze the deployment to confirm root cause.",
                ],
            ),
            # Revealed after analyze_deployment
            2: Observation(
                logs=(
                    "2024-01-15 18:10:07 [ERROR] Cascade: stock-svc → inventory-svc → api-gateway\n"
                    "--- Deployment Analysis Report ---\n"
                    "2024-01-15 18:05:00 [INFO]  stock-svc v1.2.3 diff: added REDIS_URL dependency\n"
                    "2024-01-15 18:05:00 [ERROR] stock-svc startup failed: env var REDIS_URL not set\n"
                    "2024-01-15 18:05:01 [ERROR] Process exited with code 1 on first boot\n"
                    "2024-01-15 18:05:01 [WARN]  Restart loop: 5 attempts, all failed\n"
                    ">>> ROOT CAUSE: stock-svc v1.2.3 requires REDIS_URL which is absent in prod env\n"
                    ">>> RECOMMENDED ACTION: Rollback to v1.2.2 (stable, no Redis dependency)\n"
                ),
                status_code=503,
                latency_ms=2050,
                error="CascadingServiceFailure",
                hints=[
                    "stock-svc v1.2.3 crashes on startup — missing REDIS_URL env variable.",
                    "Rolling back to v1.2.2 will immediately restore the service chain.",
                ],
            ),
        },
    },
}

TASK_NAMES = list(TASKS.keys())
