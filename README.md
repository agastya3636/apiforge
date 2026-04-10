# API Debugging Environment

An **OpenEnv-compatible** reinforcement learning environment where an AI agent acts as a backend engineer diagnosing and resolving real-world API failures.

---

## Problem Description

The agent receives structured API telemetry (logs, status codes, latency, error types) and must issue string-based diagnostic/repair actions to resolve the underlying issue. Three difficulty levels are included, ranging from a simple timeout fix to a multi-step deployment rollback.

---

## Project Structure

```
apiforge/
├── env/
│   ├── __init__.py         # Package init, exports APIDebugEnv
│   ├── environment.py      # Core APIDebugEnv class (reset / step / state)
│   ├── tasks.py            # Task definitions & scenario state machines
│   ├── grader.py           # Per-task reward graders
│   └── models.py           # Pydantic models (Observation, StepResult, TaskConfig)
├── inference.py            # Deterministic baseline inference script
├── openenv.yaml            # OpenEnv manifest
├── Dockerfile              # Docker image (python:3.10-slim)
├── requirements.txt
└── README.md
```

---

## Action Space

| Action | Description |
|---|---|
| `check_logs` | Fetch and analyze extended application logs |
| `retry_request` | Manually retry the failing request |
| `fix_auth_token` | Refresh / rotate the authentication token |
| `increase_timeout` | Raise the gateway timeout threshold |
| `restart_service` | Restart the affected service process |
| `rollback_version` | Roll back to the previous stable deployment |
| `analyze_deployment` | Inspect the diff and metadata of the latest deployment |
| `do_nothing` | Take no action (zero reward) |

---

## Observation Space

Each step returns a JSON-serializable `Observation` object:

```json
{
  "logs": "2024-01-15 14:23:31 [ERROR] TimeoutError: upstream did not respond ...",
  "status_code": 504,
  "latency_ms": 30000,
  "error": "TimeoutError",
  "hints": []
}
```

| Field | Type | Description |
|---|---|---|
| `logs` | `str` | Timestamped application log output |
| `status_code` | `int` | HTTP status code of the failing request |
| `latency_ms` | `int` | Response latency in milliseconds |
| `error` | `str` | Error class name |
| `hints` | `list[str]` | Contextual hints revealed by investigation actions |

---

## Tasks

### Easy — `timeout_debug`

**Problem:** API returns `504 Gateway Timeout` after 30 000 ms.

| Action path | Reward |
|---|---|
| `increase_timeout` | **1.0** |
| `retry_request` | **0.5** (partial — surfaced but did not fix) |
| Any other action | **0.0** |

---

### Medium — `auth_failure`

**Problem:** API returns `401 Unauthorized` due to a stale token after secret rotation.

| Action path | Reward |
|---|---|
| `check_logs` → `fix_auth_token` | **1.0** (0.3 + 0.7) |
| `fix_auth_token` (without log check) | **0.6** |
| `check_logs` only | **0.3** |
| Wrong actions | **0.0** |

---

### Hard — `deployment_500`

**Problem:** Deployment `v2.4.2` introduced a breaking schema change causing 100 % `500` errors.

| Step | Action | Reward |
|---|---|---|
| 1 | `check_logs` | +0.3 |
| 2 | `analyze_deployment` | +0.3 |
| 3 | `rollback_version` | +0.4 |
| — | **Total (full path)** | **1.0** |

---

## Reward Logic

- All rewards are in `[0.0, 1.0]`.
- Rewards are stepwise and cumulative within an episode.
- Partial rewards are meaningful: they reflect real diagnostic value.
- Episodes terminate when a definitive fix (or irreversible partial fix) is applied, or when `max_steps` is reached.

---

## Setup

### Prerequisites

- Python 3.10+
- An OpenAI-compatible API key (or Hugging Face Inference Endpoint)

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run Locally

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-..."          # or OPENAI_API_KEY

python inference.py
```

### Expected output format

```
[START]
task: timeout_debug

[STEP]
action: increase_timeout
reward: 1.0

[END]
final_score: 1.0

[START]
task: auth_failure

[STEP]
action: check_logs
reward: 0.3

[STEP]
action: fix_auth_token
reward: 0.7

[END]
final_score: 1.0

...
```

### Use the environment directly in Python

```python
from env.environment import APIDebugEnv

env = APIDebugEnv(seed=42)

obs = env.reset("timeout_debug")
print(obs.model_dump())

result = env.step("increase_timeout")
print(result.reward, result.done)   # 1.0  True

score = env.final_score()
print(score)                        # 1.0
```

---

## Docker

### Build

```bash
docker build -t api-debug-env .
```

### Run

```bash
docker run --rm \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  -e HF_TOKEN="sk-..." \
  api-debug-env
```

---

## Deploy on Hugging Face Spaces

1. Create a new **Docker** Space on [huggingface.co/spaces](https://huggingface.co/spaces).
2. Push this repository to the Space's git remote.
3. Set the following **Space secrets** (Settings → Variables and Secrets):
   - `HF_TOKEN` — your Hugging Face API token or OpenAI key
   - `API_BASE_URL` — e.g. `https://api-inference.huggingface.co/v1`
   - `MODEL_NAME` — e.g. `meta-llama/Llama-3.1-8B-Instruct`
4. The Space will automatically build the Docker image and run `inference.py`.

---

## OpenEnv Compliance

| Requirement | Status |
|---|---|
| `reset()` | ✅ |
| `step(action)` | ✅ |
| `state()` | ✅ |
| Pydantic typed models | ✅ |
| `openenv.yaml` | ✅ |
| ≥ 3 tasks (easy / medium / hard) | ✅ |
| Graders with `[0.0, 1.0]` rewards | ✅ |
| Deterministic inference script | ✅ |
| ≤ 2 vCPU / 8 GB RAM / 20 min | ✅ |
