# Intelligent API Rate Limiting

A working HTTP gateway that adapts its request budget to upstream latency and errors. Includes atomic Redis limiters, a bounded backend, an open-arrival traffic generator, a live Grafana dashboard, and reproducible static-versus-adaptive experiments.

**Question:** can an explainable feedback controller improve successful throughput within a latency target when backend capacity changes?

This is a single-developer systems experiment, not a replacement for a production gateway. Adaptation is evaluated rather than presumed to win. No ML is needed for the initial controller.

![Architecture](docs/architecture.png)

## Quick start with Docker

Requires Docker Engine with Compose v2. Run from the repository root:

```bash
docker compose up --build -d
```

Open [the live Grafana dashboard](http://localhost:3000/d/intelligent-rate-limiting), [gateway OpenAPI docs](http://localhost:8000/docs), or [Prometheus](http://localhost:9090). Grafana is provisioned automatically; no login is needed for this local demo.

Send a request (on Windows use `curl.exe`):

```bash
curl -i -H "X-API-Key: demo-alice" http://localhost:8000/proxy/work
```

Configured clients are `demo-alice`, `demo-bob`, and `demo-carol`. The proxy exposes `/work` and `/echo/*` at the configured backend. It preserves ordinary HTTP methods, repeated query parameters, request bodies, and relevant response headers. Body size is limited to 1 MiB. Credentials and hop-by-hop headers are stripped before forwarding. There are no automatic retries.

### Live demo

Install the Python tools:

```bash
uv sync --frozen --extra dev
uv run irl run --scenario demo --output results/runs/live-demo
```

Watch Grafana while the demo moves through normal traffic, a burst, degraded backend capacity, and recovery. The measured demo lasts 100 seconds plus a 10-second warm-up. `run` targets the existing services on ports 8000/8001 and changes backend capacity; do not run it alongside another experiment. Match local `IRL_POLICY`/`IRL_RATE` to the target gateway when recording its configuration.

The saved report is `results/runs/live-demo/REPORT.md`. Live controller decisions are stored separately in `results/runs/live/decisions.jsonl` by the gateway. Isolated `suite` trials capture their own decisions alongside raw requests.

Stop the stack with `docker compose down`. Ports are published only on localhost; backend admin controls are not exposed through the gateway.

## Reproducible comparisons

The suite launches its **own** gateway and backend on ports 18000/18001, using Redis from the running stack. Every trial gets a unique Redis namespace and fresh backend state. It does not use the running demo gateway and does not feed the live demo dashboard.

First calibrate this machine:

```bash
uv run irl calibrate --output results/runs/calibration
```

Calibration sweeps offered rates without rate limiting, retaining the highest observed rate with at least 99% successful responses, successful-request p95 within the configured target, and at most 1% missed launches. It suggests conservative and higher static budgets at 60% and 90% of that rate. If every tested rate passes, expand `--rates` to locate saturation; calibration reports whether a failing upper bound was found.

Compare capacity changes across all six configurations:

```bash
uv run irl suite --calibration results/runs/calibration/calibration.json --scenarios capacity --output results/runs/capacity
```

The configurations are no limiter, two static Token Bucket rates, two static Sliding Window rates, and adaptive Token Bucket starting at the conservative rate. Each comparison uses five repetitions, identical workload seeds across policies, randomized policy order, 30 seconds of warm-up, and 300 seconds of measurement. The capacity comparison takes approximately 2 hours 45 minutes plus process startup and analysis. The full seven-scenario suite takes about 19 hours; run selected scenarios during development.

For a shorter **uncalibrated smoke check**:

```bash
uv run irl suite --scenarios demo --repetitions 1 --rates 90 150 --scale 0.15 --output results/runs/smoke
```

Scaling shortens time, not request rates. Very short runs cannot establish controller quality or meaningful tail behavior. The report is under the suite's `evaluation/` directory. Output directories must be new: the CLI refuses to overwrite existing runs.

If this host cannot sustain the default offered rates, use `configs/portable.yaml` with the `portable-demo` scenario. It slows the backend so 30–100 RPS can still cause actual overload. In PowerShell:

```powershell
$env:IRL_CONFIG = 'configs/portable.yaml'
uv run irl calibrate --rates 10 20 30 40 60 --output results/runs/portable-calibration
uv run irl suite --calibration results/runs/portable-calibration/calibration.json --scenarios portable-demo --repetitions 1 --output results/runs/portable-demo
Remove-Item Env:IRL_CONFIG
```

Always use the same profile for calibration and evaluation. A different profile is a different experiment, not a way to rescue invalid results from the original workload. Reports exclude runs with excessive missed launches.

For one isolated run of the configured policy, `uv run irl demo --scenario demo` owns and stops its service processes automatically. With `IRL_CONFIG=configs/portable.yaml`, use `uv run irl demo --scenario portable-demo` for the lower-rate version. `IRL_POLICY` and `IRL_RATE` select its policy and initial budget; `suite` remains the command for matched comparisons.

Without installing Python locally:

```bash
docker compose --profile tools run --rm runner suite --scenarios demo --repetitions 1 --rates 90 150 --output results/runs/container-demo
```

Rebuild reports from saved runs:

```bash
uv run irl report results/runs/capacity/evaluation
```

See [benchmark methodology](docs/benchmarks.md) and [verified implementation results](results/README.md). Full benchmark results are generated, not fabricated or checked in as claimed improvements.

## How control works

**Enforcement:** a Redis Lua script checks global and client budgets in one operation. A rejection consumes neither budget. Token Bucket permits bounded bursts; Sliding Window Log enforces an exact rolling count. Redis server time supplies a common clock. Keys expire after inactivity. All configured clients receive equal shares; idle clients' unused shares are not borrowed.

**Feedback:** once per second, the controller evaluates a trailing five-second observation window. It applies EWMA (`alpha=0.2`) to p90 upstream latency and upstream failure fraction. It decreases the total rate by 20% under overload, or adds 5 RPS after three healthy, demand-saturated evaluations. Min/max rates bound the result. Latency-based changes require 30 samples; a fresh timeout triggers immediate backoff at the next evaluation. Gateway rejections never enter backend-health statistics.

Rate changes settle existing buckets at the old rate and clamp balances to the new capacity. They do not reset buckets or grant a new burst. Every adaptive decision records its inputs, reason, proposed rate, and whether publication to Redis succeeded.

**Safety:** the gateway has a fixed in-flight ceiling, bounded upstream connections, and a timeout. No requests wait for admission. The backend has a bounded worker pool and bounded queue. Its configurable sleep represents a simulated I/O dependency; the HTTP traffic and resource contention are real. CPU and worker metrics are observable, but CPU is not used as a proxy for all backend capacity.

See [algorithm details](docs/algorithms.md) for edge cases and tradeoffs.

## Local development

Python 3.12 and a reachable Redis server are required. `uv.lock` fixes all direct and transitive dependencies.

```bash
uv sync --frozen --extra dev
docker compose up -d redis
uv run irl serve backend
# In a second terminal:
uv run irl serve gateway
```

Edit `configs/default.yaml`; workload definitions live in `configs/scenarios.yaml`. Environment overrides: `IRL_CONFIG`, `IRL_REDIS_URL`, `IRL_UPSTREAM`, `IRL_POLICY`, `IRL_RATE`, `IRL_NAMESPACE`, `IRL_DECISION_LOG`, and `IRL_ADMIN_TOKEN`. Both services must use the same admin token. CLI-owned isolated trials inherit the selected configuration and assign their own upstream and namespace.

Choose a fresh namespace when restarting a policy or changing static budgets. Persistent Redis state intentionally survives ordinary gateway restarts. Never run two gateway/controller processes in the same namespace; that deployment is outside v1.

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
docker compose config --quiet
```

Redis tests skip with an explicit reason when Redis is unavailable. CI sets `REQUIRE_REDIS=1` so missing Redis fails the build. To use another Redis instance, set `TEST_REDIS_URL` (default `redis://localhost:6379/15`). Tests clean only their own unique keys, never flush a database.

Regenerate committed visual assets:

```bash
uv run python scripts/provision_dashboard.py
uv run python scripts/render_architecture.py
```

## Repository

| Directory | Contents |
|---|---|
| `src/irl/` | Gateway, Redis scripts, controller, backend, generator, experiment CLI, reporting |
| `tests/` | Unit, real-Redis integration, HTTP behavior, cancellation and accounting tests |
| `configs/` | Validated settings and seven benchmark scenarios plus the demo |
| `deploy/` | Docker image, Prometheus configuration, provisioned Grafana dashboard |
| `scripts/` | Rebuildable architecture diagram and dashboard definitions |
| `docs/` | Architecture, algorithms, methodology, and development milestones |
| `results/` | Verification record and generated-run instructions |

## Boundaries and next steps

The project covers one gateway process, one upstream, registered clients, equal-cost requests, and local Docker deployment. It does not classify clients as malicious. Fixed client shares sacrifice utilization when clients are idle. A fixed request rate cannot fully model mixed request costs; adaptive concurrency is an important follow-up comparison. Feedback lag, thresholds, and EWMA can cause underutilization or oscillation, especially after abrupt changes.

Redis failure makes configured rate-limited policies reject with `503`; the `none` baseline intentionally bypasses Redis admission. Readiness still checks Redis for a common experiment setup. Quota/capacity rejections return `429`, upstream timeouts `504`, and connection errors `502`. These outcomes are measured separately.

The local demo uses known credentials, anonymous Grafana viewing, and no TLS. Production authentication, external deployment, distributed controllers, Redis Cluster, streaming, work-conserving client allocation, and ML are separate extensions.

Related established approaches: [Envoy adaptive concurrency](https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/adaptive_concurrency_filter.html), [Netflix concurrency-limits](https://github.com/Netflix/concurrency-limits), and [Redis limiter algorithms](https://redis.io/tutorials/howtos/ratelimiting/).
