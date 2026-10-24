# Benchmark protocol

## Calibration and held-out evaluation

Calibrate a healthy backend with no rate limiter, using the same gateway, connection pool, timeout, and fixed in-flight ceiling as evaluation. Sweep offered RPS; each point defaults to 5 seconds warm-up and 15 seconds measurement. Retain the highest tested rate with at least 99% successful launched requests, successful-request p95 at or below the target, and at most 1% missed launches. A sweep with no passing point fails. A sweep with no failing point above the chosen capacity has not located saturation; expand it before claiming a tuned baseline.

Conservative and higher static rates are 60% and 90% of the observed healthy capacity. Both Token Bucket and Sliding Window use both budgets. Adaptive Token Bucket starts at the conservative rate. Controller parameters remain fixed across evaluation scenarios; the suite records the exact settings. Calibration results are rejected if backend capacity/queue/service-time, gateway timeout/in-flight ceiling, or latency target differs from the evaluation configuration.

This calibration estimates capacity only for this workload and machine. The backend sleeps to represent I/O service times; it does not prove CPU-bound production capacity. Use a dedicated machine or record competing workloads before stronger performance claims.

## Arrival model and accounting

The Python scheduler advances an intended launch timestamp independently of completed requests. Constant phases use equally spaced arrivals; ramp phases update the spacing with the phase's linear rate trajectory. Client selection is seeded and weighted; request-keyed service jitter is deterministic. Equal demand means equal expected offered traffic, not exact round-robin scheduling.

Each offered arrival produces one CSV row, even when not launched. If generator concurrency is exhausted or scheduling lag exceeds the configured threshold, record a missed launch and its reason. Bound network concurrency, the writer queue, the backend queue, gateway in-flight work, and Redis limiter windows. CSV disk writes run outside the event loop. Slow disk can increase launch lag and invalidate the run rather than silently changing the offered workload.

The manifest captures workload, settings, seed, client names, timestamps, Python/OS/hardware information, and the current Git commit/dirty state when available. Run manifests exclude API keys and the admin token. The CLI removes transient service configurations containing credentials after stopping its child processes.

Do not run two suites on the same ports. The runner refuses to bind occupied ports. Workloads assign results by scheduled arrival: warm-up requests are excluded even if they finish during measurement, and measurement requests are included even if they finish during drain. All launched requests drain before the run is marked complete.

Accounting identities:

```text
offered = missed launches + launched
launched = success + rejected + gateway error + backend error
         + upstream timeout + upstream transport error
         + client timeout + client transport error
completed = launched requests that received an HTTP response
```

The `X-Admission` response header identifies observed admissions. Admission is **unknown** if the client receives no HTTP response; it is never inferred as a backend rejection. A `429` rejection is distinct from infrastructure failure (`503` at the gateway) or a backend's own `503`. An unknown client or route is a gateway error rather than quota rejection.

## Measurements

Goodput counts successful responses whose client-observed duration is within the target, divided by scheduled measurement duration. Also report successful RPS, successful-only p50/p95/p99, rejection-only p95, gateway admission-check p95 overhead, offered/launched/admitted/completed counts, outcomes, and per-client success RPS. Fast rejections cannot improve the successful-request latency percentile.

Jain's index is `(sum(x))² / (N × sum(x²))` for per-client successful throughput. Report it only for scenarios explicitly marked comparable equal demand. The aggressive-client scenario reports per-client throughput without claiming raw-throughput equality is the appropriate fairness objective.

Recovery after a recorded worker-capacity restoration means three consecutive five-second windows with successful p95 within target and goodput at least 80% of pre-degradation goodput. The reference is the 30 seconds before degradation (or the available shorter interval). A missing recovery is `null`, not zero. Recovery is relative to observed pre-degradation service, not proof of reaching theoretical maximum capacity.

Prometheus provides live upstream latency histograms, request outcomes, admission overhead, current budget, in-flight work, backend workers/queue, and backend process CPU. Saved backend samples record worker capacity, active work, waiting work, and configured service time once per second. The generator's raw request records remain the source for end-user latency percentiles.

## Repetitions and interpretation

Use 30 seconds warm-up and 300 seconds measurement, five repetitions, matched seeds, and randomized policy order. The seven evaluation workloads are steady, ramp, burst, sustained overload, capacity drop/recovery, equal-demand fairness, and an aggressive client. The full matrix has 210 runs; start with one scenario while developing.

Charts aggregate only complete runs with at most 1% missed launches. Error bars show sample standard deviation, not confidence intervals. Each run's percentile is calculated from its raw observations; percentiles are not pooled by averaging their inputs or presented as an aggregate fleet percentile. Every run remains listed, including invalid runs. Time-series plots include offered traffic, goodput, rejections, errors, successful p95, and applied budgets.

Useful questions include whether adaptation reduces time above the latency target, whether it protects ordinary clients during an aggressive client's traffic, how much healthy throughput it sacrifices, and how long it takes to recover. A tuned static policy may win under stable capacity. Present that result if observed.

## Reproduction artifacts

Each isolated run contains `manifest.json`, `requests.csv`, `backend.json`, `events.json`, service logs, and adaptive `decisions.jsonl` when applicable. Reporting adds `summary.json` and `timeseries.png`. Evaluation roots contain `summaries.json`, `comparison.png`, and `REPORT.md`. Suite roots also hold `suite.json` with the frozen execution schedule and calibration status.

Large raw runs are ignored under `results/runs/`. Preserve their manifests and commands when publishing selected results. Do not compare runs across machines as though they had the same backend capacity.

## Portable verification profile

`configs/portable.yaml` and `portable-demo` use a smaller, slower backend and lower offered rates to support hosts with limited generator throughput. Calibrate that profile independently. Preserve the high-rate run's invalid status rather than relabeling it as a valid lower-rate experiment. A shortened portable smoke run verifies the pipeline and control responses, but does not replace five full-length repetitions of the evaluation matrix.
