# Algorithms and failure behavior

This document describes the preserved conventional enforcement and AIMD baseline.
The additional trained PPO controllers, normalized observations, reward, and safety
layer are documented in [the ML implementation guide](rl.md).

## Token Bucket

The global refill rate is `R`; each of `N` registered clients gets `R/N`. Capacity is `max(1, rate × burst_seconds)`. At each operation, refill according to elapsed Redis server time, cap the balance, and require one token from both buckets. The atomic script computes both decisions before deducting either token. Denied operations may leave stored timestamps unchanged; the next check recomputes refill from the same original state.

The Redis policy key holds the current rate. Controller updates atomically settle every existing registered bucket at the previous rate before clamping it to its new capacity. A higher rate does not mint a fresh full bucket. The update is O(number of configured clients), suitable for this small fixed client registry. This is not a scalable dynamic-tenant design.

Redis `TIME` supplies timestamps. Tests can pass an explicit time to exercise deterministic boundaries. Tiny token deficits below `1e-9` are tolerated to avoid floating-point boundary rejection; balances remain nonnegative. Production Redis time has microsecond resolution and can follow wall-clock adjustments; negative deltas are clamped. A monotonic distributed clock is outside this experiment.

Idle bucket keys expire after at least twice their refill-to-capacity time. The policy key persists. Namespaces isolate trials; use a fresh namespace when changing algorithm or configuration. API keys never appear in Redis key names.

## Sliding Window Log

Each accepted request gets a unique member in global and client sorted sets. Remove entries at or before `now - window`, check both counts, then insert in both sets only if both pass. Limits are `max(1, floor(R × window))` globally and `max(1, floor(R × window / N))` per client. Fractional shares therefore have quantization loss. This is deliberate and reported, not a claim of identical burst semantics to Token Bucket.

State scales with accepted requests in the window. Admission checks are atomic; trimming expired entries on a denial does not consume a request budget. On a lowered limit, the retry estimate uses the expiry of enough entries to get below the new limit, rather than just the single oldest entry. Retry estimates are advisory because other traffic and future controller changes can alter admission.

## Controller

Once per interval, summarize a trailing window of **completed upstream attempts**. Compute nearest-rank p90 latency and the fraction with HTTP 5xx or transport failure, including upstream timeouts. EWMA is `new = alpha × observation + (1-alpha) × previous`. A decision requires fresh observations; old retained samples alone cannot keep changing the budget.

With enough latency samples, latency EWMA above the target triggers multiplicative decrease. A fresh unhealthy error EWMA can trigger decrease even at low volume; timeouts set a one-evaluation pending flag so a timeout is not repeatedly treated as a new timeout. Trailing-window errors may still justify subsequent backoff when new observations arrive. Configured minimum and maximum rates bound updates.

Healthy observations increase a counter only when admitted RPS reaches the configured fraction of the current budget. Three qualifying evaluations add a fixed step and clear the counter. Low demand, sparse observations, and unhealthy feedback clear the healthy streak. Sparse healthy traffic holds its budget; there is no probing without demand.

The default floor is 10 RPS. Configuration requires `min_rate × window >= 1.25 × min_samples`: a floor of 5 RPS with a five-second window could never collect 30 samples, leaving the controller unable to recover. Headroom accounts for partial client-bucket utilization; highly skewed traffic can still leave sample volume insufficient, in which case holding is intentional.

Observation and admission deques each hold at most 100,000 records. If an experiment exceeds that, decisions mark `truncated=true`; do not interpret that run as an exact five-second control window. The supported local workload suite is well below this ceiling.

The controller uses overlapping windows, so smoothing adds lag. It may back off repeatedly while a queue drains and recover slowly. This is an intentional baseline controller whose behavior should be measured. It is not Envoy's gradient controller and is not described as novel congestion control.

## HTTP and resource handling

- Configured identities are checked before reading the request body or consulting Redis.
- Global and client budgets use one atomic Redis Lua operation. In-flight capacity is reserved first and always released in `finally`.
- Forwarding is restricted to the configured upstream and public route prefixes. Backend controls and metrics cannot be reached through the proxy.
- Request and response hop-by-hop headers, including names nominated by `Connection`, are stripped. HTTPX decodes response content, so original encoding and length headers are not reused.
- Upstream attempts use a fixed timeout and bounded connection pool. Admission never queues or retries.
- Cancellation releases gateway in-flight capacity and backend queue/worker slots. Cancelling an HTTP client does not guarantee remote server work is immediately cancelled; backend work remains bounded.
- Failure to publish an adaptive update is logged with `applied=false`; the controller restores its last rate. Redis admission fails closed until connectivity returns. Disk/logging failure terminates the control task; readiness returns `503` and the last published Redis budget remains in effect.

Neither rate limiting nor an in-flight ceiling guarantees equal successful throughput: upstream scheduling and request timing still matter. Equal-demand fairness is an empirical result.
