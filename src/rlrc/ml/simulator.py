"""Request-level discrete events: tokens, workers, queue, delayed replies, and timeouts.

Exogenous arrivals, identities and service variates are sampled BEFORE decisions,
so matched seeds give every controller the same workload, regardless of admission.
"""

import heapq
import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rlrc.ml.features import Snapshot, Telemetry


@dataclass
class SimConfig:
    duration: int = 160
    clients: int = 3
    max_rate: float = 120
    min_rate: float = 10
    initial_rate: float = 20
    target: float = 0.3
    timeout: float = 2
    inflight_limit: int = 64
    queue: int = 8
    burst: float = 0.25
    overhead: float = 0.005
    family: str = "random"
    arrival: str = "regular"
    workers: int = 4
    service_time: float = 0.1
    jitter: float = 0.1
    distribution: str = "uniform"
    record_responses: bool = False
    fixed_offered: float | None = None
    changes: tuple[tuple[int, int, float], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __post_init__(self) -> None:
        values = [
            self.max_rate,
            self.min_rate,
            self.initial_rate,
            self.target,
            self.timeout,
            self.service_time,
            self.overhead,
            self.jitter,
            self.burst,
        ]
        if not np.isfinite(values).all():
            raise ValueError("Simulator values must be finite")
        if not (
            self.duration >= 2
            and self.clients >= 1
            and self.workers >= 1
            and self.inflight_limit >= 1
            and self.queue >= 0
            and 0 < self.min_rate <= self.initial_rate <= self.max_rate
            and self.target > 0
            and self.timeout > 0
            and self.service_time > 0
            and 0 <= self.jitter <= 1
            and self.overhead >= 0
            and self.burst > 0
        ):
            raise ValueError("Invalid simulator configuration")
        if self.arrival not in {"regular", "poisson"} or self.distribution not in {
            "uniform",
            "lognormal",
        }:
            raise ValueError("Unknown arrival/service distribution")
        if self.fixed_offered is not None and (
            not math.isfinite(self.fixed_offered) or self.fixed_offered < 0
        ):
            raise ValueError("Offered rate must be finite and nonnegative")
        if any(
            not (0 <= at < self.duration and workers >= 1 and service > 0)
            for at, workers, service in self.changes
        ):
            raise ValueError("Invalid capacity change")


@dataclass
class Job:
    arrival: float
    service: float
    client: int
    responded: bool = False


class Simulator:
    def __init__(self, config: SimConfig, seed: int) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.rate = config.initial_rate
        self.tokens = np.maximum(1, self.rates * self.cfg.burst)
        self.token_at = 0.0
        self.logs: list[deque[float]] = [deque() for _ in range(config.clients + 1)]
        self.busy = 0
        self.inflight = 0
        self.queue: deque[Job] = deque()
        self.events: list[tuple[float, int, str, Job]] = []
        self.serial = 0
        self.history: list[dict[str, Any]] = []
        self._make_trace()

    @property
    def rates(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        return np.asarray([self.rate, *([self.rate / self.cfg.clients] * self.cfg.clients)])

    def _make_trace(self) -> None:
        c, rng = self.cfg, self.rng
        n = c.duration
        family = c.family
        if family == "random":
            family = str(
                rng.choice(
                    [
                        "steady",
                        "ramp",
                        "burst",
                        "sudden",
                        "gradual",
                        "overload",
                        "aggressive",
                        "recovery",
                    ]
                )
            )
        self.family = family
        base_workers = c.workers if c.family != "random" else int(rng.integers(2, 9))
        service = c.service_time if c.family != "random" else float(rng.uniform(0.04, 0.13))
        capacity = base_workers / service
        demand = capacity * float(rng.uniform(0.55, 1.9))
        self.demands: np.ndarray[Any, Any] = np.full(n, min(c.max_rate * 2, demand))
        self.workers = np.full(n, base_workers)
        self.services = np.full(n, service)
        cut = int(rng.uniform(0.22, 0.35) * n)
        restore = int(rng.uniform(0.58, 0.72) * n)
        lower = max(1, int(base_workers * float(rng.uniform(0.2, 0.55))))
        if family in {"sudden", "recovery", "aggressive", "unseen"}:
            self.workers[cut:restore] = lower
            self.demands[:] = min(c.max_rate * 2, capacity * 1.6)
        if family == "gradual":
            self.services[cut:restore] = np.linspace(service, service * 2.8, restore - cut)
        if family in {"ramp", "unseen"}:
            self.demands = np.linspace(capacity * 0.3, min(c.max_rate * 2, capacity * 2.5), n)
        if family in {"burst", "unseen"}:
            for start in (cut, restore):
                self.demands[start : min(n, start + max(3, n // 12))] *= 2
        if family == "overload":
            self.demands[:] = min(c.max_rate * 2.5, capacity * 2.8)
        if family == "steady":
            self.demands[:] = min(c.max_rate * 2, capacity * 0.8)
        if family == "fairness":
            self.demands[:] = min(c.max_rate * 2, capacity * 2)
        if family == "falling":
            self.demands = np.linspace(capacity * 2, capacity * 0.2, n)
        if c.fixed_offered is not None:
            self.demands[:] = c.fixed_offered
        for at, workers, service_time in c.changes:
            self.workers[at:] = workers
            self.services[at:] = service_time
        self.weights = np.ones(c.clients)
        if family == "aggressive":
            self.weights[0] = 10
        self.weights /= sum(self.weights)
        self.arrivals: list[
            tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]
        ] = []
        distribution = (
            str(rng.choice(["uniform", "lognormal"])) if c.family == "random" else c.distribution
        )
        for step in range(n):
            d = float(self.demands[step])
            if c.arrival == "poisson":
                count = int(rng.poisson(d))
                times = step + np.sort(rng.uniform(0, 1, count))
            else:
                count = max(0, int(round(d)))
                times = step + np.arange(count) / max(d, 1)
            clients = rng.choice(c.clients, size=count, p=self.weights)
            # Randomized uniform/lognormal service distributions, chosen per episode.
            variates = (
                rng.uniform(1 - c.jitter, 1 + c.jitter, count)
                if distribution == "uniform"
                else rng.lognormal(-(0.3**2) / 2, 0.3, count)
            )
            self.arrivals.append((times, clients, variates * self.services[step]))
        self.current_workers = int(self.workers[0])

    def _push(self, at: float, kind: str, job: Job) -> None:
        self.serial += 1
        heapq.heappush(self.events, (at, self.serial, kind, job))

    def _start_waiters(self, now: float) -> None:
        while self.queue and self.busy < self.current_workers:
            job = self.queue.popleft()
            self.busy += 1
            self._push(now + job.service, "finish", job)

    def _drain(self, until: float, telemetry: Telemetry, result: dict[str, Any]) -> None:
        while self.events and self.events[0][0] <= until:
            at, _, kind, job = heapq.heappop(self.events)
            if kind == "finish":
                self.busy -= 1
                self._start_waiters(at)
                self._push(at + self.cfg.overhead, "success", job)
            elif not job.responded:
                job.responded = True
                self.inflight -= 1
                success, timeout = kind == "success", kind == "timeout"
                latency = at - job.arrival
                telemetry.observe(
                    latency, success, not success, timeout, self.cfg.target, str(job.client)
                )
                result["latencies"].append(latency) if success else None
                result["rejection_latencies"].append(latency) if kind == "rejected" else None
                result["timeouts"] += int(timeout)
                result["backend_errors"] += int(kind == "overflow")
                result["success"] += int(success)
                result["good"] += int(success and latency <= self.cfg.target)
                result["late"] += int(success and latency > self.cfg.target)
                if success:
                    result["per_client"][job.client] += 1
                if self.cfg.record_responses:
                    result["responses"].append((job.arrival, latency, success, kind, job.client))

    def _refill(self, now: float) -> None:
        caps = np.maximum(1, self.rates * self.cfg.burst)
        self.tokens = np.minimum(caps, self.tokens + (now - self.token_at) * self.rates)
        self.token_at = now

    def advance(self, rate: float, policy: str = "token_bucket") -> tuple[Snapshot, dict[str, Any]]:
        step = int(self.t)
        if step >= self.cfg.duration:
            raise RuntimeError("Episode ended; reset before stepping again")
        self._refill(self.t)
        previous = self.rate
        self.rate = float(np.clip(rate, self.cfg.min_rate, self.cfg.max_rate))
        self.tokens = np.minimum(self.tokens, np.maximum(1, self.rates * self.cfg.burst))
        self.current_workers = int(self.workers[step])
        self._start_waiters(self.t)
        telemetry = Telemetry()
        result: dict[str, Any] = {
            "offered": 0,
            "admitted": 0,
            "success": 0,
            "good": 0,
            "late": 0,
            "timeouts": 0,
            "backend_errors": 0,
            "rejected": 0,
            "latencies": [],
            "rejection_latencies": [],
            "per_client": [0] * self.cfg.clients,
            "responses": [],
        }
        times, clients, services = self.arrivals[step]
        for at, client, service in zip(times, clients, services, strict=True):
            now, ci = float(at), int(client)
            self._drain(now, telemetry, result)
            telemetry.offered += 1
            result["offered"] += 1
            allowed = self.inflight < self.cfg.inflight_limit
            if policy == "sliding_window":
                for index in (0, ci + 1):
                    while self.logs[index] and self.logs[index][0] <= now - 1:
                        self.logs[index].popleft()
                    limit = max(1, math.floor(self.rate / (1 if index == 0 else self.cfg.clients)))
                    allowed &= len(self.logs[index]) < limit
                if allowed:
                    self.logs[0].append(now)
                    self.logs[ci + 1].append(now)
            elif policy != "none":
                self._refill(now)
                allowed &= bool(self.tokens[0] >= 1 - 1e-9 and self.tokens[ci + 1] >= 1 - 1e-9)
                if allowed:
                    self.tokens[0] = max(0, self.tokens[0] - 1)
                    self.tokens[ci + 1] = max(0, self.tokens[ci + 1] - 1)
            if not allowed:
                result["rejected"] += 1
                continue
            result["admitted"] += 1
            telemetry.admitted += 1
            self.inflight += 1
            job = Job(now, float(service), ci)
            self._push(now + self.cfg.timeout, "timeout", job)
            if self.busy < self.current_workers:
                self.busy += 1
                self._push(now + job.service, "finish", job)
            elif len(self.queue) < self.cfg.queue:
                self.queue.append(job)
            else:
                self._push(now + self.cfg.overhead, "overflow", job)
        self.t += 1
        self._drain(self.t, telemetry, result)
        s = telemetry.snapshot(self.rate, 1, self.inflight, self.rate / previous - 1)
        s.queue_util = len(self.queue) / max(1, self.cfg.queue)
        s.busy_util = self.busy / max(1, self.current_workers)
        s.backend_available = 1
        result.update(
            at=step,
            rate=self.rate,
            queue=len(self.queue),
            workers=self.current_workers,
            inflight=self.inflight,
            demand=float(self.demands[step]),
        )
        self.history.append(result)
        return s, result

    def drain(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "latencies": [],
            "rejection_latencies": [],
            "timeouts": 0,
            "backend_errors": 0,
            "success": 0,
            "good": 0,
            "late": 0,
            "per_client": [0] * self.cfg.clients,
            "responses": [],
        }
        self._drain(self.t + self.cfg.timeout + 0.1, Telemetry(), result)
        return result
