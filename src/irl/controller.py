"""A bounded observation window and an independently testable feedback policy."""

import math
from collections import deque
from dataclasses import asdict, dataclass

from irl.config import ControllerConfig


@dataclass(frozen=True)
class Observation:
    at: float
    latency: float
    failed: bool = False
    timeout: bool = False


@dataclass
class Decision:
    at: float
    previous_rate: float
    rate: float
    reason: str
    samples: int
    p90: float | None
    latency_ewma: float | None
    error_ewma: float | None
    admitted_rps: float
    truncated: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class FeedbackController:
    def __init__(self, config: ControllerConfig, rate: float, started_at: float) -> None:
        self.config = config
        self.rate = rate
        self.started_at = started_at
        self.observations: deque[Observation] = deque(maxlen=100_000)
        self.admissions: deque[float] = deque(maxlen=100_000)
        self.latency_ewma: float | None = None
        self.error_ewma: float | None = None
        self.healthy = 0
        self.timeout_pending = False
        self.last_observed_at = -math.inf
        self.truncated = False

    def admit(self, now: float) -> None:
        self.truncated |= len(self.admissions) == self.admissions.maxlen
        self.admissions.append(now)

    def observe(self, observation: Observation) -> None:
        self.truncated |= len(self.observations) == self.observations.maxlen
        self.observations.append(observation)
        self.timeout_pending |= observation.timeout

    def evaluate(self, now: float) -> Decision:
        cfg = self.config
        cutoff = now - cfg.window
        while self.observations and self.observations[0].at < cutoff:
            self.observations.popleft()
        while self.admissions and self.admissions[0] < cutoff:
            self.admissions.popleft()
        samples = list(self.observations)
        duration = min(cfg.window, max(0.001, now - self.started_at))
        demand = len(self.admissions) / duration
        previous = self.rate
        p90: float | None = None
        reason = "insufficient_samples"
        fresh = bool(samples) and samples[-1].at > self.last_observed_at
        if samples and fresh:
            ordered = sorted(o.latency for o in samples)
            p90 = ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]
            errors = sum(o.failed for o in samples) / len(samples)
            a = cfg.alpha
            self.latency_ewma = (
                p90 if self.latency_ewma is None else (a * p90 + (1 - a) * self.latency_ewma)
            )
            self.error_ewma = (
                errors if self.error_ewma is None else (a * errors + (1 - a) * self.error_ewma)
            )
            self.last_observed_at = samples[-1].at
        enough = fresh and len(samples) >= cfg.min_samples
        overloaded = (
            enough and self.latency_ewma is not None and self.latency_ewma > cfg.latency_target
        ) or (fresh and self.error_ewma is not None and self.error_ewma > cfg.error_threshold)
        if self.timeout_pending or overloaded:
            self.rate = max(cfg.min_rate, self.rate * cfg.decrease_factor)
            reason = "upstream_timeout" if self.timeout_pending else "backend_unhealthy"
            self.healthy = 0
        elif enough:
            if demand >= cfg.demand_fraction * self.rate:
                self.healthy += 1
                reason = "healthy_wait"
                if self.healthy >= cfg.healthy_evaluations:
                    self.rate = min(cfg.max_rate, self.rate + cfg.increase_step)
                    self.healthy = 0
                    reason = "healthy_increase"
            else:
                self.healthy = 0
                reason = "low_demand"
        else:
            self.healthy = 0
        self.timeout_pending = False
        self.rate = min(cfg.max_rate, max(cfg.min_rate, self.rate))
        return Decision(
            now,
            previous,
            self.rate,
            reason,
            len(samples),
            p90,
            self.latency_ewma,
            self.error_ewma,
            demand,
            self.truncated,
        )
