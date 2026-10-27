"""Shared physical telemetry, feature definitions and fixed inference normalization."""

from collections import Counter
from dataclasses import asdict, dataclass, field

import numpy as np
from numpy.typing import NDArray

FEATURE_VERSION = 1
ACTIONS = (0.8, 0.9, 0.95, 1.0, 1.05, 1.1)
FEATURES = (
    "budget",
    "offered",
    "admitted",
    "completed",
    "goodput",
    "latency_ewma",
    "p90",
    "errors",
    "timeouts",
    "rejected",
    "inflight",
    "previous_change",
    "demand_trend",
    "latency_trend",
    "sample_volume",
    "has_samples",
)


@dataclass
class Snapshot:
    rate: float = 20
    offered: float = 0
    admitted: float = 0
    completed: float = 0
    goodput: float = 0
    p90: float = 0
    errors: float = 0
    timeouts: float = 0
    rejected: float = 0
    inflight: float = 0
    samples: int = 0
    queue_util: float = 0
    busy_util: float = 0
    backend_available: float = 0
    previous_change: float = 0

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


class FeatureBuilder:
    def __init__(
        self,
        max_rate: float,
        target: float,
        inflight_limit: int,
        backend: bool = False,
        prediction: bool = False,
    ) -> None:
        self.max_rate, self.target, self.inflight_limit = max_rate, target, inflight_limit
        self.backend, self.prediction = backend, prediction
        self.previous = Snapshot()
        self.ewma: float | None = None

    @property
    def names(self) -> list[str]:
        return [
            *FEATURES,
            *(["queue_util", "busy_util", "backend_available"] if self.backend else []),
            *(["overload_risk"] if self.prediction else []),
        ]

    def build(self, s: Snapshot, risk: float = 0) -> NDArray[np.float32]:
        if not np.isfinite(list(s.as_dict().values())).all() or not np.isfinite(risk):
            raise ValueError("Non-finite telemetry")
        if s.samples:
            self.ewma = s.p90 if self.ewma is None else 0.2 * s.p90 + 0.8 * self.ewma
        m, t = self.max_rate, self.target * 4
        values = [
            s.rate / m,
            s.offered / m,
            s.admitted / m,
            s.completed / m,
            s.goodput / m,
            (self.ewma or 0) / t,
            s.p90 / t,
            s.errors,
            s.timeouts,
            s.rejected / max(1, s.offered),
            s.inflight / self.inflight_limit,
            s.previous_change / 0.2,
            (s.offered - self.previous.offered) / m,
            (s.p90 - self.previous.p90) / t,
            min(1, s.samples / 100),
            float(s.samples > 0),
        ]
        if self.backend:
            values += [s.queue_util, s.busy_util, s.backend_available]
        if self.prediction:
            values.append(risk)
        self.previous = s
        return np.clip(np.asarray(values, dtype=np.float32), -1, 4)


@dataclass
class Telemetry:
    """O(1) request counters; no network or model calls on the request hot path."""

    offered: int = 0
    admitted: int = 0
    successes: int = 0
    good: int = 0
    failed: int = 0
    timed_out: int = 0
    latencies: list[float] = field(default_factory=list)
    per_client: Counter[str] = field(default_factory=Counter)

    def observe(
        self, latency: float, success: bool, failed: bool, timeout: bool, target: float, client: str
    ) -> None:
        if len(self.latencies) < 100_000:
            self.latencies.append(latency)
        self.successes += int(success)
        self.good += int(success and latency <= target)
        self.failed += int(failed)
        self.timed_out += int(timeout)
        if success:
            self.per_client[client] += 1

    def snapshot(self, rate: float, duration: float, inflight: int, change: float = 0) -> Snapshot:
        n = len(self.latencies)
        return Snapshot(
            rate=rate,
            offered=self.offered / duration,
            admitted=self.admitted / duration,
            completed=n / duration,
            goodput=self.good / duration,
            p90=float(np.quantile(self.latencies, 0.9, method="inverted_cdf")) if n else 0,
            errors=self.failed / max(1, n),
            timeouts=self.timed_out / max(1, n),
            rejected=max(0, self.offered - self.admitted) / duration,
            inflight=inflight,
            samples=n,
            previous_change=change,
        )


def apply_action(rate: float, action: int, low: float, high: float) -> float:
    if not isinstance(action, int | np.integer) or not 0 <= action < len(ACTIONS):
        raise ValueError(f"Invalid discrete action: {action}")
    return float(np.clip(rate * ACTIONS[action], low, high))
