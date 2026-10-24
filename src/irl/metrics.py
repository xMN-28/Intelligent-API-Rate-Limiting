"""Per-app registries keep tests and multiple service processes isolated."""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class GatewayMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "irl_gateway_requests",
            "Finished gateway requests",
            ["client", "outcome"],
            registry=self.registry,
        )
        self.admitted = Counter(
            "irl_gateway_admitted", "Upstream attempts", ["client"], registry=self.registry
        )
        self.latency = Histogram(
            "irl_upstream_duration_seconds",
            "Upstream attempt duration",
            buckets=(0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1, 2, 5),
            registry=self.registry,
        )
        self.overhead = Histogram(
            "irl_admission_duration_seconds",
            "Admission check duration",
            buckets=(0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1),
            registry=self.registry,
        )
        self.inflight = Gauge(
            "irl_gateway_inflight", "Current forwarded requests", registry=self.registry
        )
        self.budget = Gauge(
            "irl_rate_budget", "Current global request budget per second", registry=self.registry
        )
        self.decisions = Counter(
            "irl_controller_decisions", "Controller decisions", ["reason"], registry=self.registry
        )
