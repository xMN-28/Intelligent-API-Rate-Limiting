"""Per-app registries keep tests and multiple service processes isolated."""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class GatewayMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.offered = Counter(
            "rlrc_gateway_offered",
            "Eligible offered HTTP traffic",
            ["client"],
            registry=self.registry,
        )
        self.inference = Histogram(
            "rlrc_controller_inference_seconds",
            "Control-interval inference",
            buckets=(0.00001, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1),
            registry=self.registry,
        )
        self.interventions = Counter(
            "rlrc_safety_interventions",
            "Proposed/applied action guards",
            ["reason"],
            registry=self.registry,
        )
        self.fallbacks = Counter(
            "rlrc_model_fallbacks", "Explicit learned-policy fallbacks", registry=self.registry
        )
        self.model_ready = Gauge(
            "rlrc_model_ready", "Learned policy loaded and validated", registry=self.registry
        )
        self.proposed_budget = Gauge(
            "rlrc_proposed_budget", "Learned controller proposed budget", registry=self.registry
        )
        self.requests = Counter(
            "rlrc_gateway_requests",
            "Finished gateway requests",
            ["client", "outcome"],
            registry=self.registry,
        )
        self.admitted = Counter(
            "rlrc_gateway_admitted", "Upstream attempts", ["client"], registry=self.registry
        )
        self.latency = Histogram(
            "rlrc_upstream_duration_seconds",
            "Upstream attempt duration",
            buckets=(0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1, 2, 5),
            registry=self.registry,
        )
        self.overhead = Histogram(
            "rlrc_admission_duration_seconds",
            "Admission check duration",
            buckets=(0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1),
            registry=self.registry,
        )
        self.inflight = Gauge(
            "rlrc_gateway_inflight", "Current forwarded requests", registry=self.registry
        )
        self.budget = Gauge(
            "rlrc_rate_budget", "Current global request budget per second", registry=self.registry
        )
        self.decisions = Counter(
            "rlrc_controller_decisions", "Controller decisions", ["reason"], registry=self.registry
        )
