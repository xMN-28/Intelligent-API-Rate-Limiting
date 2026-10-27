"""Validated, reproducible configuration; environment overrides support isolated runs."""

import os
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ControllerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    interval: float = Field(1.0, gt=0)
    window: float = Field(5.0, gt=0)
    alpha: float = Field(0.2, gt=0, le=1)
    latency_target: float = Field(0.2, gt=0)
    error_threshold: float = Field(0.02, ge=0, le=1)
    decrease_factor: float = Field(0.8, gt=0, lt=1)
    increase_step: float = Field(5.0, gt=0)
    healthy_evaluations: int = Field(3, ge=1)
    demand_fraction: float = Field(0.8, gt=0, le=1)
    min_samples: int = Field(30, ge=1)
    min_rate: float = Field(10.0, ge=1)
    max_rate: float = Field(500.0, ge=1)

    @model_validator(mode="after")
    def valid_bounds(self) -> Self:
        if self.min_rate > self.max_rate:
            raise ValueError("min_rate must not exceed max_rate")
        if self.min_rate * self.window < self.min_samples * 1.25:
            raise ValueError("min_rate × window must cover min_samples with 25% headroom")
        return self


class MLConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_path: str = "models/ppo-gateway"
    predictor_path: str | None = None
    backend_telemetry: bool = False
    max_inference_ms: float = Field(50, gt=0)


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    upstream: str = "http://localhost:8001"
    redis_url: str = "redis://localhost:6379/0"
    policy: Literal[
        "none",
        "token_bucket",
        "sliding_window",
        "adaptive",
        "rl",
        "rl_safe",
        "predictive",
        "hybrid",
    ] = "adaptive"
    rate: float = Field(100.0, ge=1)
    burst_seconds: float = Field(0.25, gt=0)
    sliding_window: float = Field(1.0, gt=0)
    clients: dict[str, str] = Field(
        default_factory=lambda: {"alice": "demo-alice", "bob": "demo-bob", "carol": "demo-carol"}
    )
    namespace: str = "demo"
    inflight_limit: int = Field(128, ge=1)
    upstream_timeout: float = Field(2.0, gt=0)
    max_body_bytes: int = Field(1_048_576, ge=1)
    decision_log: str = "results/runs/live/decisions.jsonl"
    admin_token: str = Field("local-demo-admin", min_length=1)
    backend_workers: int = Field(8, ge=1)
    backend_queue: int = Field(32, ge=0)
    service_time: float = Field(0.04, ge=0)
    service_jitter: float = Field(0.1, ge=0, le=1)
    controller: ControllerConfig = Field(default_factory=lambda: ControllerConfig())
    ml: MLConfig = Field(default_factory=lambda: MLConfig())

    @model_validator(mode="after")
    def validate_clients(self) -> Self:
        if not self.clients or len(set(self.clients.values())) != len(self.clients):
            raise ValueError("clients must be nonempty and API keys unique")
        if any(not name or not key for name, key in self.clients.items()):
            raise ValueError("client names and API keys must not be empty")
        if self.policy in {"adaptive", "rl", "rl_safe", "predictive", "hybrid"} and not (
            self.controller.min_rate <= self.rate <= self.controller.max_rate
        ):
            raise ValueError("initial adaptive rate must lie within controller bounds")
        return self


def load_settings() -> Settings:
    path = Path(os.getenv("RLRC_CONFIG", "configs/default.yaml"))
    if "RLRC_CONFIG" in os.environ and not path.is_file():
        raise FileNotFoundError(f"RLRC_CONFIG does not exist: {path}")
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    data = data or {}
    for name in (
        "upstream",
        "redis_url",
        "policy",
        "rate",
        "namespace",
        "decision_log",
        "admin_token",
    ):
        value = os.getenv(f"RLRC_{name.upper()}")
        if value is not None:
            data[name] = value
    return Settings.model_validate(data)
