"""Matched simulator evaluation with independent useful-work and SLO measurements."""

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from rlrc.config import ControllerConfig
from rlrc.controller import FeedbackController, Observation
from rlrc.ml.features import FeatureBuilder, Snapshot, apply_action
from rlrc.ml.simulator import SimConfig, Simulator


def evaluate_episode(
    config: SimConfig,
    seed: int,
    policy: str,
    model: Any = None,
    backend: bool = False,
    safe: bool = False,
    predictor: Any = None,
    warmup: int = 20,
    aimd_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from rlrc.ml.safety import safety_filter

    sim = Simulator(replace(config, record_responses=True), seed)
    builder = FeatureBuilder(
        config.max_rate,
        config.target,
        config.inflight_limit,
        backend,
        predictor is not None and policy == "rl_prediction",
    )
    cfg = ControllerConfig(
        latency_target=config.target, min_rate=config.min_rate, max_rate=config.max_rate
    )
    if aimd_overrides:
        cfg = ControllerConfig.model_validate(cfg.model_dump() | aimd_overrides)
    aimd = FeedbackController(cfg, config.initial_rate, 0)
    snapshot = Snapshot(rate=config.initial_rate)
    actions, inferences, interventions = [], [], []
    observation = builder.build(snapshot)
    for step in range(config.duration):
        rate = sim.rate
        reason = "none"
        if policy in {"rl", "rl_prediction"}:
            begin = time.perf_counter()
            action = int(model.predict(observation, deterministic=True)[0])
            inferences.append((time.perf_counter() - begin) * 1000)
            proposal = apply_action(rate, action, config.min_rate, config.max_rate)
            rate, reason = safety_filter(snapshot, proposal, cfg, safe)
        elif policy == "adaptive" or policy == "predictive":
            rate = aimd.evaluate(step + 0.000001).rate if step else rate
            if policy == "predictive" and predictor is not None:
                risk = float(predictor.predict_proba(observation.reshape(1, -1))[0, 1])
                if risk > 0.7:
                    rate = min(rate, sim.rate * 0.9)
                    reason = "predicted_overload"
                aimd.rate = rate
        elif policy in {"static_high", "sliding_high"}:
            rate = config.initial_rate * 1.5
        snapshot, result = sim.advance(
            rate,
            "none"
            if policy == "none"
            else "sliding_window"
            if policy.startswith("sliding")
            else "token_bucket",
        )
        for _ in range(result["admitted"]):
            aimd.admit(step + 0.5)
        # Preserve baseline five-second windows and nearest-rank latency semantics.
        for latency in result["latencies"]:
            aimd.observe(Observation(step + 0.9, latency))
        for _ in range(result["backend_errors"]):
            aimd.observe(Observation(step + 0.9, config.overhead, True))
        for _ in range(result["timeouts"]):
            aimd.observe(Observation(step + 0.9, config.timeout, True, True))
        observation = builder.build(snapshot)
        if builder.prediction:
            observation[-1] = float(predictor.predict_proba(observation[:-1].reshape(1, -1))[0, 1])
        actions.append(rate)
        interventions.append(reason)
    tail = sim.drain()
    measured = sim.history[warmup:]
    responses = [
        r
        for row in [*sim.history, tail]
        for r in row["responses"]
        if warmup <= r[0] < config.duration
    ]
    success = [r for r in responses if r[2]]
    good = [r for r in success if r[1] <= config.target]
    offered = sum(r["offered"] for r in measured)
    admitted = sum(r["admitted"] for r in measured)
    duration = config.duration - warmup
    per_client = [sum(r[4] == i for r in success) / duration for i in range(config.clients)]
    denom = sum(x * x for x in per_client)
    latency = [r[1] for r in success]
    return {
        "seed": seed,
        "family": sim.family,
        "policy": policy,
        "safe": safe,
        "backend_observations": backend,
        "duration": duration,
        "offered": offered,
        "admitted": admitted,
        "success": len(success),
        "goodput_rps": len(good) / duration,
        "success_rps": len(success) / duration,
        "latency_p50": float(np.quantile(latency, 0.5)) if latency else None,
        "latency_p95": float(np.quantile(latency, 0.95)) if latency else None,
        "latency_p99": float(np.quantile(latency, 0.99)) if latency else None,
        "backend_error_fraction": sum(r[3] == "overflow" for r in responses) / max(1, offered),
        "timeout_fraction": sum(r[3] == "timeout" for r in responses) / max(1, offered),
        "rejection_fraction": (offered - admitted) / max(1, offered),
        "jain_fairness": sum(per_client) ** 2 / (len(per_client) * denom)
        if denom and np.ptp(sim.weights) < 1e-8
        else None,
        "violation_seconds": sum(
            bool(r["latencies"]) and np.quantile(r["latencies"], 0.95) > config.target
            for r in measured
        ),
        "oscillation": float(np.mean(np.abs(np.diff(np.log(actions[warmup:]))))),
        "inference_p95_ms": float(np.quantile(inferences, 0.95)) if inferences else 0,
        "budgets": actions,
        "interventions": interventions,
        "series": [
            {
                k: row[k]
                for k in (
                    "at",
                    "rate",
                    "good",
                    "success",
                    "offered",
                    "queue",
                    "workers",
                    "backend_errors",
                    "rejected",
                )
            }
            for row in sim.history
        ],
    }


def validation(
    model: Any,
    config: SimConfig,
    backend: bool = False,
    predictor: Any = None,
    seeds: list[int] | None = None,
) -> dict[str, float]:
    rows = [
        evaluate_episode(
            config,
            seed,
            "rl_prediction" if predictor else "rl",
            model,
            backend=backend,
            predictor=predictor,
        )
        for seed in (seeds or [20001, 20002, 20003, 20004, 20005, 20006])
    ]
    # The same randomized validation scenarios are held fixed for checkpoint selection.
    return {
        "goodput": float(np.mean([r["goodput_rps"] for r in rows])),
        "failures": float(np.mean([r["backend_error_fraction"] for r in rows])),
        "score": float(
            np.mean(
                [
                    r["goodput_rps"]
                    - 10 * r["backend_error_fraction"]
                    - 5 * r["violation_seconds"] / r["duration"]
                    for r in rows
                ]
            )
        ),
    }


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def scalar(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Unsupported JSON value: {type(value).__name__}")

    path.write_text(
        json.dumps(data, indent=2, allow_nan=False, default=scalar) + "\n", encoding="utf-8"
    )
