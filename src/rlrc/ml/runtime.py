"""Small deterministic PPO actor deployment; no Torch imports on a gateway request."""

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rlrc.config import Settings
from rlrc.ml.features import FEATURE_VERSION, FeatureBuilder, Snapshot, apply_action
from rlrc.ml.safety import safety_filter


def load_predictor(path: Path, config: dict[str, Any], features: list[str]) -> Any:
    """Only load trusted locally trained sklearn files with matching feature semantics."""
    metadata = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
    if metadata["feature_version"] != FEATURE_VERSION or metadata["features"] != features:
        raise ValueError("Predictor feature schema mismatch")
    for name in ("max_rate", "min_rate", "target", "inflight_limit", "clients"):
        if metadata["simulator"][name] != config[name]:
            raise ValueError(f"Predictor normalization mismatch: {name}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Predictor checksum mismatch")
    import joblib
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=1)
    return joblib.load(path)


class NumpyActor:
    def __init__(self, directory: Path, cfg: Settings) -> None:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if not metadata.get("complete") or metadata["feature_version"] != FEATURE_VERSION:
            raise ValueError("Incomplete model or feature schema mismatch")
        sim = metadata["simulator"]
        for name, value in {
            "max_rate": cfg.controller.max_rate,
            "min_rate": cfg.controller.min_rate,
            "target": cfg.controller.latency_target,
            "inflight_limit": cfg.inflight_limit,
            "clients": len(cfg.clients),
        }.items():
            if sim[name] != value:
                raise ValueError(f"Model/runtime normalization mismatch: {name}")
        artifact = directory / "policy.npz"
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != metadata["export_sha256"]:
            raise ValueError("Model artifact checksum mismatch")
        self.features = metadata["features"]
        self.layers = []
        with np.load(artifact, allow_pickle=False) as arrays:
            for index in range(3):
                self.layers.append((arrays[f"w{index}"].copy(), arrays[f"b{index}"].copy()))
        expected = len(self.features)
        for w, b in self.layers:
            if w.shape[1] != expected or b.shape != (w.shape[0],):
                raise ValueError("Invalid actor matrix shapes")
            if not np.isfinite(w).all() or not np.isfinite(b).all():
                raise ValueError("Non-finite actor weights")
            expected = w.shape[0]
        if expected != 6:
            raise ValueError("Actor has the wrong action space")

    def action(self, observation: NDArray[np.float32]) -> int:
        if observation.shape != (len(self.features),) or not np.isfinite(observation).all():
            raise ValueError("Invalid normalized observation")
        x = observation
        for index, (weight, bias) in enumerate(self.layers):
            x = weight @ x + bias
            if index < len(self.layers) - 1:
                x = np.tanh(x)
        if not np.isfinite(x).all():
            raise ValueError("Non-finite action logits")
        return int(np.argmax(x))


class LearnedController:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.actor: NumpyActor | None = None
        self.predictor: Any = None
        self.load_error: str | None = None
        self.builder = FeatureBuilder(
            cfg.controller.max_rate,
            cfg.controller.latency_target,
            cfg.inflight_limit,
            cfg.ml.backend_telemetry,
            cfg.policy == "hybrid",
        )
        try:
            if cfg.controller.interval != 1:
                raise ValueError("Trained policies require a one-second control interval")
            if cfg.policy in {"rl", "rl_safe", "hybrid"}:
                self.actor = NumpyActor(Path(cfg.ml.model_path), cfg)
                if self.actor.features != self.builder.names:
                    raise ValueError("Actor observation variant does not match configuration")
            if cfg.policy in {"predictive", "hybrid"}:
                if cfg.ml.predictor_path is None:
                    raise FileNotFoundError("Predictor path not configured")
                self.predictor = load_predictor(
                    Path(cfg.ml.predictor_path),
                    {
                        "max_rate": cfg.controller.max_rate,
                        "min_rate": cfg.controller.min_rate,
                        "target": cfg.controller.latency_target,
                        "inflight_limit": cfg.inflight_limit,
                        "clients": len(cfg.clients),
                    },
                    self.builder.names[:-1] if cfg.policy == "hybrid" else self.builder.names,
                )
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"

    def decide(self, snapshot: Snapshot, fallback: float) -> dict[str, Any]:
        begin = time.perf_counter()
        fallback_invalid = not math.isfinite(fallback) or not (
            self.cfg.controller.min_rate <= fallback <= self.cfg.controller.max_rate
        )
        if fallback_invalid:
            fallback = self.cfg.rate
        record: dict[str, Any] = {
            "proposed_rate": fallback,
            "rate": fallback,
            "action": None,
            "safety": "fallback",
            "risk": None,
            "fallback_reason": self.load_error,
            "fallback_input_invalid": fallback_invalid,
        }
        try:
            if self.load_error:
                return record
            observation = self.builder.build(snapshot)
            if self.predictor is not None:
                features = observation[:-1] if self.cfg.policy == "hybrid" else observation
                risk = float(self.predictor.predict_proba(features.reshape(1, -1))[0, 1])
                if not math.isfinite(risk) or not 0 <= risk <= 1:
                    raise ValueError("Invalid predicted risk")
                record["risk"] = risk
                if self.cfg.policy == "hybrid":
                    observation[-1] = risk
            if self.cfg.policy == "predictive":
                proposal = min(fallback, snapshot.rate * 0.9) if record["risk"] > 0.7 else fallback
            else:
                if self.actor is None:
                    raise RuntimeError("Actor unavailable")
                action = self.actor.action(observation)
                record["action"] = action
                proposal = apply_action(
                    snapshot.rate,
                    action,
                    self.cfg.controller.min_rate,
                    self.cfg.controller.max_rate,
                )
            record["proposed_rate"] = proposal
            applied, intervention = safety_filter(
                snapshot, proposal, self.cfg.controller, self.cfg.policy in {"rl_safe", "hybrid"}
            )
            if (time.perf_counter() - begin) * 1000 > self.cfg.ml.max_inference_ms:
                raise TimeoutError("Controller inference exceeded configured time budget")
            record.update(rate=applied, safety=intervention, fallback_reason=None)
        except Exception as exc:
            record.update(
                rate=fallback, safety="fallback", fallback_reason=f"{type(exc).__name__}: {exc}"
            )
        finally:
            record["inference_ms"] = (time.perf_counter() - begin) * 1000
        return record
