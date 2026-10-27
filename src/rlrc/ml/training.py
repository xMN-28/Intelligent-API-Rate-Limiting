"""Reproducible PPO training, hardware probes, checkpoints and fixed validation selection."""

import argparse
import hashlib
import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from rlrc.loadgen import machine_manifest
from rlrc.ml.environment import RateControlEnv
from rlrc.ml.evaluation import save_json, validation
from rlrc.ml.features import FEATURE_VERSION, FeatureBuilder
from rlrc.ml.simulator import SimConfig


def export_policy(model: PPO, output: Path) -> str:
    """Export the trained deterministic categorical actor, without pickle at deployment."""
    layers = [
        layer
        for layer in model.policy.mlp_extractor.policy_net
        if isinstance(layer, torch.nn.Linear)
    ]
    assert isinstance(model.policy.action_net, torch.nn.Linear)
    layers.append(model.policy.action_net)
    arrays = {}
    for index, layer in enumerate(layers):
        arrays[f"w{index}"] = layer.weight.detach().cpu().numpy()
        arrays[f"b{index}"] = layer.bias.detach().cpu().numpy()
    np.savez_compressed(output / "policy.npz", allow_pickle=False, **arrays)
    return hashlib.sha256((output / "policy.npz").read_bytes()).hexdigest()


def make_env(config: SimConfig, backend: bool, predictor: Any = None) -> Any:
    def create() -> Any:
        torch.set_num_threads(1)
        return Monitor(RateControlEnv(config, backend, predictor))

    return create


def vector_env(
    config: SimConfig, count: int, mode: str, backend: bool = False, predictor: Any = None
) -> VecEnv:
    factories = [make_env(config, backend, predictor) for _ in range(count)]
    return (
        SubprocVecEnv(factories, start_method="spawn")
        if mode == "subproc"
        else DummyVecEnv(factories)
    )


def source_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path("src/rlrc").rglob("*.py")):
        digest.update(path.as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def hardware_probe(output: Path, steps: int = 2048) -> dict[str, Any]:
    torch.set_num_threads(1)
    results: list[dict[str, Any]] = []
    config = SimConfig(duration=80)
    for device in ["cpu", *(["cuda"] if torch.cuda.is_available() else [])]:
        for mode, count in [("dummy", 1), ("dummy", 4), ("subproc", 4)]:
            begin = time.perf_counter()
            env = vector_env(config, count, mode)
            startup = time.perf_counter() - begin
            model = PPO(
                "MlpPolicy",
                env,
                n_steps=128,
                batch_size=256 if count > 1 else 128,
                n_epochs=3,
                policy_kwargs={"net_arch": dict(pi=[64, 64], vf=[64, 64])},
                seed=11,
                device=device,
                verbose=0,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            begin = time.perf_counter()
            model.learn(steps)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            results.append(
                {
                    "device": device,
                    "mode": mode,
                    "n_envs": count,
                    "steps": model.num_timesteps,
                    "seconds": elapsed,
                    "steps_s": model.num_timesteps / elapsed,
                    "startup_seconds": startup,
                    "rss_mb": psutil.Process().memory_info().rss / 2**20,
                    "cuda_allocated_mb": torch.cuda.max_memory_allocated() / 2**20
                    if device == "cuda"
                    else 0,
                }
            )
            print(results[-1], flush=True)
            env.close()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    fastest = max(results, key=lambda row: row["steps_s"])
    data = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "ram_gb": psutil.virtual_memory().total / 2**30,
        "results": results,
        "recommended": fastest,
    }
    save_json(output, data)
    return data


class ProgressCallback(BaseCallback):
    def __init__(
        self,
        output: Path,
        config: SimConfig,
        backend: bool,
        predictor: Any,
        every: int,
        started: float,
    ) -> None:
        super().__init__()
        self.output, self.config, self.backend = output, config, backend
        self.predictor, self.every, self.started = predictor, every, started
        self.next_check = every
        self.best = -float("inf")
        self.records: list[dict[str, Any]] = []
        self.accumulator: dict[str, list[float]] = {}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for name, value in info.get("components", {}).items():
                self.accumulator.setdefault(name, []).append(value)
        if self.num_timesteps >= self.next_check:
            result = validation(self.model, self.config, self.backend, self.predictor)
            components = {k: float(np.mean(v)) for k, v in self.accumulator.items()}
            self.accumulator.clear()
            record = {
                "steps": self.num_timesteps,
                "seconds": time.perf_counter() - self.started,
                **result,
                "reward_components": components,
            }
            self.records.append(record)
            save_json(self.output / "validation.json", self.records)
            for key, value in result.items():
                self.logger.record(f"validation/{key}", value)
            for key, value in components.items():
                self.logger.record(f"reward/{key}", value)
            self.model.save(self.output / f"checkpoint-{self.num_timesteps}.zip")
            if result["score"] > self.best:
                self.best = result["score"]
                self.model.save(self.output / "best.zip")
            print(json.dumps(record), flush=True)
            self.next_check = self.num_timesteps + self.every
        return True


def train(
    output: Path,
    steps: int = 131072,
    seed: int = 42,
    device: str = "cpu",
    count: int = 4,
    mode: str = "dummy",
    backend: bool = False,
    resume: Path | None = None,
    predictor_path: Path | None = None,
    validation_interval: int = 32768,
    config: SimConfig | None = None,
) -> Path:
    if steps < 1 or not 1 <= count <= 8 or validation_interval < 1:
        raise ValueError(
            "Positive training steps/validation interval and 1–8 environments required"
        )
    if output.exists():
        raise FileExistsError(
            "Choose a new output directory; use --resume to load an older checkpoint"
        )
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    requested_device = device
    device_fallback = None
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        device_fallback = "CUDA unavailable; using CPU"
        print(device_fallback, flush=True)
    config = config or SimConfig()
    predictor = None
    if predictor_path:
        from rlrc.ml.runtime import load_predictor

        predictor = load_predictor(
            predictor_path,
            config.as_dict(),
            FeatureBuilder(config.max_rate, config.target, config.inflight_limit).names,
        )
    check_env(RateControlEnv(config, backend, predictor))
    env = vector_env(config, count, mode, backend, predictor)
    env.seed(seed)
    builder = FeatureBuilder(
        config.max_rate, config.target, config.inflight_limit, backend, predictor is not None
    )
    manifest: dict[str, Any] = {
        "seed": seed,
        "simulator": asdict(config),
        "backend": backend,
        "features": builder.names,
        "feature_version": FEATURE_VERSION,
        "normalization": "fixed shared FeatureBuilder; Box[-1,4]",
        "device": device,
        "requested_device": requested_device,
        "device_fallback": device_fallback,
        "n_envs": count,
        "vectorization": mode,
        "steps_requested": steps,
        "validation_interval": validation_interval,
        "validation_seeds": [20001, 20002, 20003, 20004, 20005, 20006],
        "test_seeds": "40000+; never used for checkpoint selection",
        "torch": torch.__version__,
        "source_digest": source_digest(),
        "machine": machine_manifest(),
        "resume": str(resume) if resume else None,
        "predictor": str(predictor_path.resolve()) if predictor_path else None,
        "complete": False,
        "architecture": {"policy": [64, 64], "value": [64, 64], "activation": "Tanh"},
        "ppo": {
            "n_steps": 256,
            "batch_size": 256,
            "n_epochs": 6,
            "learning_rate": 0.0003,
            "gamma": 0.98,
            "gae_lambda": 0.95,
            "ent_coef": 0.01,
        },
    }
    save_json(output / "metadata.json", manifest)
    begin = time.perf_counter()
    callback = ProgressCallback(output, config, backend, predictor, validation_interval, begin)
    model = (
        PPO.load(resume, env=env, device=device)
        if resume
        else PPO(
            "MlpPolicy",
            env,
            seed=seed,
            device=device,
            verbose=0,
            tensorboard_log=str(output / "tensorboard"),
            **manifest["ppo"],
            policy_kwargs={"net_arch": dict(pi=[64, 64], vf=[64, 64])},
        )
    )
    model.tensorboard_log = str(output / "tensorboard")
    if resume:
        previous_meta = resume.parent / "metadata.json"
        if not previous_meta.exists():
            raise ValueError("Resume checkpoint requires its metadata.json")
        previous = json.loads(previous_meta.read_text())
        previous["simulator"] = SimConfig(**previous["simulator"]).as_dict()
        for key in ("features", "simulator", "feature_version"):
            if json.dumps(previous[key], sort_keys=True) != json.dumps(
                manifest[key], sort_keys=True
            ):
                raise ValueError(f"Resume mismatch: {key}")
        callback.next_check = model.num_timesteps + validation_interval
    initial_steps = model.num_timesteps
    try:
        model.learn(
            steps, callback=callback, reset_num_timesteps=not bool(resume), tb_log_name="ppo"
        )
        model.save(output / "final.zip")
        if not (output / "best.zip").exists():
            model.save(output / "best.zip")
        best = PPO.load(output / "best.zip", device="cpu")
        manifest["selected_steps"] = best.num_timesteps
        manifest["export_sha256"] = export_policy(best, output)
        manifest["final_validation"] = validation(best, config, backend, predictor)
        manifest["complete"] = True
    finally:
        # Keep a resumable checkpoint even when training is interrupted.
        model.save(output / "last.zip")
        manifest["steps_completed"] = model.num_timesteps
        manifest["elapsed_seconds"] = time.perf_counter() - begin
        manifest["steps_s"] = (model.num_timesteps - initial_steps) / max(
            0.001, manifest["elapsed_seconds"]
        )
        manifest["rss_mb_at_end"] = psutil.Process().memory_info().rss / 2**20
        if (output / "best.zip").exists():
            manifest["model_sha256"] = hashlib.sha256(
                (output / "best.zip").read_bytes()
            ).hexdigest()
        save_json(output / "metadata.json", manifest)
        env.close()
    return output / "best.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    hardware = sub.add_parser("hardware")
    hardware.add_argument("--output", type=Path, default=Path("results/ml-runs/hardware.json"))
    hardware.add_argument("--steps", type=int, default=2048)
    training = sub.add_parser("train")
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--steps", type=int, default=131072)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    training.add_argument("--n-envs", type=int, default=4)
    training.add_argument("--vec", choices=["dummy", "subproc"], default="dummy")
    training.add_argument("--backend", action="store_true")
    training.add_argument("--resume", type=Path)
    training.add_argument("--predictor", type=Path)
    training.add_argument("--validation-interval", type=int, default=32768)
    training.add_argument("--sim-config", type=Path)
    args = parser.parse_args()
    if args.command == "hardware":
        hardware_probe(args.output, args.steps)
    else:
        config = None
        if args.sim_config:
            import yaml

            config = SimConfig(**yaml.safe_load(args.sim_config.read_text()))
        print(
            train(
                args.output,
                args.steps,
                args.seed,
                args.device,
                args.n_envs,
                args.vec,
                args.backend,
                args.resume,
                args.predictor,
                args.validation_interval,
                config,
            )
        )


if __name__ == "__main__":
    main()
