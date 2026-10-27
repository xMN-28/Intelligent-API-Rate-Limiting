"""Staged matched-seed simulator and real HTTP experiments; never mix their evidence."""

import argparse
import asyncio
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from rlrc.config import Settings, load_settings
from rlrc.experiments import calibrate, run_trial
from rlrc.loadgen import Phase, Scenario, load_scenario
from rlrc.ml.evaluation import evaluate_episode, save_json
from rlrc.ml.runtime import NumpyActor, load_predictor
from rlrc.ml.simulator import SimConfig, Simulator
from rlrc.report import build_report, summarize

POLICIES = [
    "none",
    "token_bucket",
    "static_high",
    "sliding_window",
    "sliding_high",
    "adaptive",
    "rl",
    "rl_safe",
    "predictive",
    "hybrid",
    "rl_backend",
]
FAMILIES = [
    "steady",
    "ramp",
    "falling",
    "burst",
    "overload",
    "sudden",
    "gradual",
    "fairness",
    "aggressive",
    "unseen",
]
SIM_POLICIES = [*POLICIES, "rl_prediction"]


class ActorAdapter:
    def __init__(self, actor: NumpyActor) -> None:
        self.actor = actor

    def predict(self, observation: Any, deterministic: bool = True) -> tuple[int, None]:
        return self.actor.action(observation), None


def artifact_directory(local: str, exported: str) -> str:
    directory = Path("models") / local
    return str(directory if directory.exists() else Path("assets/models") / exported)


def controller_config(cfg: Settings, name: str, rate: float) -> Settings:
    policy = {
        "static_high": "token_bucket",
        "sliding_high": "sliding_window",
        "rl_backend": "rl",
        "adaptive_tuned": "adaptive",
    }.get(name, name)
    data = cfg.model_dump()
    data.update(policy=policy, rate=rate)
    if name == "adaptive_tuned":
        parameters = json.loads(Path("models/aimd-tuned.json").read_text())["selected"][
            "parameters"
        ]
        data["controller"].update(parameters)
    data["ml"].update(
        model_path=artifact_directory("ppo-gateway", "ppo-gateway"),
        backend_telemetry=False,
        predictor_path=None,
    )
    if name == "rl_backend":
        data["ml"].update(
            model_path=artifact_directory("ppo-backend-v2", "ppo-backend"), backend_telemetry=True
        )
    if name in {"hybrid", "predictive"}:
        data["ml"]["predictor_path"] = str(
            Path(artifact_directory("overload-predictor", "overload-predictor"))
            / "predictor.joblib"
        )
    if name == "hybrid":
        data["ml"]["model_path"] = artifact_directory("ppo-prediction-v2", "ppo-prediction")
    return Settings.model_validate(data)


def simulator_suite(
    output: Path, cfg: Settings, repetitions: int = 5, policies: list[str] | None = None
) -> Path:
    from rlrc.ml.features import FeatureBuilder

    if cfg.sliding_window != 1:
        raise ValueError("Simulator Sliding Window comparator currently requires a 1s window")
    selected = policies or SIM_POLICIES
    if not set(selected) <= set(SIM_POLICIES):
        raise ValueError("Unknown simulator policy")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    predictor = (
        load_predictor(
            Path(artifact_directory("overload-predictor", "overload-predictor"))
            / "predictor.joblib",
            {
                "max_rate": cfg.controller.max_rate,
                "min_rate": cfg.controller.min_rate,
                "target": cfg.controller.latency_target,
                "clients": len(cfg.clients),
                "inflight_limit": cfg.inflight_limit,
            },
            FeatureBuilder(
                cfg.controller.max_rate, cfg.controller.latency_target, cfg.inflight_limit
            ).names,
        )
        if set(selected) & {"hybrid", "rl_prediction", "predictive"}
        else None
    )
    models = {
        name: ActorAdapter(
            NumpyActor(
                Path(controller_config(cfg, name, cfg.rate).ml.model_path),
                controller_config(cfg, name, cfg.rate),
            )
        )
        for name, needed in [
            ("rl", {"rl", "rl_safe"}),
            ("rl_backend", {"rl_backend"}),
            ("hybrid", {"hybrid", "rl_prediction"}),
        ]
        if set(selected) & needed
    }
    results = []
    begin = time.perf_counter()
    for family in FAMILIES:
        for repeat in range(repetitions):
            names = selected.copy()
            random.Random(40000 + repeat).shuffle(names)
            for name in names:
                config = SimConfig(
                    duration=180,
                    family=family,
                    initial_rate=cfg.rate,
                    max_rate=cfg.controller.max_rate,
                    min_rate=cfg.controller.min_rate,
                    target=cfg.controller.latency_target,
                    inflight_limit=cfg.inflight_limit,
                    queue=cfg.backend_queue,
                    workers=cfg.backend_workers,
                    service_time=cfg.service_time,
                    clients=len(cfg.clients),
                    timeout=cfg.upstream_timeout,
                    burst=cfg.burst_seconds,
                    jitter=cfg.service_jitter,
                )
                policy = (
                    "rl_prediction"
                    if name in {"hybrid", "rl_prediction"}
                    else "rl"
                    if name in {"rl_backend", "rl_safe"}
                    else name
                )
                result = evaluate_episode(
                    config,
                    40000 + repeat,
                    policy,
                    model=models.get(
                        "rl" if name == "rl_safe" else "hybrid" if name == "rl_prediction" else name
                    ),
                    backend=name == "rl_backend",
                    safe=name in {"rl_safe", "hybrid"},
                    predictor=predictor
                    if name in {"hybrid", "predictive", "rl_prediction"}
                    else None,
                    warmup=30,
                )
                result["label"] = name
                save_json(output / f"{family}-{repeat}-{name}.json", result)
                results.append(
                    {
                        k: v
                        for k, v in result.items()
                        if k not in {"series", "budgets", "interventions"}
                    }
                )
        print(f"Simulator held-out family complete: {family}", flush=True)
    save_json(output / "summaries.json", results)
    save_json(
        output / "protocol.json",
        {
            "seeds": [40000, 40000 + repetitions - 1],
            "duration": 180,
            "warmup": 30,
            "seconds": time.perf_counter() - begin,
            "policies": selected,
            "family": FAMILIES,
            "settings": cfg.model_dump(exclude={"clients", "admin_token"}),
            "evidence": "simulator, NOT real HTTP",
        },
    )
    return output


async def fidelity(output: Path, cfg: Settings) -> Path:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    calibration = await calibrate(
        cfg, output / "calibration", [10, 20, 30, 40, 60], duration=15, warmup=5, seed=71000
    )
    comparisons = []
    for rate in [10, 20, 30, 40, 60]:
        real = summarize(output / "calibration" / f"rate-{rate}")
        sim = Simulator(
            SimConfig(
                duration=20,
                family="fixed",
                fixed_offered=rate,
                workers=cfg.backend_workers,
                service_time=cfg.service_time,
                target=cfg.controller.latency_target,
                queue=cfg.backend_queue,
                clients=len(cfg.clients),
                timeout=cfg.upstream_timeout,
                inflight_limit=cfg.inflight_limit,
                jitter=cfg.service_jitter,
                overhead=0.005,
                record_responses=True,
            ),
            71000,
        )
        for _ in range(20):
            sim.advance(cfg.rate, "none")
        tail = sim.drain()
        responses = [r for row in [*sim.history, tail] for r in row["responses"] if r[0] >= 5]
        latency = [r[1] for r in responses if r[2]]
        with (output / "calibration" / f"rate-{rate}" / "requests.csv").open() as file:
            requests = [r for r in csv.DictReader(file) if float(r["offset"]) >= 0]
        with (output / "calibration" / f"rate-{rate}" / "backend.json").open() as file:
            backend = json.load(file)
        comparisons.append(
            {
                "offered": rate,
                "real": real,
                "sim_success_rps": len(latency) / 15,
                "sim_goodput_rps": sum(x <= cfg.controller.latency_target for x in latency) / 15,
                "sim_p95": float(np.quantile(latency, 0.95)) if latency else None,
                "sim_backend_errors": sum(r[3] == "overflow" for r in responses),
                "sim_mean_queue": float(np.mean([r["queue"] for r in sim.history[5:]])),
                "real_mean_queue": float(
                    np.mean([r["waiting"] for r in backend if r["offset"] >= 0 and "waiting" in r])
                ),
                "real_service_elapsed_p50": float(
                    np.median(
                        [float(r["backend_elapsed"]) for r in requests if r.get("backend_elapsed")]
                    )
                ),
            }
        )
    save_json(output / "steady-fidelity.json", comparisons)
    # Dynamic overload/recovery calibration is separate from final policy comparisons.
    scenario = Scenario(
        warmup=10,
        phases=[
            Phase(name="normal", duration=20, rate=60, workers=4, service_time=0.1),
            Phase(name="degraded", duration=20, rate=60, workers=1, service_time=0.1),
            Phase(name="recovered", duration=30, rate=60, workers=4, service_time=0.1),
        ],
    )
    await run_trial(
        cfg.model_copy(update={"policy": "none"}),
        scenario,
        output / "dynamic",
        seed=71001,
        scenario_name="fidelity-dynamic",
    )
    dynamic = Simulator(
        SimConfig(
            duration=80,
            family="fixed",
            fixed_offered=60,
            workers=4,
            service_time=0.1,
            queue=cfg.backend_queue,
            clients=len(cfg.clients),
            target=cfg.controller.latency_target,
            timeout=cfg.upstream_timeout,
            inflight_limit=cfg.inflight_limit,
            jitter=cfg.service_jitter,
            changes=((30, 1, 0.1), (50, 4, 0.1)),
        ),
        71001,
    )
    for _ in range(80):
        dynamic.advance(cfg.rate, "none")
    save_json(output / "dynamic-simulation.json", dynamic.history)
    summarize(output / "dynamic")
    save_json(
        output / "protocol.json",
        {
            "fit": (
                "No retrospective fit to final evaluation. Nominal service times "
                "and 5ms transport overhead are tested as-is."
            ),
            "calibration": calibration,
            "seeds": [71000, 71001],
        },
    )
    return output


async def real_suite(
    output: Path,
    cfg: Settings,
    calibration_path: Path,
    repetitions: int = 3,
    policies: list[str] | None = None,
    smoke: bool = False,
    scenario_file: Path | None = None,
    scenario_name: str = "ml-capacity",
) -> Path:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    calibration = json.loads(calibration_path.read_text())
    for field in (
        "backend_workers",
        "backend_queue",
        "service_time",
        "service_jitter",
        "inflight_limit",
        "upstream_timeout",
    ):
        if calibration["settings"][field] != getattr(cfg, field):
            raise ValueError(f"Calibration mismatch: {field}")
    if calibration["latency_target"] != cfg.controller.latency_target:
        raise ValueError("Calibration latency target mismatch")
    rates = calibration["static_rates"]
    names = policies or POLICIES
    # Chosen BEFORE inspecting final outcomes; 60s recovery avoids handicapping the slow AIMD.
    scenario = Scenario(
        warmup=30,
        fairness_comparable=True,
        phases=[
            Phase(name="normal", duration=20, rate=60, workers=4, service_time=0.1),
            Phase(name="degraded", duration=20, rate=60, workers=1, service_time=0.1),
            Phase(name="recovered", duration=60, rate=60, workers=4, service_time=0.1),
        ],
    )
    if smoke:
        scenario = Scenario(warmup=2, phases=[Phase(name="smoke", duration=8, rate=45)])
    elif scenario_file is not None:
        scenario = load_scenario(scenario_file, scenario_name)
    schedule: list[dict[str, Any]] = []
    for repeat in range(repetitions):
        order = names.copy()
        random.Random(81000 + repeat).shuffle(order)
        for name in order:
            schedule.append({"repeat": repeat, "label": name, "seed": 81000 + repeat})
    save_json(
        output / "protocol.json",
        {
            "evidence": "real HTTP/Redis",
            "smoke": smoke,
            "scenario": scenario.model_dump(),
            "schedule": schedule,
            "calibration": str(calibration_path),
            "static_rates": rates,
            "source_commit": __import__(
                "rlrc.loadgen", fromlist=["machine_manifest"]
            ).machine_manifest(),
            "settings": cfg.model_dump(exclude={"clients", "admin_token"}),
        },
    )
    for index, item in enumerate(schedule):
        print(f"HTTP [{index + 1}/{len(schedule)}] {item}", flush=True)
        rate = rates[1] if item["label"] in {"static_high", "sliding_high"} else rates[0]
        settings = controller_config(cfg, item["label"], rate)
        run = output / f"run-{index:03d}"
        await run_trial(settings, scenario, run, seed=item["seed"], scenario_name=scenario_name)
        summary = summarize(run)
        summary["label"] = item["label"]
        save_json(run / "ml-summary.json", summary)
        save_json(run / "experiment.json", item)
        if not summary["valid_generator"]:
            print("INVALID GENERATOR RUN retained; no comparative claim permitted", flush=True)
    return build_report(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["sim", "fidelity", "real"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--policies", nargs="+")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--scenario-file", type=Path)
    parser.add_argument("--scenario", default="ml-capacity")
    args = parser.parse_args()
    cfg = load_settings()
    if args.command == "sim":
        if args.calibration is not None:
            cfg = cfg.model_copy(
                update={"rate": json.loads(args.calibration.read_text())["static_rates"][0]}
            )
        simulator_suite(args.output, cfg, args.repetitions, args.policies)
    elif args.command == "fidelity":
        asyncio.run(fidelity(args.output, cfg))
    else:
        if args.calibration is None:
            parser.error("real requires --calibration")
        asyncio.run(
            real_suite(
                args.output,
                cfg,
                args.calibration,
                args.repetitions,
                args.policies,
                args.smoke,
                args.scenario_file,
                args.scenario,
            )
        )


if __name__ == "__main__":
    main()
