"""Own and clean up real service subprocesses for isolated benchmark trials."""

import asyncio
import json
import os
import random
import socket
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx

from irl.config import Settings
from irl.loadgen import Phase, Scenario, load_scenario, run_workload
from irl.report import build_report, summarize


async def wait_ready(
    url: str, process: subprocess.Popen[bytes], deadline_seconds: float = 20
) -> None:
    async with httpx.AsyncClient(timeout=1) as http:
        deadline = asyncio.get_running_loop().time() + deadline_seconds
        while asyncio.get_running_loop().time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Service exited with {process.returncode}; inspect service logs"
                )
            try:
                if (await http.get(url)).status_code == 200:
                    return
            except httpx.RequestError:
                pass
            await asyncio.sleep(0.1)
    raise TimeoutError(f"Service not ready: {url}; inspect service logs and Redis connectivity")


async def run_trial(
    cfg: Settings,
    scenario: Scenario,
    output: Path,
    *,
    seed: int,
    scenario_name: str,
    gateway_port: int = 18000,
    backend_port: int = 18001,
    bind: str = "127.0.0.1",
) -> Path:
    for port in (gateway_port, backend_port):
        with socket.socket() as probe:
            probe.bind((bind, port))
    if gateway_port == backend_port:
        raise ValueError("Gateway and backend ports must differ")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    cfg = cfg.model_copy(
        update={
            "upstream": f"http://127.0.0.1:{backend_port}",
            "namespace": f"trial-{uuid.uuid4().hex}",
            "decision_log": str((output / "decisions.jsonl").resolve()),
        }
    )
    config_path = output / "service-config.json"
    config_path.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if not key.startswith("IRL_")}
    env["IRL_CONFIG"] = str(config_path.resolve())
    children: list[subprocess.Popen[bytes]] = []
    logs = []
    try:
        for name, port in (("backend", backend_port), ("gateway", gateway_port)):
            log = (output / f"{name}.log").open("wb")
            logs.append(log)
            child = await asyncio.to_thread(
                subprocess.Popen,
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    f"irl.{name}:create_app",
                    "--factory",
                    "--host",
                    bind,
                    "--port",
                    str(port),
                    "--no-access-log",
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            children.append(child)
            endpoint = "/readyz" if name == "gateway" else "/healthz"
            await wait_ready(f"http://127.0.0.1:{port}" + endpoint, child)
        await run_workload(
            cfg,
            scenario,
            f"http://127.0.0.1:{gateway_port}",
            output,
            seed=seed,
            backend_url=cfg.upstream,
            scenario_name=scenario_name,
        )
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:
                    await asyncio.to_thread(child.wait, timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    await asyncio.to_thread(child.wait)
        for log in logs:
            log.close()
        # Don't leave demo API keys or the admin token in shareable artifacts.
        config_path.unlink(missing_ok=True)
    return output


async def calibrate(
    cfg: Settings,
    output: Path,
    rates: list[float],
    *,
    duration: float = 15,
    warmup: float = 5,
    seed: int = 42,
    gateway_port: int = 18000,
    backend_port: int = 18001,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    results = []
    for rate in sorted(rates):
        print(f"Calibration: {rate:g} offered RPS", flush=True)
        scenario = Scenario(
            warmup=warmup, phases=[Phase(name="calibration", duration=duration, rate=rate)]
        )
        trial_cfg = cfg.model_copy(update={"policy": "none"})
        run = await run_trial(
            trial_cfg,
            scenario,
            output / f"rate-{rate:g}",
            seed=seed,
            scenario_name="calibration",
            gateway_port=gateway_port,
            backend_port=backend_port,
        )
        summary = summarize(run)
        success_fraction = summary["outcomes"].get("success", 0) / max(1, summary["launched"])
        summary["passes"] = bool(
            summary["valid_generator"]
            and success_fraction >= 0.99
            and summary["latency_p95"] is not None
            and summary["latency_p95"] <= cfg.controller.latency_target
        )
        summary["offered_rate"] = rate
        results.append(summary)
    passing = [r["offered_rate"] for r in results if r["passes"]]
    if not passing:
        raise RuntimeError(
            "No calibration rate met the SLO; lower the sweep or fix generator errors"
        )
    capacity = max(passing)
    result = {
        "safe_offered_rps": capacity,
        "static_rates": [max(1, 0.6 * capacity), max(1, 0.9 * capacity)],
        "latency_target": cfg.controller.latency_target,
        "saturated_upper_bound_found": any(
            not r["passes"] for r in results if r["offered_rate"] > capacity
        ),
        "settings": cfg.model_dump(exclude={"clients", "admin_token"}),
        "runs": results,
    }
    (output / "calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    build_report(output)
    return result


async def run_suite(
    cfg: Settings,
    output: Path,
    scenario_file: Path,
    names: list[str],
    *,
    repetitions: int = 5,
    seed: int = 42,
    scale: float = 1,
    rates: list[float] | None = None,
    calibration: Path | None = None,
    gateway_port: int = 18000,
    backend_port: int = 18001,
    bind: str = "127.0.0.1",
) -> Path:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    calibrated = calibration is not None
    if calibration:
        calibration_data = json.loads(calibration.read_text())
        if calibration_data["latency_target"] != cfg.controller.latency_target:
            raise ValueError("Calibration latency target differs from configured target")
        for field in (
            "backend_workers",
            "backend_queue",
            "service_time",
            "service_jitter",
            "inflight_limit",
            "upstream_timeout",
        ):
            if calibration_data["settings"][field] != getattr(cfg, field):
                raise ValueError(f"Calibration has different {field}; recalibrate")
        rates = calibration_data["static_rates"]
    if rates is None:
        calibration_data = await calibrate(
            cfg,
            output / "calibration",
            [30, 60, 90, 120, 150, 180, 210, 240, 300],
            gateway_port=gateway_port,
            backend_port=backend_port,
        )
        rates = calibration_data["static_rates"]
        calibrated = True
    if len(rates) != 2 or any(rate < 1 for rate in rates):
        raise ValueError("Provide exactly two positive static rates (conservative, higher)")
    if not cfg.controller.min_rate <= rates[0] <= cfg.controller.max_rate:
        raise ValueError("Conservative starting rate must lie within frozen controller bounds")
    schedule: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        for name in names:
            policies = [
                ("none", rates[0]),
                ("token_bucket", rates[0]),
                ("token_bucket", rates[1]),
                ("sliding_window", rates[0]),
                ("sliding_window", rates[1]),
                ("adaptive", rates[0]),
            ]
            random.Random(seed + repetition).shuffle(policies)
            for policy, rate in policies:
                schedule.append(
                    {
                        "repetition": repetition,
                        "scenario": name,
                        "policy": policy,
                        "rate": rate,
                        "seed": seed + repetition,
                    }
                )
    (output / "suite.json").write_text(
        json.dumps({"calibrated": calibrated, "scale": scale, "schedule": schedule}, indent=2)
    )
    evaluation = output / "evaluation"
    evaluation.mkdir()
    for index, item in enumerate(schedule):
        print(f"[{index + 1}/{len(schedule)}] {item}", flush=True)
        trial_cfg = cfg.model_copy(update={"policy": item["policy"], "rate": item["rate"]})
        scenario = load_scenario(scenario_file, item["scenario"], scale)
        await run_trial(
            trial_cfg,
            scenario,
            evaluation / f"run-{index:03d}",
            seed=item["seed"],
            scenario_name=item["scenario"],
            gateway_port=gateway_port,
            backend_port=backend_port,
            bind=bind,
        )
    return build_report(evaluation)
