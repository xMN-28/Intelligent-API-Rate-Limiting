"""Command-line entry points for local and Docker-based experiments."""

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

import uvicorn

from rlrc.config import load_settings
from rlrc.experiments import calibrate, run_suite, run_trial
from rlrc.loadgen import load_scenario, run_workload
from rlrc.report import build_report


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "ml":
        from rlrc.ml.__main__ import main as ml_main

        sys.argv = [sys.argv[0], *sys.argv[2:]]
        ml_main()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ml", help="Optional PPO training, hardware probes, and benchmarks")
    serve = commands.add_parser("serve", help="Run one service, with exactly one worker")
    serve.add_argument("service", choices=["gateway", "backend"])
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int)
    report = commands.add_parser("report", help="Rebuild charts and Markdown from raw runs")
    report.add_argument("root", type=Path)
    for name in ("run", "demo", "suite", "calibrate"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path)
        command.add_argument("--seed", type=int, default=42)
        if name != "calibrate":
            command.add_argument(
                "--scenario-file", type=Path, default=Path("configs/scenarios.yaml")
            )
            command.add_argument("--scale", type=float, default=1)
        if name == "run":
            command.add_argument("--scenario", default="demo")
            command.add_argument("--gateway-url", default="http://localhost:8000")
            command.add_argument("--backend-url", default="http://localhost:8001")
        else:
            command.add_argument("--gateway-port", type=int, default=18000)
            command.add_argument("--backend-port", type=int, default=18001)
        if name == "demo":
            command.add_argument("--scenario", default="demo")
        if name == "suite":
            command.add_argument(
                "--scenarios",
                nargs="+",
                default=[
                    "steady",
                    "ramp",
                    "burst",
                    "overload",
                    "capacity",
                    "fairness",
                    "aggressive",
                ],
            )
            command.add_argument("--repetitions", type=int, default=5)
            command.add_argument(
                "--rates",
                type=float,
                nargs=2,
                help="Uncalibrated smoke/demo override: conservative and higher RPS",
            )
            command.add_argument("--calibration", type=Path)
            command.add_argument("--bind", default="127.0.0.1")
        if name == "calibrate":
            command.add_argument(
                "--rates", type=float, nargs="+", default=[30, 60, 90, 120, 150, 180, 210, 240, 300]
            )
            command.add_argument("--duration", type=float, default=15)
            command.add_argument("--warmup", type=float, default=5)
    args = parser.parse_args()
    if args.command == "serve":
        uvicorn.run(
            f"rlrc.{args.service}:create_app",
            factory=True,
            host=args.host,
            port=args.port or (8000 if args.service == "gateway" else 8001),
            access_log=False,
        )
        return
    if args.command == "report":
        print(build_report(args.root))
        return
    cfg = load_settings()
    output = args.output or Path("results/runs") / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    if args.command == "calibrate":
        if args.duration <= 0 or args.warmup < 0 or any(r <= 0 for r in args.rates):
            parser.error("duration and rates must be positive; warmup must be nonnegative")
        asyncio.run(
            calibrate(
                cfg,
                output,
                args.rates,
                duration=args.duration,
                warmup=args.warmup,
                seed=args.seed,
                gateway_port=args.gateway_port,
                backend_port=args.backend_port,
            )
        )
        print(output / "calibration.json")
    elif args.command == "demo":
        scenario = load_scenario(args.scenario_file, args.scenario, args.scale)
        asyncio.run(
            run_trial(
                cfg,
                scenario,
                output,
                seed=args.seed,
                scenario_name=args.scenario,
                gateway_port=args.gateway_port,
                backend_port=args.backend_port,
            )
        )
        print(build_report(output))
    elif args.command == "suite":
        if args.repetitions < 1:
            parser.error("repetitions must be at least one")
        print(
            asyncio.run(
                run_suite(
                    cfg,
                    output,
                    args.scenario_file,
                    args.scenarios,
                    repetitions=args.repetitions,
                    seed=args.seed,
                    scale=args.scale,
                    rates=args.rates,
                    calibration=args.calibration,
                    gateway_port=args.gateway_port,
                    backend_port=args.backend_port,
                    bind=args.bind,
                )
            )
        )
    else:
        scenario = load_scenario(args.scenario_file, args.scenario, args.scale)
        asyncio.run(
            run_workload(
                cfg,
                scenario,
                args.gateway_url.rstrip("/"),
                output,
                seed=args.seed,
                backend_url=args.backend_url.rstrip("/"),
                scenario_name=args.scenario,
            )
        )
        print(build_report(output))


if __name__ == "__main__":
    main()
