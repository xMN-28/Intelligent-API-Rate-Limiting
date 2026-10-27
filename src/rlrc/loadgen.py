"""Bounded open-arrival HTTP workloads with explicit accounting of missed launches."""

import asyncio
import csv
import json
import platform
import random
import subprocess
import time
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import httpx
import psutil
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rlrc.config import Settings


class Phase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    duration: float = Field(gt=0)
    rate: float = Field(gt=0, le=100_000)
    end_rate: float | None = Field(None, gt=0, le=100_000)
    workers: int | None = Field(None, ge=1, le=1024)
    service_time: float | None = Field(None, ge=0, le=10)
    weights: dict[str, float] | None = None

    @model_validator(mode="after")
    def weights_valid(self) -> "Phase":
        if self.weights is not None and (
            not self.weights or any(w <= 0 for w in self.weights.values())
        ):
            raise ValueError("weights must be nonempty and positive")
        return self


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    warmup: float = Field(30, ge=0)
    max_inflight: int = Field(512, ge=1)
    max_launch_lag: float = Field(0.1, gt=0)
    fairness_comparable: bool = False
    phases: list[Phase] = Field(min_length=1)

    @property
    def duration(self) -> float:
        return sum(p.duration for p in self.phases)


def load_scenario(path: Path, name: str, scale: float = 1) -> Scenario:
    if scale <= 0:
        raise ValueError("scale must be positive")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))[name]
    scenario = Scenario.model_validate(data)
    return scenario.model_copy(
        update={
            "warmup": scenario.warmup * scale,
            "phases": [
                p.model_copy(update={"duration": p.duration * scale}) for p in scenario.phases
            ],
        }
    )


def machine_manifest() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_bytes": psutil.virtual_memory().total,
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


COLUMNS = [
    "offset",
    "phase",
    "client",
    "launched",
    "completion",
    "latency",
    "status",
    "outcome",
    "admitted",
    "reason",
    "launch_lag",
    "admission_ms",
    "upstream_ms",
    "backend_elapsed",
    "service_time",
]


class CsvSink:
    """One bounded queue and one writer; disk I/O runs outside the event loop."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=10_000)

    async def write(self) -> None:
        def open_file() -> Any:
            return self.path.open("w", encoding="utf-8", newline="")

        file = await asyncio.to_thread(open_file)
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        await asyncio.to_thread(writer.writeheader)
        try:
            done = False
            while not done:
                item = await self.queue.get()
                batch = []
                if item is None:
                    break
                batch.append(item)
                while len(batch) < 500:
                    try:
                        item = self.queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if item is None:
                        done = True
                        break
                    batch.append(item)
                await asyncio.to_thread(writer.writerows, batch)
        finally:
            await asyncio.to_thread(file.close)


async def run_workload(
    cfg: Settings,
    scenario: Scenario,
    gateway_url: str,
    output: Path,
    *,
    seed: int = 42,
    backend_url: str | None = None,
    scenario_name: str = "custom",
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Run already exists: {output}")
    names = list(cfg.clients)
    for phase in scenario.phases:
        if phase.weights and set(phase.weights) - set(names):
            raise ValueError("scenario references unregistered clients")
    manifest = {
        "settings": cfg.model_dump(exclude={"admin_token", "clients"}),
        "clients": names,
        "scenario": scenario.model_dump(),
        "seed": seed,
        "scenario_name": scenario_name,
        "machine": machine_manifest(),
        "started_at": time.time(),
        "complete": False,
        "gateway_url": gateway_url,
        "backend_url": backend_url,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    sink = CsvSink(output / "requests.csv")
    writer = asyncio.create_task(sink.write())
    rng = random.Random(seed)
    tasks: set[asyncio.Task[None]] = set()
    samples: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    sequence = 0
    stopped = asyncio.Event()

    async with (
        httpx.AsyncClient(
            timeout=cfg.upstream_timeout + 2,
            limits=httpx.Limits(
                max_connections=scenario.max_inflight, max_keepalive_connections=128
            ),
        ) as http,
        httpx.AsyncClient(timeout=2) as control_http,
    ):
        # Client construction can be slow (certificates/proxies). Start the clock afterward.
        epoch = time.monotonic() + 0.1
        measured_start = epoch + scenario.warmup
        manifest["measured_start_wall"] = time.time() + 0.1 + scenario.warmup

        async def sample_backend() -> None:
            if backend_url is None:
                return
            while not stopped.is_set():
                try:
                    response = await control_http.get(
                        backend_url + "/internal/state", headers={"X-Admin-Token": cfg.admin_token}
                    )
                    response.raise_for_status()
                    samples.append({"offset": time.monotonic() - measured_start, **response.json()})
                except httpx.HTTPError as exc:
                    samples.append(
                        {"offset": time.monotonic() - measured_start, "error": type(exc).__name__}
                    )
                try:
                    await asyncio.wait_for(stopped.wait(), timeout=1)
                except TimeoutError:
                    pass

        sampler = asyncio.create_task(sample_backend())

        async def request_one(row: dict[str, Any], key: str, request_seed: str) -> None:
            launched = time.monotonic()
            row["launch_lag"] = max(0, launched - (measured_start + row["offset"]))
            if row["launch_lag"] > scenario.max_launch_lag:
                row.update(launched=False, outcome="missed_launch", reason="scheduler_lag")
                await sink.queue.put(row)
                return
            try:
                response = await http.get(
                    gateway_url + "/proxy/work",
                    headers={"X-API-Key": key, "X-Work-Seed": request_seed},
                )
                row.update(
                    status=response.status_code,
                    admitted=response.headers.get("x-admission") == "admitted",
                    reason=response.headers.get("x-rejection-reason", ""),
                    admission_ms=response.headers.get("x-admission-duration-ms", ""),
                    upstream_ms=response.headers.get("x-upstream-duration-ms", ""),
                )
                if 200 <= response.status_code < 300:
                    try:
                        payload = response.json()
                        row["backend_elapsed"] = payload.get("elapsed", "")
                        row["service_time"] = payload.get("service_time", "")
                    except (ValueError, AttributeError):
                        pass
                if response.headers.get("x-admission") == "rejected":
                    row["outcome"] = "rejected" if response.status_code == 429 else "gateway_error"
                elif response.status_code == 504:
                    row["outcome"] = "upstream_timeout"
                elif response.status_code == 502:
                    row["outcome"] = "upstream_transport_error"
                else:
                    row["outcome"] = (
                        "success" if 200 <= response.status_code < 300 else "backend_error"
                    )
            except httpx.TimeoutException:
                row["outcome"] = "client_timeout"
            except httpx.RequestError:
                row["outcome"] = "transport_error"
            finally:
                row.update(
                    completion=time.monotonic() - measured_start,
                    latency=time.monotonic() - launched,
                )
                await sink.queue.put(row)

        def spawn(coro: Coroutine[Any, Any, None]) -> None:
            task = asyncio.create_task(coro)
            tasks.add(task)
            task.add_done_callback(tasks.discard)

        try:
            phases = list(scenario.phases)
            if scenario.warmup:
                phases.insert(
                    0,
                    Phase(
                        name="warmup",
                        duration=scenario.warmup,
                        rate=phases[0].rate,
                        weights=phases[0].weights,
                    ),
                )
            phase_start = epoch
            for phase in phases:
                await asyncio.sleep(max(0, phase_start - time.monotonic()))
                if phase.workers is not None or phase.service_time is not None:
                    if backend_url is None:
                        raise ValueError("capacity scenarios require --backend-url")
                    change = {
                        "workers": phase.workers or cfg.backend_workers,
                        "service_time": phase.service_time
                        if phase.service_time is not None
                        else cfg.service_time,
                    }
                    response = await control_http.post(
                        backend_url + "/internal/capacity",
                        json=change,
                        headers={"X-Admin-Token": cfg.admin_token},
                    )
                    response.raise_for_status()
                    events.append({"offset": time.monotonic() - measured_start, **change})
                phase_offset = 0.0
                weights = [phase.weights.get(n, 0) for n in names] if phase.weights else None
                # Relative arithmetic keeps the offered trace independent of host uptime.
                while phase_offset < phase.duration - 1e-9:
                    scheduled = phase_start + phase_offset
                    await asyncio.sleep(max(0, scheduled - time.monotonic()))
                    lag = max(0, time.monotonic() - scheduled)
                    client = rng.choices(names, weights=weights, k=1)[0]
                    row: dict[str, Any] = dict.fromkeys(COLUMNS, "")
                    row.update(
                        offset=scheduled - measured_start,
                        phase=phase.name,
                        client=client,
                        launched=False,
                        admitted=False,
                        launch_lag=lag,
                    )
                    if lag > scenario.max_launch_lag or len(tasks) >= scenario.max_inflight:
                        row["outcome"] = "missed_launch"
                        row["reason"] = (
                            "scheduler_lag"
                            if lag > scenario.max_launch_lag
                            else "generator_inflight"
                        )
                        await sink.queue.put(row)
                    else:
                        row["launched"] = True
                        spawn(request_one(row, cfg.clients[client], f"{seed}:{sequence}"))
                    fraction = phase_offset / phase.duration
                    rate = phase.rate + ((phase.end_rate or phase.rate) - phase.rate) * fraction
                    phase_offset += 1 / rate
                    sequence += 1
                    if writer.done():
                        await writer  # Surface disk failures rather than deadlocking a full queue.
                phase_start += phase.duration
            if tasks:
                await asyncio.gather(*tasks)
            manifest["complete"] = True
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            stopped.set()
            await sampler
            if not writer.done():
                await sink.queue.put(None)
            await writer
            manifest["finished_at"] = time.time()
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            (output / "backend.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
            (output / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    return output
