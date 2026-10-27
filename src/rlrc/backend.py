"""A real bounded service with explicitly simulated dependency service times."""

import asyncio
import hashlib
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import psutil
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field

from rlrc.config import Settings, load_settings


class CapacityUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workers: int = Field(ge=1, le=1024)
    service_time: float = Field(ge=0, le=10)


class WorkerPool:
    def __init__(self, workers: int, queue: int) -> None:
        self.workers = workers
        self.queue = queue
        self.active = 0
        self.waiting = 0
        self.condition = asyncio.Condition()

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        async with self.condition:
            if self.active >= self.workers:
                if self.waiting >= self.queue:
                    raise HTTPException(503, "backend_queue_full")
                self.waiting += 1
                try:
                    await self.condition.wait_for(lambda: self.active < self.workers)
                finally:
                    self.waiting -= 1
            self.active += 1
        try:
            yield
        finally:
            async with self.condition:
                self.active -= 1
                self.condition.notify_all()

    async def resize(self, workers: int) -> None:
        async with self.condition:
            self.workers = workers
            self.condition.notify_all()


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or load_settings()
    app = FastAPI(title="Controlled backend")
    pool = WorkerPool(cfg.backend_workers, cfg.backend_queue)
    app.state.pool = pool
    app.state.service_time = cfg.service_time
    registry = CollectorRegistry()
    completed = Counter("rlrc_backend_completed", "Backend completed work", registry=registry)
    active = Gauge("rlrc_backend_active", "Busy workers", registry=registry)
    waiting = Gauge("rlrc_backend_waiting", "Waiting work", registry=registry)
    capacity = Gauge("rlrc_backend_workers", "Configured workers", registry=registry)
    cpu = Gauge("rlrc_backend_cpu_percent", "Backend process CPU percent", registry=registry)
    process = psutil.Process()
    process.cpu_percent()

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ready"}

    @app.get("/metrics")
    async def metrics() -> Response:
        active.set(pool.active)
        waiting.set(pool.waiting)
        capacity.set(pool.workers)
        cpu.set(process.cpu_percent())
        return Response(generate_latest(registry), headers={"Content-Type": CONTENT_TYPE_LATEST})

    @app.post("/internal/capacity")
    async def update_capacity(
        update: CapacityUpdate, x_admin_token: Annotated[str, Header()] = ""
    ) -> dict[str, object]:
        if not secrets.compare_digest(x_admin_token, cfg.admin_token):
            raise HTTPException(403, "invalid_admin_token")
        app.state.service_time = update.service_time
        await pool.resize(update.workers)
        return update.model_dump()

    @app.get("/internal/state")
    async def state(x_admin_token: Annotated[str, Header()] = "") -> dict[str, object]:
        if not secrets.compare_digest(x_admin_token, cfg.admin_token):
            raise HTTPException(403, "invalid_admin_token")
        return {
            "active": pool.active,
            "waiting": pool.waiting,
            "workers": pool.workers,
            "service_time": app.state.service_time,
            "cpu_percent": process.cpu_percent(),
        }

    @app.api_route("/work", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def work(request: Request) -> dict[str, object]:
        started = time.monotonic()
        # Request-keyed jitter makes a seed reproduce service times across policies.
        seed = request.headers.get("x-work-seed", "0")
        fraction = int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        delay = app.state.service_time * (1 + cfg.service_jitter * (2 * fraction - 1))
        async with pool.slot():
            await asyncio.sleep(delay)
            completed.inc()
        return {"ok": True, "service_time": delay, "elapsed": time.monotonic() - started}

    @app.api_route(
        "/echo/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
    )
    async def echo(path: str, request: Request) -> dict[str, object]:
        return {
            "path": path,
            "method": request.method,
            "query": list(request.query_params.multi_items()),
            "body": (await request.body()).decode("utf-8", errors="replace"),
            "headers": dict(request.headers),
        }

    return app
