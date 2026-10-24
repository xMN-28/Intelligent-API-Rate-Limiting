"""Single-process HTTP admission gateway. No retries and no admission queue."""

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from redis.asyncio import Redis
from redis.exceptions import RedisError

from irl.config import Settings, load_settings
from irl.controller import FeedbackController, Observation
from irl.limiter import RedisLimiter
from irl.metrics import GatewayMetrics

LOG = logging.getLogger(__name__)
HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


def safe_headers(headers: httpx.Headers | Any) -> list[tuple[str, str]]:
    connection = headers.get("connection", "")
    excluded = HOP_HEADERS | {h.strip().lower() for h in connection.split(",")}
    items = headers.multi_items() if isinstance(headers, httpx.Headers) else headers.items()
    return [(k, v) for k, v in items if k.lower() not in excluded]


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or load_settings()
    metrics = GatewayMetrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        redis = Redis.from_url(
            cfg.redis_url, decode_responses=True, socket_connect_timeout=0.5, socket_timeout=0.5
        )
        limiter = RedisLimiter(redis, cfg)
        # Start even if Redis is down: liveness stays up, readiness/admission fail closed.
        try:
            await limiter.initialize()
        except RedisError:
            LOG.warning("Redis unavailable at startup; admission will fail closed")
        transport = getattr(app.state, "upstream_transport", None)
        http = httpx.AsyncClient(
            base_url=cfg.upstream.rstrip("/") + "/",
            timeout=cfg.upstream_timeout,
            limits=httpx.Limits(
                max_connections=cfg.inflight_limit, max_keepalive_connections=cfg.inflight_limit
            ),
            follow_redirects=False,
            transport=transport,
        )
        controller = FeedbackController(cfg.controller, cfg.rate, time.monotonic())
        app.state.redis = redis
        app.state.limiter = limiter
        app.state.http = http
        app.state.controller = controller
        app.state.inflight = 0
        metrics.budget.set(cfg.rate)

        async def control_loop() -> None:
            log_path = Path(cfg.decision_log)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            output = await asyncio.to_thread(log_path.open, "a", encoding="utf-8", buffering=1)
            try:
                while True:
                    await asyncio.sleep(cfg.controller.interval)
                    decision = controller.evaluate(time.monotonic())
                    record = decision.as_dict() | {"wall_time": time.time(), "applied": False}
                    try:
                        await limiter.initialize()
                        await limiter.set_rate(decision.rate)
                        record["applied"] = True
                        metrics.budget.set(decision.rate)
                    except RedisError as exc:
                        controller.rate = decision.previous_rate
                        record["error"] = type(exc).__name__
                    metrics.decisions.labels(decision.reason).inc()
                    await asyncio.to_thread(output.write, json.dumps(record) + "\n")
            finally:
                await asyncio.to_thread(output.close)

        task = asyncio.create_task(control_loop()) if cfg.policy == "adaptive" else None
        app.state.controller_task = task
        try:
            yield
        finally:
            if task:
                task.cancel()
                try:
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                except Exception:
                    LOG.exception("Adaptive controller failed")
            await http.aclose()
            await redis.aclose()

    app = FastAPI(title="Intelligent API Rate Limiting", lifespan=lifespan)
    app.state.metrics = metrics
    app.state.settings = cfg

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/readyz")
    async def ready() -> Response:
        task = app.state.controller_task
        if task is not None and task.done():
            return JSONResponse({"status": "controller_unavailable"}, status_code=503)
        try:
            await app.state.redis.ping()
        except RedisError:
            return JSONResponse({"status": "redis_unavailable"}, status_code=503)
        return JSONResponse({"status": "ready", "policy": cfg.policy})

    @app.get("/metrics")
    async def metric_endpoint() -> Response:
        return Response(
            generate_latest(metrics.registry), headers={"Content-Type": CONTENT_TYPE_LATEST}
        )

    @app.api_route(
        "/proxy/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
    )
    async def proxy(path: str, request: Request) -> Response:
        client = next(
            (name for name, key in cfg.clients.items() if key == request.headers.get("x-api-key")),
            None,
        )

        def reject(status: int, reason: str, retry: str | None = None) -> Response:
            metrics.requests.labels(client or "unknown", reason).inc()
            headers = {"X-Admission": "rejected", "X-Rejection-Reason": reason}
            if retry:
                headers["Retry-After"] = retry
            return JSONResponse({"detail": reason}, status_code=status, headers=headers)

        if client is None:
            return reject(401, "unknown_client")
        # Explicit public upstream surface; never forward backend control/metrics paths.
        if (
            path.split("/", 1)[0] not in {"work", "echo"}
            or any(segment in {".", ".."} for segment in path.split("/"))
            or "%" in path
            or "\\" in path
        ):
            return reject(404, "route_not_exposed")
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > cfg.max_body_bytes:
                return reject(413, "body_too_large")
        # Reservation is synchronous, so coroutines cannot race this single-process counter.
        if app.state.inflight >= cfg.inflight_limit:
            return reject(429, "inflight_limit", "1")
        app.state.inflight += 1
        admitted = False
        started: float | None = None
        try:
            check_started = time.perf_counter()
            if cfg.policy != "none":
                try:
                    await app.state.limiter.initialize()
                    admission = await app.state.limiter.check(client)
                except RedisError:
                    return reject(503, "redis_unavailable", "1")
                if not admission.allowed:
                    return reject(429, admission.reason, admission.retry_header)
                metrics.budget.set(admission.rate)
            overhead = time.perf_counter() - check_started
            metrics.overhead.observe(overhead)
            admitted = True
            metrics.admitted.labels(client).inc()
            metrics.inflight.inc()
            app.state.controller.admit(time.monotonic())
            started = time.monotonic()
            headers = [
                (k, v)
                for k, v in safe_headers(request.headers)
                if k.lower() not in {"x-api-key", "x-admin-token", "accept-encoding"}
            ]
            headers.append(("accept-encoding", "identity"))
            # Build a relative URL ourselves; user input cannot select an upstream host.
            url = httpx.URL(path="/" + path, query=request.scope["query_string"])
            upstream = await app.state.http.request(
                request.method, url, content=bytes(body), headers=headers
            )
            elapsed = time.monotonic() - started
            app.state.controller.observe(
                Observation(time.monotonic(), elapsed, upstream.status_code >= 500)
            )
            metrics.latency.observe(elapsed)
            metrics.requests.labels(
                client, "upstream_error" if upstream.status_code >= 500 else "forwarded"
            ).inc()
            response = Response(upstream.content, status_code=upstream.status_code)
            # HTTPX decodes content; don't forward the original content-encoding/length.
            for key, value in safe_headers(upstream.headers):
                if key.lower() not in {"content-encoding", "x-admission", "x-rejection-reason"}:
                    response.headers.append(key, value)
            response.headers["X-Admission"] = "admitted"
            response.headers["X-Admission-Duration-Ms"] = f"{overhead * 1000:.3f}"
            response.headers["X-Upstream-Duration-Ms"] = f"{elapsed * 1000:.3f}"
            return response
        except httpx.TimeoutException:
            elapsed = time.monotonic() - (started or time.monotonic())
            app.state.controller.observe(Observation(time.monotonic(), elapsed, True, True))
            metrics.latency.observe(elapsed)
            metrics.requests.labels(client, "upstream_timeout").inc()
            return JSONResponse(
                {"detail": "upstream_timeout"}, status_code=504, headers={"X-Admission": "admitted"}
            )
        except httpx.RequestError:
            elapsed = time.monotonic() - (started or time.monotonic())
            app.state.controller.observe(Observation(time.monotonic(), elapsed, True))
            metrics.latency.observe(elapsed)
            metrics.requests.labels(client, "upstream_transport_error").inc()
            return JSONResponse(
                {"detail": "upstream_transport_error"},
                status_code=502,
                headers={"X-Admission": "admitted"},
            )
        finally:
            app.state.inflight -= 1
            if admitted:
                metrics.inflight.dec()

    return app
