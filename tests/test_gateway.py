import asyncio

import httpx
import pytest

from rlrc.backend import create_app as backend_app
from rlrc.gateway import create_app


@pytest.mark.redis
async def test_real_admission_and_forwarding(redis_client, settings):
    app = create_app(settings)
    app.state.upstream_transport = httpx.ASGITransport(app=backend_app(settings))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as http:
            response = await http.post(
                "/proxy/echo/hello?q=1&q=2",
                content=b'{"hello": "world"}',
                headers={"X-API-Key": "a", "X-Custom": "kept"},
            )
            assert response.status_code == 200
            assert response.headers["x-admission"] == "admitted"
            data = response.json()
            assert data["method"] == "POST"
            assert data["query"] == [["q", "1"], ["q", "2"]]
            assert data["body"] == '{"hello": "world"}'
            assert data["headers"]["x-custom"] == "kept"
            assert "x-api-key" not in data["headers"]
            for _ in range(30):
                response = await http.get("/proxy/work", headers={"X-API-Key": "a"})
            # Concurrent burst at a frozen low rate deterministically rejects.
            await app.state.limiter.set_rate(1)
            responses = await asyncio.gather(
                *(http.get("/proxy/work", headers={"X-API-Key": "a"}) for _ in range(10))
            )
            assert any(r.status_code == 429 and "retry-after" in r.headers for r in responses)
            assert app.state.inflight == 0


async def test_unavailable_redis_fails_closed_and_health_stays_up(settings):
    settings.redis_url = "redis://127.0.0.1:1/0"
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as http:
            assert (await http.get("/healthz")).status_code == 200
            assert (await http.get("/readyz")).status_code == 503
            assert (await http.get("/proxy/work", headers={"X-API-Key": "a"})).status_code == 503
            assert app.state.inflight == 0


@pytest.mark.parametrize("path", ["internal/capacity", "metrics", "echo/%252e%252e/internal/state"])
async def test_public_proxy_cannot_access_control_plane(settings, path):
    settings.policy = "none"
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as http:
            assert (await http.get("/proxy/" + path, headers={"X-API-Key": "a"})).status_code == 404


async def test_unknown_identity_and_body_limit(settings):
    settings.policy = "none"
    settings.max_body_bytes = 4
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as http:
            assert (
                await http.get("/proxy/work", headers={"X-API-Key": "invented"})
            ).status_code == 401
            assert (
                await http.post("/proxy/work", content=b"12345", headers={"X-API-Key": "a"})
            ).status_code == 413


@pytest.mark.parametrize("failure,status", [(httpx.ReadTimeout, 504), (httpx.ConnectError, 502)])
async def test_upstream_failure_classification(settings, failure, status):
    settings.policy = "none"

    def fail(request):
        raise failure("test failure", request=request)

    app = create_app(settings)
    app.state.upstream_transport = httpx.MockTransport(fail)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as http:
            response = await http.get("/proxy/work", headers={"X-API-Key": "a"})
            assert response.status_code == status
            assert response.headers["x-admission"] == "admitted"
            assert app.state.inflight == 0
            assert len(app.state.controller.observations) == 1


async def test_inflight_ceiling_and_cancellation_release(settings):
    settings.policy = "none"
    settings.inflight_limit = 1
    entered = asyncio.Event()

    async def block(request):
        entered.set()
        await asyncio.Event().wait()

    app = create_app(settings)
    app.state.upstream_transport = httpx.MockTransport(block)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as http:
            task = asyncio.create_task(http.get("/proxy/work", headers={"X-API-Key": "a"}))
            await entered.wait()
            assert (await http.get("/proxy/work", headers={"X-API-Key": "b"})).status_code == 429
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert app.state.inflight == 0


@pytest.mark.redis
async def test_failed_controller_publication_is_logged_and_rolls_back(redis_client, settings):
    import json
    import time
    from pathlib import Path
    from unittest.mock import AsyncMock

    from redis.exceptions import ConnectionError

    from rlrc.controller import Observation

    settings.policy = "adaptive"
    settings.controller.interval = 0.01
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        app.state.limiter.set_rate = AsyncMock(side_effect=ConnectionError("test outage"))
        for _ in range(40):
            app.state.controller.observe(Observation(time.monotonic(), 1))
        await asyncio.sleep(0.08)
        assert app.state.controller.rate == settings.rate
        assert await app.state.limiter.get_rate() == settings.rate
    decisions = [json.loads(line) for line in Path(settings.decision_log).read_text().splitlines()]
    assert decisions
    assert any(d["reason"] == "backend_unhealthy" for d in decisions)
    assert all(not d["applied"] for d in decisions)


@pytest.mark.redis
async def test_rejections_never_enter_backend_health_samples(redis_client, settings):
    settings.rate = 1
    settings.policy = "token_bucket"
    app = create_app(settings)
    app.state.upstream_transport = httpx.MockTransport(lambda r: httpx.Response(200))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as http:
            responses = await asyncio.gather(
                *(http.get("/proxy/work", headers={"X-API-Key": "a"}) for _ in range(20))
            )
            admitted = sum(r.headers["x-admission"] == "admitted" for r in responses)
            assert admitted == 1
            assert len(app.state.controller.observations) == admitted
