import asyncio

import httpx
import pytest
from fastapi import HTTPException

from rlrc.backend import WorkerPool, create_app


async def test_bounded_queue_rejects_excess_and_cleans_cancelled_waiter():
    pool = WorkerPool(1, 1)
    async with pool.slot():

        async def wait():
            async with pool.slot():
                pass

        task = asyncio.create_task(wait())
        await asyncio.sleep(0.01)
        assert pool.waiting == 1
        with pytest.raises(HTTPException) as exc:
            async with pool.slot():
                pass
        assert exc.value.status_code == 503
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert pool.waiting == 0
    assert pool.active == 0


async def test_capacity_shrink_drains_existing_work():
    pool = WorkerPool(2, 0)
    async with pool.slot(), pool.slot():
        await pool.resize(1)
        assert pool.active == 2
        with pytest.raises(HTTPException):
            async with pool.slot():
                pass
    assert pool.active == 0


async def test_internal_controls_require_token_and_metrics_are_real(settings):
    app = create_app(settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://backend"
    ) as http:
        assert (
            await http.post("/internal/capacity", json={"workers": 2, "service_time": 0.01})
        ).status_code == 403
        response = await http.post(
            "/internal/capacity",
            json={"workers": 2, "service_time": 0.01},
            headers={"X-Admin-Token": settings.admin_token},
        )
        assert response.status_code == 200
        assert (await http.get("/work")).json()["ok"]
        metrics = (await http.get("/metrics")).text
        assert "rlrc_backend_completed_total 1.0" in metrics
        assert "rlrc_backend_workers 2.0" in metrics
