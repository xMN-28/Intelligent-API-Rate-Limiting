import os
import uuid

import pytest
from redis.asyncio import Redis
from redis.exceptions import RedisError

from irl.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(
        redis_url=os.getenv("TEST_REDIS_URL", "redis://localhost:6379/15"),
        namespace=f"test-{uuid.uuid4().hex}",
        rate=20,
        burst_seconds=1,
        clients={"alice": "a", "bob": "b"},
        policy="token_bucket",
        decision_log=str(tmp_path / "decisions.jsonl"),
    )


@pytest.fixture
async def redis_client(settings):
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        await redis.ping()
    except RedisError:
        await redis.aclose()
        if os.getenv("REQUIRE_REDIS") == "1":
            pytest.fail("Real Redis is required but unavailable")
        pytest.skip(
            "Redis unavailable; set TEST_REDIS_URL and REQUIRE_REDIS=1 to require integration"
        )
    yield redis
    keys = [key async for key in redis.scan_iter(match=f"irl:{{{settings.namespace}}}*")]
    if keys:
        await redis.delete(*keys)
    await redis.aclose()
