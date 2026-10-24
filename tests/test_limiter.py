import asyncio

import pytest

from irl.limiter import RedisLimiter

pytestmark = pytest.mark.redis


@pytest.mark.parametrize("policy", ["token_bucket", "sliding_window"])
async def test_concurrent_admission_is_atomic(redis_client, settings, policy):
    settings.policy = policy
    limiter = RedisLimiter(redis_client, settings)
    await limiter.initialize()
    outcomes = await asyncio.gather(*(limiter.check("alice", now=100) for _ in range(100)))
    assert sum(o.allowed for o in outcomes) == 10
    other = await asyncio.gather(*(limiter.check("bob", now=100) for _ in range(100)))
    assert sum(o.allowed for o in other) == 10
    denied = await limiter.check("alice", now=100)
    assert not denied.allowed
    assert denied.retry_after > 0
    assert denied.retry_header == "1"


@pytest.mark.parametrize("policy", ["token_bucket", "sliding_window"])
async def test_client_denial_does_not_consume_global_budget(redis_client, settings, policy):
    settings.policy = policy
    limiter = RedisLimiter(redis_client, settings)
    await limiter.initialize()
    for _ in range(10):
        assert (await limiter.check("alice", now=100)).allowed
    for _ in range(50):
        assert not (await limiter.check("alice", now=100)).allowed
    assert all([(await limiter.check("bob", now=100)).allowed for _ in range(10)])


async def test_token_refill_and_idle_burst_cap(redis_client, settings):
    limiter = RedisLimiter(redis_client, settings)
    await limiter.initialize()
    for _ in range(10):
        await limiter.check("alice", now=100)
    assert not (await limiter.check("alice", now=100.099)).allowed
    assert (await limiter.check("alice", now=100.1)).allowed
    outcomes = [await limiter.check("alice", now=200) for _ in range(20)]
    assert sum(o.allowed for o in outcomes) == 10
    assert await redis_client.pttl(limiter.client_key("alice")) > 0


async def test_rate_update_settles_old_rate_and_does_not_mint_tokens(redis_client, settings):
    limiter = RedisLimiter(redis_client, settings)
    await limiter.initialize()
    for _ in range(10):
        await limiter.check("alice", now=100)
    await limiter.set_rate(40, now=100)
    assert not (await limiter.check("alice", now=100)).allowed
    assert (await limiter.check("alice", now=100.05)).allowed
    await limiter.set_rate(2, now=100.05)
    state = await redis_client.hgetall(limiter.client_key("alice"))
    assert float(state["tokens"]) <= 1
    assert await limiter.get_rate() == 2


async def test_sliding_boundary_and_cleanup(redis_client, settings):
    settings.policy = "sliding_window"
    limiter = RedisLimiter(redis_client, settings)
    await limiter.initialize()
    for _ in range(10):
        await limiter.check("alice", now=100)
    assert not (await limiter.check("alice", now=100.999)).allowed
    assert (await limiter.check("alice", now=101)).allowed
    assert await redis_client.zcard(limiter.client_key("alice")) == 1
    assert 0 < await redis_client.pttl(limiter.client_key("alice")) <= 2000


async def test_sliding_decrease_retry_waits_for_enough_expiry(redis_client, settings):
    settings.policy = "sliding_window"
    limiter = RedisLimiter(redis_client, settings)
    await limiter.initialize()
    for index in range(10):
        await limiter.check("alice", now=100 + index * 0.01)
    await limiter.set_rate(4)
    denied = await limiter.check("alice", now=100.1)
    assert not denied.allowed
    assert denied.retry_after == pytest.approx(0.98)


async def test_expired_token_bucket_starts_full(redis_client, settings):
    limiter = RedisLimiter(redis_client, settings)
    await limiter.initialize()
    for _ in range(10):
        await limiter.check("alice", now=100)
    await redis_client.pexpire(limiter.client_key("alice"), 1)
    await asyncio.sleep(0.02)
    assert (await limiter.check("alice", now=102)).allowed


@pytest.mark.parametrize("policy", ["token_bucket", "sliding_window"])
async def test_global_denial_does_not_consume_client_budget(redis_client, settings, policy):
    settings.policy = policy
    settings.rate = 1
    limiter = RedisLimiter(redis_client, settings)
    await limiter.initialize()
    assert (await limiter.check("alice", now=100)).allowed
    denied = await limiter.check("bob", now=100)
    assert not denied.allowed
    assert denied.reason == "global"
    assert not await redis_client.exists(limiter.client_key("bob"))
    assert (await limiter.check("bob", now=101)).allowed
