"""Atomic global + client admission, using Redis server time in production."""

import hashlib
import math
import uuid
from dataclasses import dataclass

from redis.asyncio import Redis

from irl.config import Settings

TOKEN_BUCKET = """
local t = redis.call('TIME')
local now = tonumber(ARGV[1]) or (tonumber(t[1]) + tonumber(t[2])/1000000)
local rate = tonumber(redis.call('GET', KEYS[1]))
local n, burst = tonumber(ARGV[2]), tonumber(ARGV[3])
local rates = {rate, rate/n}
local tokens, caps = {}, {}
local retry, reason = 0, 'allowed'
for i=1,2 do
  caps[i] = math.max(1, rates[i]*burst)
  local state = redis.call('HMGET', KEYS[i+1], 'tokens', 'ts')
  tokens[i] = math.min(caps[i], (tonumber(state[1]) or caps[i]) +
      math.max(0, now-(tonumber(state[2]) or now))*rates[i])
  if tokens[i] < 1-0.000000001 then
    local wait = (1-tokens[i])/rates[i]
    if wait > retry then retry=wait; reason=(i==1 and 'global' or 'client') end
  end
end
if retry > 0 then return {0, tostring(retry), tostring(rate), reason} end
for i=1,2 do
  redis.call('HSET', KEYS[i+1], 'tokens', math.max(0,tokens[i]-1), 'ts', now)
  redis.call('PEXPIRE', KEYS[i+1], math.ceil(math.max(1, caps[i]/rates[i]*2)*1000))
end
return {1, '0', tostring(rate), 'allowed'}
"""

SLIDING_WINDOW = """
local t = redis.call('TIME')
local now = tonumber(ARGV[1]) or (tonumber(t[1]) + tonumber(t[2])/1000000)
local rate = tonumber(redis.call('GET', KEYS[1]))
local n, window = tonumber(ARGV[2]), tonumber(ARGV[3])
local limits = {math.max(1, math.floor(rate*window)),
                math.max(1, math.floor(rate*window/n))}
local retry, reason = 0, 'allowed'
for i=1,2 do
  redis.call('ZREMRANGEBYSCORE', KEYS[i+1], '-inf', now-window)
  local count = redis.call('ZCARD', KEYS[i+1])
  if count >= limits[i] then
    local oldest = redis.call('ZRANGE', KEYS[i+1], count-limits[i], count-limits[i], 'WITHSCORES')
    local wait = tonumber(oldest[2])+window-now
    if wait > retry then retry=wait; reason=(i==1 and 'global' or 'client') end
  end
end
if retry > 0 then return {0, tostring(retry), tostring(rate), reason} end
for i=1,2 do
  redis.call('ZADD', KEYS[i+1], now, ARGV[4])
  redis.call('PEXPIRE', KEYS[i+1], math.ceil(window*2000))
end
return {1, '0', tostring(rate), 'allowed'}
"""

# Settle every registered bucket at the OLD rate before publishing a new rate.
# This avoids retroactive refill or a fresh burst on controller updates.
UPDATE_RATE = """
local t = redis.call('TIME')
local now = tonumber(ARGV[1]) or (tonumber(t[1]) + tonumber(t[2])/1000000)
local old = tonumber(redis.call('GET', KEYS[1])) or tonumber(ARGV[2])
local new, n, burst = tonumber(ARGV[2]), tonumber(ARGV[3]), tonumber(ARGV[4])
for i=2,#KEYS do
  if redis.call('EXISTS', KEYS[i]) == 1 then
    local factor = (i==2 and 1 or n)
    local previous = old/factor
    local cap = math.max(1, new/factor*burst)
    local state = redis.call('HMGET', KEYS[i], 'tokens', 'ts')
    local tokens = math.min(cap, tonumber(state[1]) + math.max(0,now-tonumber(state[2]))*previous)
    redis.call('HSET', KEYS[i], 'tokens', tokens, 'ts', now)
    redis.call('PEXPIRE', KEYS[i], math.ceil(math.max(1,cap/(new/factor)*2)*1000))
  end
end
redis.call('SET', KEYS[1], new)
return tostring(new)
"""


@dataclass(frozen=True)
class Admission:
    allowed: bool
    retry_after: float
    rate: float
    reason: str

    @property
    def retry_header(self) -> str:
        return str(max(1, math.ceil(self.retry_after)))


class RedisLimiter:
    def __init__(self, redis: Redis, settings: Settings) -> None:
        self.redis = redis
        self.settings = settings
        self.prefix = f"irl:{{{settings.namespace}}}"
        self.policy_key = f"{self.prefix}:rate"
        self.global_key = f"{self.prefix}:{self.algorithm}:global"
        self._check = redis.register_script(
            SLIDING_WINDOW if self.algorithm == "sliding_window" else TOKEN_BUCKET
        )
        self._update = redis.register_script(UPDATE_RATE)

    @property
    def algorithm(self) -> str:
        return "sliding_window" if self.settings.policy == "sliding_window" else "token_bucket"

    def client_key(self, client: str) -> str:
        digest = hashlib.sha256(client.encode()).hexdigest()[:24]
        return f"{self.prefix}:{self.algorithm}:client:{digest}"

    async def initialize(self) -> None:
        await self.redis.set(self.policy_key, self.settings.rate, nx=True)

    async def check(self, client: str, *, now: float | None = None) -> Admission:
        result = await self._check(
            keys=[self.policy_key, self.global_key, self.client_key(client)],
            args=[
                "" if now is None else now,
                len(self.settings.clients),
                self.settings.sliding_window
                if self.algorithm == "sliding_window"
                else self.settings.burst_seconds,
                uuid.uuid4().hex,
            ],
        )
        return Admission(bool(int(result[0])), float(result[1]), float(result[2]), str(result[3]))

    async def set_rate(self, rate: float, *, now: float | None = None) -> None:
        if self.algorithm == "token_bucket":
            await self._update(
                keys=[
                    self.policy_key,
                    self.global_key,
                    *(self.client_key(c) for c in self.settings.clients),
                ],
                args=[
                    "" if now is None else now,
                    rate,
                    len(self.settings.clients),
                    self.settings.burst_seconds,
                ],
            )
        else:
            await self.redis.set(self.policy_key, rate)

    async def get_rate(self) -> float:
        value = await self.redis.get(self.policy_key)
        return float(value) if value is not None else self.settings.rate
