import logging
import time

from app.redis_client import get_redis

log = logging.getLogger(__name__)

_TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])
local d = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(d[1])
local ts = tonumber(d[2])
if tokens == nil then tokens = capacity; ts = now end
tokens = math.min(capacity, tokens + math.max(0, now - ts) * rate)
local allowed = 0
if tokens >= cost then tokens = tokens - cost; allowed = 1 end
redis.call('HSET', key, 'tokens', tostring(tokens), 'ts', tostring(now))
redis.call('EXPIRE', key, math.ceil(capacity / rate) * 2)
return {allowed, tostring(tokens)}
"""


def take_token(bucket: str, capacity: int, per_second: float, cost: int = 1) -> tuple[bool, float]:
    """Token bucket in Redis. Returns (allowed, retry_after_seconds). Fails open if Redis is down."""
    try:
        allowed, tokens = get_redis().eval(
            _TOKEN_BUCKET_LUA, 1, f"rl:{bucket}", capacity, per_second, time.time(), cost
        )
    except Exception:
        log.warning("rate limiter unavailable, failing open", exc_info=True)
        return True, 0.0
    if int(allowed) == 1:
        return True, 0.0
    return False, max(0.0, (cost - float(tokens)) / per_second)
