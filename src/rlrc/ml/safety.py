"""An explicit, separately evaluated layer; raw PPO is always bounded, never exploratory."""

import math

from rlrc.config import ControllerConfig
from rlrc.ml.features import Snapshot


def safety_filter(
    s: Snapshot, proposal: float, config: ControllerConfig, enabled: bool = True
) -> tuple[float, str]:
    if not math.isfinite(proposal) or proposal <= 0:
        raise ValueError("Invalid proposed budget")
    bounded = min(config.max_rate, max(config.min_rate, proposal))
    reason = "bounds" if bounded != proposal else "model"
    # Raw PPO still has min/max and the shared fixed in-flight ceiling.
    if not enabled:
        return bounded, reason
    if s.timeouts > 0:
        return max(config.min_rate, min(bounded, s.rate * 0.8)), "timeout_backoff"
    if s.samples < 5 and bounded > s.rate:
        return min(config.max_rate, max(config.min_rate, s.rate)), "insufficient_samples"
    if s.errors > config.error_threshold or (s.samples >= 5 and s.p90 > config.latency_target):
        return max(config.min_rate, min(bounded, s.rate * 0.9)), "unhealthy_backoff"
    limited = min(s.rate * 1.1, max(s.rate * 0.8, bounded))
    if limited != bounded:
        reason = "jump_limit"
    return min(config.max_rate, max(config.min_rate, limited)), reason
