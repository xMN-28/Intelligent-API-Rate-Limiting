"""Gymnasium environment. Episodes end by time truncation, not fictitious task success."""

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from rlrc.ml.features import ACTIONS, FeatureBuilder, Snapshot, apply_action
from rlrc.ml.simulator import SimConfig, Simulator


def reward_components(
    result: dict[str, Any], change: float, equal_clients: bool
) -> dict[str, float]:
    denom = max(
        1, result["offered"], result["success"] + result["backend_errors"] + result["timeouts"]
    )
    clients = np.asarray(result["per_client"], dtype=float)
    fairness = (
        float(clients.sum() ** 2 / (len(clients) * np.square(clients).sum()))
        if clients.any()
        else 1
    )
    return {
        "goodput": result["good"] / denom,
        "latency": -0.7 * result["late"] / denom,
        "failures": -2 * (result["backend_errors"] + result["timeouts"]) / denom,
        "rejections": -0.03 * result["rejected"] / max(1, result["offered"]),
        "change": -0.02 * abs(change),
        "fairness": -0.02 * (1 - fairness) if equal_clients and clients.sum() >= 6 else 0,
    }


class RateControlEnv(gym.Env[NDArray[np.float32], int]):
    metadata = {"render_modes": []}

    def __init__(
        self, config: SimConfig | None = None, backend: bool = False, predictor: Any = None
    ) -> None:
        self.config = config or SimConfig()
        self.backend = backend
        self.predictor = predictor
        self.action_space = gym.spaces.Discrete(len(ACTIONS))  # type: ignore[assignment]
        self.builder = self._builder()
        self.observation_space = gym.spaces.Box(-1, 4, (len(self.builder.names),), dtype=np.float32)
        self.sim: Simulator | None = None
        self.last = Snapshot(rate=self.config.initial_rate)

    def _builder(self) -> FeatureBuilder:
        c = self.config
        return FeatureBuilder(
            c.max_rate, c.target, c.inflight_limit, self.backend, self.predictor is not None
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.dropout_rng = np.random.default_rng(episode_seed + 700000)
        self.sim = Simulator(self.config, episode_seed)
        self.builder = self._builder()
        self.last = Snapshot(rate=self.config.initial_rate)
        return self.builder.build(self.last), {"seed": episode_seed, "family": self.sim.family}

    def step(self, action: int) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self.sim is None:
            raise RuntimeError("Call reset first")
        c = self.config
        rate = apply_action(self.sim.rate, action, c.min_rate, c.max_rate)
        change = rate / self.sim.rate - 1
        self.last, result = self.sim.advance(rate)
        if self.backend and self.dropout_rng.random() < 0.05:
            self.last.backend_available = self.last.busy_util = self.last.queue_util = 0
        components = reward_components(result, change, bool(np.ptp(self.sim.weights) < 1e-8))
        obs = self.builder.build(self.last)
        if self.predictor is not None:
            risk = float(self.predictor.predict_proba(obs[:-1].reshape(1, -1))[0, 1])
            obs[-1] = risk
        info = {"components": components, "metrics": result, "snapshot": self.last.as_dict()}
        return obs, float(sum(components.values())), False, self.sim.t >= c.duration, info
