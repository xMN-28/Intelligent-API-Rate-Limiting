import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from rlrc.config import ControllerConfig, MLConfig, Settings
from rlrc.ml.environment import RateControlEnv, reward_components
from rlrc.ml.features import FeatureBuilder, Snapshot, apply_action
from rlrc.ml.runtime import LearnedController, NumpyActor
from rlrc.ml.safety import safety_filter
from rlrc.ml.simulator import SimConfig, Simulator
from rlrc.ml.training import train


def test_environment_checker_reset_and_determinism():
    check_env(RateControlEnv())
    left, right = RateControlEnv(), RateControlEnv()
    assert np.array_equal(left.reset(seed=81)[0], right.reset(seed=81)[0])
    for action in [0, 5, 3, 1, 4] * 5:
        a, b = left.step(action), right.step(action)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:] == b[1:]


def test_environment_time_truncation_and_reset():
    env = RateControlEnv(SimConfig(duration=2))
    with pytest.raises(RuntimeError):
        env.step(3)
    env.reset(seed=1)
    assert env.step(3)[2:4] == (False, False)
    assert env.step(3)[2:4] == (False, True)
    with pytest.raises(RuntimeError):
        env.step(3)
    assert env.reset(seed=2)[0].shape == (16,)


def test_fixed_normalization_and_optional_telemetry():
    builder = FeatureBuilder(100, 0.2, 50)
    obs = builder.build(Snapshot(rate=50, offered=20, p90=0.2, samples=10))
    assert obs.dtype == np.float32
    assert obs[0] == pytest.approx(0.5)
    assert obs[1] == pytest.approx(0.2)
    assert obs[6] == pytest.approx(0.25)
    assert len(FeatureBuilder(100, 0.2, 50, backend=True).build(Snapshot())) == 19
    with pytest.raises(ValueError):
        builder.build(Snapshot(p90=float("nan")))
    assert np.max(builder.build(Snapshot(offered=1e9))) <= 4


@pytest.mark.parametrize(
    "action,expected", [(0, 80), (1, 90), (2, 95), (3, 100), (4, 105), (5, 110)]
)
def test_discrete_actions(action, expected):
    assert apply_action(100, action, 10, 120) == pytest.approx(expected)
    assert 10 <= apply_action(120, action, 10, 120) <= 120


@pytest.mark.parametrize("action", [-1, 6, 1.5, float("nan")])
def test_invalid_actions(action):
    with pytest.raises(ValueError):
        apply_action(100, action, 10, 120)


def test_reward_does_not_favor_reject_all_or_overload():
    def reward(good, late=0, fail=0, rejected=0):
        row = dict(
            offered=100,
            success=good + late,
            good=good,
            late=late,
            backend_errors=fail,
            timeouts=0,
            rejected=rejected,
            per_client=[good / 3] * 3,
        )
        return sum(reward_components(row, 0, True).values())

    assert reward(30, rejected=70) > reward(0, rejected=100)
    assert reward(30, rejected=70) > reward(30, fail=70)
    assert reward(30, rejected=70) > reward(30, late=70)
    empty = dict(
        offered=0,
        success=0,
        good=0,
        late=0,
        backend_errors=0,
        timeouts=0,
        rejected=0,
        per_client=[0, 0, 0],
    )
    assert sum(reward_components(empty, 0, True).values()) == 0


def test_safety_timeout_sparse_and_unhealthy():
    config = ControllerConfig(max_rate=120)
    s = Snapshot(rate=100, samples=20, timeouts=0.05)
    assert safety_filter(s, 110, config) == (80, "timeout_backoff")
    assert safety_filter(Snapshot(rate=100), 110, config) == (100, "insufficient_samples")
    assert safety_filter(Snapshot(rate=100, samples=20, errors=0.2), 110, config) == (
        90,
        "unhealthy_backoff",
    )
    assert safety_filter(Snapshot(rate=100), 110, config, False)[0] == 110
    with pytest.raises(ValueError):
        safety_filter(s, float("nan"), config)


def test_simulator_queue_limits_and_unlimited_overload():
    sim = Simulator(
        SimConfig(
            duration=20,
            family="fixed",
            fixed_offered=100,
            workers=1,
            service_time=0.1,
            queue=3,
            record_responses=True,
        ),
        5,
    )
    for _ in range(20):
        _, row = sim.advance(100, "none")
        assert row["queue"] <= 3
        assert row["inflight"] <= sim.cfg.inflight_limit
    assert sum(r["backend_errors"] for r in sim.history) > 0
    assert sum(r["success"] for r in sim.history) < 250


def test_simulator_timeout_does_not_cancel_backend_work():
    sim = Simulator(
        SimConfig(
            duration=3,
            family="fixed",
            fixed_offered=10,
            timeout=0.05,
            workers=1,
            service_time=0.2,
            queue=2,
        ),
        17,
    )
    _, row = sim.advance(100, "none")
    assert row["timeouts"] > 0
    assert sim.busy > 0


def test_missing_model_fallback_is_explicit(tmp_path):
    cfg = Settings(policy="rl", ml=MLConfig(model_path=str(tmp_path / "missing")))
    controller = LearnedController(cfg)
    result = controller.decide(Snapshot(rate=100), 80)
    assert result["rate"] == 80
    assert result["fallback_reason"]
    assert result["safety"] == "fallback"
    invalid = controller.decide(Snapshot(p90=float("nan")), float("nan"))
    assert np.isfinite(invalid["rate"]) and invalid["fallback_input_invalid"]
    cfg.controller.interval = 0.5
    assert "one-second" in LearnedController(cfg).load_error


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    directory = tmp_path_factory.mktemp("rl-model") / "trained"
    path = train(
        directory, steps=256, count=1, config=SimConfig(duration=40), validation_interval=1024
    )
    return path


def test_checkpoint_export_parity_and_checksum(tiny_model, tmp_path):
    cfg = Settings.model_validate(
        {
            "policy": "rl",
            "rate": 20,
            "inflight_limit": 64,
            "controller": {"max_rate": 120, "latency_target": 0.3},
            "ml": {"model_path": str(tiny_model.parent)},
        }
    )
    actor = NumpyActor(tiny_model.parent, cfg)
    ppo = PPO.load(tiny_model, device="cpu")
    rng = np.random.default_rng(4)
    for _ in range(100):
        obs = rng.uniform(-1, 4, size=16).astype(np.float32)
        assert actor.action(obs) == int(ppo.predict(obs, deterministic=True)[0])
    with pytest.raises(ValueError):
        actor.action(np.full(16, np.nan, dtype=np.float32))
    wrong_cfg = cfg.model_copy(update={"inflight_limit": 65})
    with pytest.raises(ValueError, match="normalization"):
        NumpyActor(tiny_model.parent, wrong_cfg)


def test_resume_checkpoint_and_metadata(tiny_model, tmp_path):
    resumed = train(
        tmp_path / "resumed",
        steps=256,
        count=1,
        resume=tiny_model,
        config=SimConfig(duration=40),
        validation_interval=1024,
    )
    metadata = json.loads((resumed.parent / "metadata.json").read_text())
    assert metadata["steps_completed"] == 512
    assert metadata["complete"]


@pytest.mark.redis
async def test_gateway_inference_failure_recovery_and_publication(
    redis_client, settings, tiny_model
):
    import asyncio

    import httpx

    from rlrc.gateway import create_app

    cfg = Settings.model_validate(
        {
            "redis_url": settings.redis_url,
            "namespace": settings.namespace,
            "policy": "rl_safe",
            "rate": 20,
            "inflight_limit": 64,
            "burst_seconds": 1,
            "decision_log": settings.decision_log,
            "controller": {"max_rate": 120, "latency_target": 0.3},
            "ml": {"model_path": str(tiny_model.parent)},
        }
    )
    app = create_app(cfg)
    app.state.upstream_transport = httpx.MockTransport(lambda r: httpx.Response(200))
    async with app.router.lifespan_context(app):
        assert app.state.learned.load_error is None
        actor = app.state.learned.actor
        original = actor.action
        actor.action = lambda observation: 99
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as http:
            for _ in range(8):
                await http.get("/proxy/work", headers={"X-API-Key": "demo-alice"})
            await asyncio.sleep(1.15)
            actor.action = original
            for _ in range(8):
                await http.get("/proxy/work", headers={"X-API-Key": "demo-bob"})
            await asyncio.sleep(1.1)
            assert (await http.get("/readyz")).json()["model_ready"]
            assert 10 <= await app.state.limiter.get_rate() <= 120
    records = [json.loads(line) for line in Path(cfg.decision_log).read_text().splitlines()]
    assert records[0]["fallback_reason"]
    assert records[-1]["fallback_reason"] is None
    assert records[-1]["applied"]
    assert "proposed_rate" in records[-1]


def test_actor_checksum_and_nan_output_fallback(tiny_model, tmp_path):
    import shutil

    cfg = Settings.model_validate(
        {
            "policy": "rl",
            "rate": 20,
            "inflight_limit": 64,
            "controller": {"max_rate": 120, "latency_target": 0.3},
            "ml": {"model_path": str(tiny_model.parent)},
        }
    )
    runtime = LearnedController(cfg)
    assert runtime.load_error is None
    runtime.actor.action = lambda obs: float("nan")
    assert runtime.decide(Snapshot(rate=20), 18)["fallback_reason"]
    directory = tmp_path / "bad"
    directory.mkdir()
    shutil.copy2(tiny_model.parent / "metadata.json", directory / "metadata.json")
    (directory / "policy.npz").write_bytes(b"corrupt artifact")
    with pytest.raises(ValueError, match="checksum"):
        NumpyActor(directory, cfg)


def test_simulator_capacity_restoration_and_no_demand():
    config = SimConfig(duration=10, family="fixed", fixed_offered=0)
    sim = Simulator(config, 71)
    for _ in range(10):
        snapshot, result = sim.advance(20)
        assert result["offered"] == result["success"] == 0
        assert snapshot.inflight == 0
    for kwargs in (
        {"duration": 0},
        {"queue": -1},
        {"service_time": 0},
        {"fixed_offered": float("nan")},
    ):
        with pytest.raises(ValueError):
            SimConfig(**kwargs)


def test_safety_bounds_even_for_stale_persisted_rate():
    from rlrc.config import ControllerConfig

    cfg = ControllerConfig(min_rate=10, max_rate=120)
    assert safety_filter(Snapshot(rate=5), 15, cfg)[0] == 10


def test_predictor_normalization_and_checksum(tmp_path):
    import shutil

    from rlrc.ml.runtime import load_predictor

    source = Path("assets/models/overload-predictor")
    directory = tmp_path / "predictor"
    shutil.copytree(source, directory)
    config = SimConfig().as_dict()
    features = FeatureBuilder(120, 0.3, 64).names
    predictor = load_predictor(directory / "predictor.joblib", config, features)
    assert np.isfinite(predictor.predict_proba(np.zeros((1, 16)))).all()
    with pytest.raises(ValueError, match="normalization"):
        load_predictor(directory / "predictor.joblib", config | {"target": 0.2}, features)
    (directory / "predictor.joblib").write_bytes(b"invalid")
    with pytest.raises(ValueError, match="checksum"):
        load_predictor(directory / "predictor.joblib", config, features)


def test_report_intervals_and_generator_isolation():
    from rlrc.ml.benchmarks import controller_config
    from rlrc.ml.reporting import interval

    assert interval([3])[1] is None
    mean, error = interval([2, 3, 4])
    assert mean == 3 and error > 0
    original = Settings()
    variant = controller_config(original, "rl_safe", 24)
    assert original.policy == "adaptive" and original.rate == 100
    assert variant.policy == "rl_safe" and variant.rate == 24


@pytest.mark.redis
async def test_rl_publication_failure_rolls_back(redis_client, settings, tiny_model):
    import asyncio
    from unittest.mock import AsyncMock

    from redis.exceptions import ConnectionError

    from rlrc.gateway import create_app

    cfg = Settings.model_validate(
        {
            "redis_url": settings.redis_url,
            "namespace": settings.namespace,
            "policy": "rl",
            "rate": 20,
            "inflight_limit": 64,
            "decision_log": settings.decision_log,
            "controller": {"max_rate": 120, "latency_target": 0.3},
            "ml": {"model_path": str(tiny_model.parent)},
        }
    )
    app = create_app(cfg)
    async with app.router.lifespan_context(app):
        app.state.learned.actor.action = lambda obs: 5
        app.state.limiter.set_rate = AsyncMock(side_effect=ConnectionError("test unavailable"))
        await asyncio.sleep(1.2)
        assert app.state.controller.rate == 20
        assert await app.state.limiter.get_rate() == 20
    records = [json.loads(line) for line in Path(cfg.decision_log).read_text().splitlines()]
    assert records[0]["proposed_rate"] == 22
    assert records[0]["applied"] is False
    assert records[0]["error"] == "ConnectionError"
