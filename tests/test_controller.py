import pytest

from rlrc.config import ControllerConfig
from rlrc.controller import FeedbackController, Observation


def feed(controller, at, latency=0.02, count=40, failed=False, demand=100):
    for index in range(count):
        controller.observe(Observation(at + index * 0.00001, latency, failed))
    for _ in range(demand):
        controller.admit(at)


def test_overload_decreases_and_respects_floor():
    cfg = ControllerConfig(min_rate=10, alpha=1)
    controller = FeedbackController(cfg, 100, 0)
    for at in range(1, 30):
        feed(controller, at, latency=0.5)
        assert controller.evaluate(at + 0.1).reason == "backend_unhealthy"
    assert controller.rate == 10


def test_sparse_latency_does_not_adjust_but_timeout_does():
    controller = FeedbackController(ControllerConfig(), 100, 0)
    feed(controller, 1, latency=0.9, count=2)
    assert controller.evaluate(1.1).rate == 100
    controller.observe(Observation(2, 2, True, True))
    assert controller.evaluate(2.1).rate == 80
    # A timeout is consumed once rather than backoff on the same timeout forever.
    assert controller.evaluate(2.2).rate == 80


def test_healthy_growth_requires_demand_and_fresh_observations():
    controller = FeedbackController(ControllerConfig(max_rate=105), 100, 0)
    for at in range(1, 4):
        feed(controller, at)
        decision = controller.evaluate(at + 0.1)
    assert decision.rate == 105
    assert decision.reason == "healthy_increase"
    assert controller.evaluate(4).rate == 105


def test_low_demand_does_not_probe_upwards():
    controller = FeedbackController(ControllerConfig(), 100, 0)
    for at in range(1, 10):
        feed(controller, at, demand=2)
        assert controller.evaluate(at + 0.1).rate == 100


def test_ewma_smoothing_and_stale_window():
    controller = FeedbackController(ControllerConfig(alpha=0.2), 100, 0)
    feed(controller, 1, latency=0.1)
    controller.evaluate(1.1)
    feed(controller, 7, latency=0.5)
    decision = controller.evaluate(7.1)
    assert decision.latency_ewma == pytest.approx(0.18)
    assert decision.samples == 40
    assert controller.evaluate(20).samples == 0


def test_errors_trigger_backoff_below_latency_min_samples():
    controller = FeedbackController(ControllerConfig(), 100, 0)
    controller.observe(Observation(1, 0.01, failed=True))
    assert controller.evaluate(1.1).rate == 80


def test_recovery_and_bounds_with_alternating_load():
    controller = FeedbackController(ControllerConfig(alpha=1, max_rate=110), 100, 0)
    for at in range(1, 80):
        feed(controller, at, latency=0.3 if at < 5 else 0.02, demand=1000)
        decision = controller.evaluate(at + 0.1)
        assert 5 <= decision.rate <= 110
    assert controller.rate == 110
