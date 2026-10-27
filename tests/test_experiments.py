import json
import socket

import pytest

from rlrc.experiments import run_trial
from rlrc.loadgen import Phase, Scenario
from rlrc.report import summarize


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.mark.redis
async def test_real_network_trial_records_and_cleans_up(redis_client, settings, tmp_path):
    scenario = Scenario(warmup=0.2, phases=[Phase(name="steady", duration=0.5, rate=10)])
    gateway_port, backend_port = free_port(), free_port()
    output = tmp_path / "network"
    await run_trial(
        settings,
        scenario,
        output,
        seed=12,
        scenario_name="network",
        gateway_port=gateway_port,
        backend_port=backend_port,
    )
    summary = summarize(output)
    assert summary["complete"]
    assert summary["outcomes"].get("success", 0) >= 4
    assert sum(summary["outcomes"].values()) == summary["offered"]
    assert not (output / "service-config.json").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    assert "admin_token" not in manifest["settings"]
    for port in (gateway_port, backend_port):
        with socket.socket() as probe:
            probe.settimeout(1)
            assert probe.connect_ex(("127.0.0.1", port)) != 0


@pytest.mark.redis
async def test_real_upstream_read_timeout_becomes_504(redis_client, settings, tmp_path):
    settings.upstream_timeout = 0.05
    settings.service_time = 0.2
    settings.backend_workers = 1
    scenario = Scenario(warmup=0, phases=[Phase(name="timeout", duration=0.3, rate=10)])
    output = tmp_path / "timeout"
    await run_trial(
        settings,
        scenario,
        output,
        seed=12,
        scenario_name="timeout",
        gateway_port=free_port(),
        backend_port=free_port(),
    )
    summary = summarize(output)
    assert summary["outcomes"].get("upstream_timeout", 0) >= 1
    assert summary["admitted"] == summary["launched"]
