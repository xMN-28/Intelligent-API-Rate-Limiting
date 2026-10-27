import csv
import json

import httpx
import pytest

from rlrc.loadgen import Phase, Scenario, load_scenario, run_workload
from rlrc.report import build_report, summarize


async def test_open_arrivals_count_missed_launches_and_reports(tmp_path, settings, monkeypatch):
    import asyncio

    original_client = httpx.AsyncClient

    async def slow(request):
        await asyncio.sleep(0.08)
        return httpx.Response(200, json={"ok": True}, headers={"X-Admission": "admitted"})

    def client(**kwargs):
        return original_client(transport=httpx.MockTransport(slow), **kwargs)

    monkeypatch.setattr("rlrc.loadgen.httpx.AsyncClient", client)
    scenario = Scenario(
        warmup=0,
        max_inflight=1,
        fairness_comparable=True,
        phases=[Phase(name="overload", duration=0.3, rate=100)],
    )
    run = tmp_path / "run"
    await run_workload(settings, scenario, "http://gateway", run)
    with (run / "requests.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert 29 <= len(rows) <= 31
    missed = [r for r in rows if r["outcome"] == "missed_launch"]
    assert len(missed) >= 20
    summary = summarize(run)
    assert summary["offered"] == summary["launched"] + len(missed)
    assert sum(summary["outcomes"].values()) == summary["offered"]
    assert not summary["valid_generator"]
    report = build_report(tmp_path)
    assert report.exists()
    assert (tmp_path / "comparison.png").exists()


async def test_warmup_excluded_and_same_seed_reproduces_clients(tmp_path, settings, monkeypatch):
    original_client = httpx.AsyncClient

    def client(**kwargs):
        return original_client(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, headers={"X-Admission": "admitted"})
            ),
            **kwargs,
        )

    monkeypatch.setattr("rlrc.loadgen.httpx.AsyncClient", client)
    scenario = Scenario(warmup=0.1, phases=[Phase(name="test", duration=0.1, rate=50)])
    sequences = []
    for index in range(2):
        run = tmp_path / str(index)
        await run_workload(settings, scenario, "http://gateway", run, seed=12)
        with (run / "requests.csv").open(newline="") as file:
            rows = sorted(csv.DictReader(file), key=lambda r: float(r["offset"]))
        sequences.append([r["client"] for r in rows])
        assert summarize(run)["offered"] < len(rows)
        assert json.loads((run / "manifest.json").read_text())["complete"]
    assert sequences[0] == sequences[1]


def test_bundled_scenarios_are_five_minutes_and_scale_explicitly():
    from pathlib import Path

    path = Path("configs/scenarios.yaml")
    for name in ("steady", "ramp", "burst", "overload", "capacity", "fairness", "aggressive"):
        scenario = load_scenario(path, name)
        assert scenario.duration == 300
        assert scenario.warmup == 30
        assert load_scenario(path, name, 0.1).duration == pytest.approx(30)
