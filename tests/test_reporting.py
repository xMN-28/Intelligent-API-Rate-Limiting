import csv
import json

import pytest

from rlrc.loadgen import COLUMNS
from rlrc.report import summarize


def test_fast_rejections_do_not_improve_success_percentiles(tmp_path, settings):
    manifest = {
        "settings": settings.model_dump(exclude={"clients", "admin_token"}),
        "clients": ["alice", "bob"],
        "complete": True,
        "scenario_name": "fairness",
        "scenario": {"phases": [{"duration": 10}], "fairness_comparable": True},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    rows = []
    for index in range(100):
        row = dict.fromkeys(COLUMNS, "")
        row.update(
            offset=index / 10,
            client="alice" if index < 50 else "bob",
            launched=True,
            status=200 if index < 10 else 429,
            outcome="success" if index < 10 else "rejected",
            admitted=index < 10,
            latency=0.5 if index < 10 else 0.001,
            launch_lag=0,
        )
        rows.append(row)
    with (tmp_path / "requests.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(tmp_path)
    assert summary["latency_p95"] == pytest.approx(0.5)
    assert summary["rejection_p95"] == pytest.approx(0.001)
    assert summary["goodput_rps"] == 0
    assert summary["jain_fairness"] == pytest.approx(0.5)
    assert summary["admitted"] == 10


def test_controller_summary_distinguishes_guards_from_overrides(tmp_path, settings):
    manifest = {
        "settings": settings.model_dump(exclude={"clients", "admin_token"}),
        "clients": ["alice"],
        "complete": True,
        "scenario_name": "control",
        "measured_start_wall": 100,
        "scenario": {"phases": [{"duration": 10}], "fairness_comparable": True},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    row = dict.fromkeys(COLUMNS, "")
    row.update(
        offset=0,
        client="alice",
        launched=True,
        status=200,
        outcome="success",
        admitted=True,
        latency=0.1,
        launch_lag=0,
    )
    with (tmp_path / "requests.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    decisions = [
        {
            "wall_time": 101,
            "rate": 18,
            "proposed_rate": 22,
            "safety": "unhealthy_backoff",
            "applied": True,
        },
        {
            "wall_time": 102,
            "rate": 18,
            "proposed_rate": 18,
            "safety": "unhealthy_backoff",
            "applied": True,
        },
        {
            "wall_time": 103,
            "rate": 18,
            "proposed_rate": 18,
            "safety": "fallback",
            "fallback_reason": "invalid model",
            "applied": True,
        },
    ]
    (tmp_path / "decisions.jsonl").write_text("\n".join(json.dumps(d) for d in decisions))
    summary = summarize(tmp_path)
    assert summary["safety_interventions"]["unhealthy_backoff"] == 2
    assert summary["safety_overrides"] == {"unhealthy_backoff": 1}
    assert summary["fallback_count"] == 1
    assert not summary["valid_controller"]
