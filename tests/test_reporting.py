import csv
import json

import pytest

from irl.loadgen import COLUMNS
from irl.report import summarize


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
