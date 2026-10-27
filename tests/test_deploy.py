import json
from pathlib import Path

import yaml


def test_dashboard_panels_reference_provisioned_datasource():
    dashboard = json.loads(Path("deploy/grafana/dashboards/gateway.json").read_text())
    datasource = yaml.safe_load(
        Path("deploy/grafana/provisioning/datasources/prometheus.yaml").read_text()
    )
    assert 8 <= len(dashboard["uid"]) <= 40
    assert f"/d/{dashboard['uid']}" in Path("README.md").read_text()
    assert {"Global admission budget", "Controller inference p95", "Safety interventions"} <= {
        panel["title"] for panel in dashboard["panels"]
    }
    assert all(
        panel["datasource"]["uid"] == datasource["datasources"][0]["uid"]
        for panel in dashboard["panels"]
    )


def test_compose_publishes_only_localhost_and_has_redis_healthcheck():
    compose = yaml.safe_load(Path("compose.yaml").read_text())
    for service in compose["services"].values():
        for port in service.get("ports", []):
            assert port.startswith("127.0.0.1:")
    assert compose["services"]["redis"]["healthcheck"]["test"] == ["CMD", "redis-cli", "ping"]
