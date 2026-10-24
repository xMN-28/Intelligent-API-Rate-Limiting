import pytest
from pydantic import ValidationError

from irl.config import ControllerConfig, Settings, load_settings


def test_controller_floor_cannot_make_sample_threshold_unreachable():
    with pytest.raises(ValidationError, match="headroom"):
        ControllerConfig(min_rate=5)


def test_bad_configuration_is_rejected():
    with pytest.raises(ValidationError):
        Settings(clients={"alice": "same", "bob": "same"})
    with pytest.raises(ValidationError):
        Settings(rate=1)
    with pytest.raises(ValidationError):
        Settings(typo="silently ignored")


def test_explicit_missing_config_does_not_silently_use_defaults(monkeypatch):
    monkeypatch.setenv("IRL_CONFIG", "does-not-exist.yaml")
    with pytest.raises(FileNotFoundError):
        load_settings()
