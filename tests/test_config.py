from datetime import date
from pathlib import Path

import pytest
import yaml

from forex_v1.config import ConfigError, load_config

CONFIG_PATH = "configs/v1.yaml"


def test_project_configuration_loads():
    config = load_config(CONFIG_PATH)

    assert config.reference_currency == "EUR"
    assert config.currencies == ("EUR", "USD", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD")
    assert config.data_frequency == "D1"
    assert config.rebalance_frequency == "weekly"
    assert config.long_only is True
    assert config.leverage_enabled is False
    assert config.initial_capital_eur == 10000
    assert config.development_fraction == 0.75
    assert config.test_fraction == 0.25
    assert config.transaction_cost_bps is None
    assert config.start_date is None
    assert config.end_date is None


def test_config_accepts_custom_date_range_and_explicit_cost(tmp_path):
    config_data = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    config_data["data"]["start_date"] = date(2020, 1, 1)
    config_data["data"]["end_date"] = date(2025, 12, 31)
    config_data["data"]["transaction_cost_bps"] = 0
    custom_config = tmp_path / "custom.yaml"
    custom_config.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    config = load_config(custom_config)

    assert config.start_date == date(2020, 1, 1)
    assert config.end_date == date(2025, 12, 31)
    assert config.transaction_cost_bps == 0


@pytest.mark.parametrize(
    ("section", "key", "invalid_value"),
    [
        ("portfolio", "initial_capital_eur", 0),
        ("portfolio", "leverage_enabled", True),
        ("project", "currencies", ["EUR", "USD", "USD", "JPY", "CHF", "CAD", "AUD", "NZD"]),
        ("data", "historical_sources", ["fred_h10", "fred_h10"]),
        ("split", "test_fraction", 0.3),
        ("data", "transaction_cost_bps", -1),
        ("data", "start_date", "2025-03-01"),
        ("data", "end_date", "2025-01-01"),
    ],
)
def test_inconsistent_configuration_is_rejected(tmp_path, section, key, invalid_value):
    config = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    config[section][key] = invalid_value
    if key in {"start_date", "end_date"}:
        config["data"]["start_date"] = "2025-03-01"
        config["data"]["end_date"] = "2025-01-01"
    invalid_config = tmp_path / "invalid.yaml"
    invalid_config.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(invalid_config)


def test_config_date_values_are_optional():
    config = load_config(CONFIG_PATH)

    assert config.start_date is None
    assert config.end_date is None
