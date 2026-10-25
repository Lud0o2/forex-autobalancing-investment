"""Loading and validation for the project-level YAML configuration."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from forex_v1.data.schemas import SUPPORTED_CURRENCIES


class ConfigError(ValueError):
    """Raised when a project configuration is missing or inconsistent."""


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    reference_currency: str
    currencies: tuple[str, ...]
    data_frequency: str
    rebalance_frequency: str
    long_only: bool
    leverage_enabled: bool
    initial_capital_eur: float
    development_fraction: float
    test_fraction: float
    historical_sources: tuple[str, ...]
    start_date: date | None
    end_date: date | None
    transaction_cost_bps: float | None


def _section(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = document.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError(f"Configuration section {name!r} must be a mapping")
    return value


def _as_date(value: Any, field_name: str) -> date | None:
    if value is None:
        return None
    if not isinstance(value, date) or isinstance(value, datetime):
        raise ConfigError(f"{field_name} must be a YAML date or null")
    return value


def _positive_finite(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field_name} must be a finite number greater than zero")
    try:
        result = float(value)
    except OverflowError as error:
        raise ConfigError(f"{field_name} must be a finite number greater than zero") from error
    if not isfinite(result) or result <= 0:
        raise ConfigError(f"{field_name} must be a finite number greater than zero")
    return result


def load_config(path: str | Path) -> ProjectConfig:
    """Load a YAML configuration and validate its V1 invariants."""
    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as config_file:
            document = yaml.safe_load(config_file)
    except OSError as error:
        raise ConfigError(f"Could not read configuration {config_path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML in configuration {config_path}: {error}") from error

    if not isinstance(document, Mapping):
        raise ConfigError("Configuration root must be a mapping")

    project = _section(document, "project")
    portfolio = _section(document, "portfolio")
    data = _section(document, "data")
    split = _section(document, "split")

    reference_currency = project.get("reference_currency")
    currencies = project.get("currencies")
    if not isinstance(reference_currency, str) or reference_currency not in SUPPORTED_CURRENCIES:
        raise ConfigError("project.reference_currency must be a supported ISO currency code")
    if not isinstance(currencies, list) or not currencies:
        raise ConfigError("project.currencies must be a non-empty list")
    if any(
        not isinstance(currency, str) or currency not in SUPPORTED_CURRENCIES
        for currency in currencies
    ):
        raise ConfigError("project.currencies contains an unsupported currency code")
    if len(currencies) != len(set(currencies)):
        raise ConfigError("project.currencies must not contain duplicates")
    if reference_currency not in currencies:
        raise ConfigError("project.currencies must include project.reference_currency")

    data_frequency = project.get("data_frequency")
    rebalance_frequency = project.get("rebalance_frequency")
    if data_frequency != "D1":
        raise ConfigError("project.data_frequency must be 'D1' in V1")
    if rebalance_frequency != "weekly":
        raise ConfigError("project.rebalance_frequency must be 'weekly' in V1")

    long_only = portfolio.get("long_only")
    leverage_enabled = portfolio.get("leverage_enabled")
    if long_only is not True or leverage_enabled is not False:
        raise ConfigError("V1 requires long_only=true and leverage_enabled=false")
    initial_capital = _positive_finite(
        portfolio.get("initial_capital_eur"), "portfolio.initial_capital_eur"
    )

    development_fraction = split.get("development_fraction")
    test_fraction = split.get("test_fraction")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (development_fraction, test_fraction)
    ):
        raise ConfigError("split fractions must be finite numbers")
    try:
        development_fraction = float(development_fraction)
        test_fraction = float(test_fraction)
    except OverflowError as error:
        raise ConfigError("split fractions must be finite numbers") from error
    if (
        not isfinite(development_fraction)
        or not isfinite(test_fraction)
        or development_fraction <= 0
        or test_fraction <= 0
        or abs(development_fraction + test_fraction - 1.0) > 1e-9
    ):
        raise ConfigError("split fractions must be positive and sum to 1")

    historical_sources = data.get("historical_sources")
    if (
        not isinstance(historical_sources, list)
        or not historical_sources
        or any(not isinstance(source, str) or not source.strip() for source in historical_sources)
    ):
        raise ConfigError("data.historical_sources must be a non-empty list of source names")
    if len(historical_sources) != len(set(historical_sources)):
        raise ConfigError("data.historical_sources must not contain duplicates")

    start_date = _as_date(data.get("start_date"), "data.start_date")
    end_date = _as_date(data.get("end_date"), "data.end_date")
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ConfigError("data.start_date must not be after data.end_date")

    transaction_cost = data.get("transaction_cost_bps")
    if transaction_cost is not None:
        if isinstance(transaction_cost, bool) or not isinstance(transaction_cost, (int, float)):
            raise ConfigError("data.transaction_cost_bps must be a finite non-negative number or null")
        try:
            transaction_cost = float(transaction_cost)
        except OverflowError as error:
            raise ConfigError(
                "data.transaction_cost_bps must be a finite non-negative number or null"
            ) from error
        if not isfinite(transaction_cost) or transaction_cost < 0:
            raise ConfigError("data.transaction_cost_bps must be a finite non-negative number or null")

    return ProjectConfig(
        reference_currency=reference_currency,
        currencies=tuple(currencies),
        data_frequency=data_frequency,
        rebalance_frequency=rebalance_frequency,
        long_only=long_only,
        leverage_enabled=leverage_enabled,
        initial_capital_eur=initial_capital,
        development_fraction=development_fraction,
        test_fraction=test_fraction,
        historical_sources=tuple(historical_sources),
        start_date=start_date,
        end_date=end_date,
        transaction_cost_bps=transaction_cost,
    )
