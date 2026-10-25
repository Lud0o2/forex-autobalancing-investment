"""Standardized market observation schemas for Forex V1."""

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from math import isfinite

SUPPORTED_CURRENCIES = frozenset({"EUR", "USD", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"})


class PriceType(str, Enum):
    """Supported observation types; reference prices are not execution quotes."""

    REFERENCE = "reference"
    CLOSE = "close"
    BID = "bid"
    ASK = "ask"


@dataclass(frozen=True, slots=True)
class FXObservation:
    """One observed rate, quoted as units of quote currency per base currency."""

    market_date: date
    base_currency: str
    quote_currency: str
    quote_per_base: float
    source: str
    price_type: PriceType
    available_at: datetime | None = None
    availability_known: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.market_date, date) or isinstance(self.market_date, datetime):
            raise TypeError("market_date must be a calendar date, not a timestamp")

        for field_name, currency in (
            ("base_currency", self.base_currency),
            ("quote_currency", self.quote_currency),
        ):
            if not isinstance(currency, str) or currency not in SUPPORTED_CURRENCIES:
                raise ValueError(
                    f"{field_name} must be one of the supported ISO currency codes: "
                    f"{', '.join(sorted(SUPPORTED_CURRENCIES))}"
                )

        if self.base_currency == self.quote_currency:
            raise ValueError("base_currency and quote_currency must differ")

        if isinstance(self.quote_per_base, bool) or not isinstance(
            self.quote_per_base, (int, float)
        ):
            raise TypeError("quote_per_base must be a finite positive number")
        try:
            quote_per_base = float(self.quote_per_base)
        except OverflowError as error:
            raise ValueError("quote_per_base must be finite and strictly positive") from error
        if not isfinite(quote_per_base) or quote_per_base <= 0:
            raise ValueError("quote_per_base must be finite and strictly positive")
        object.__setattr__(self, "quote_per_base", quote_per_base)

        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string")

        if not isinstance(self.price_type, PriceType):
            try:
                object.__setattr__(self, "price_type", PriceType(self.price_type))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Unsupported price_type: {self.price_type!r}") from error

        if not isinstance(self.availability_known, bool):
            raise TypeError("availability_known must be a boolean")
        if self.availability_known:
            if not isinstance(self.available_at, datetime):
                raise TypeError("available_at must be a datetime when availability is known")
            if self.available_at.tzinfo is None or self.available_at.utcoffset() is None:
                raise ValueError("available_at must include an explicit timezone")
        elif self.available_at is not None:
            raise ValueError("available_at must be None when availability_known is false")
