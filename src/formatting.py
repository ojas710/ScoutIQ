"""Currency conversion and formatting helpers (values stored in EUR)."""
from __future__ import annotations

import math


def to_millions(value_eur: float | None) -> float | None:
    """Convert EUR to millions of EUR. Returns None for missing input."""
    if value_eur is None or (isinstance(value_eur, float) and math.isnan(value_eur)):
        return None
    return float(value_eur) / 1_000_000.0


def format_eur_m(value_eur: float | None, decimals: int = 1) -> str:
    """Format EUR as e.g. '€12.5m' (sub-million values as '€750k'). Missing -> 'n/a'."""
    m = to_millions(value_eur)
    if m is None:
        return "n/a"
    sign = "-" if m < 0 else ""
    m = abs(m)
    if m < 1:
        return f"{sign}€{m * 1000:.0f}k"
    return f"{sign}€{m:.{decimals}f}m"
