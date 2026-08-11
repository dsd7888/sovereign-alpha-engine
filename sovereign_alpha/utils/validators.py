"""
Input validation at system boundaries (external data, user-supplied weights).

Internal call sites that already guarantee well-formed data should NOT call
these — they exist for the edges where untrusted or unreliable data enters
the system: yfinance responses, Screener.in scrapes, optimiser output.
"""
from __future__ import annotations

import math


def clean_ticker_list(tickers: list[str]) -> list[str]:
    """Strip whitespace, drop empties, de-duplicate while preserving order."""
    seen: set[str] = set()
    cleaned: list[str] = []
    for t in tickers:
        t = t.strip().upper()
        if not t or t in seen:
            continue
        seen.add(t)
        cleaned.append(t)
    return cleaned


def is_valid_price(price: float | None) -> bool:
    """A tradeable price must be a finite, strictly positive number."""
    return price is not None and math.isfinite(price) and price > 0


def normalise_weights(weights: dict[str, float], tol: float = 1e-6) -> dict[str, float]:
    """
    Return ``weights`` rescaled to sum to exactly 1.0.

    Portfolio weights arriving from Monte Carlo draws or manual overrides can
    drift slightly from 1.0 due to floating point accumulation; silently
    trading against an un-normalised weight vector would misallocate capital.
    Raises ``ValueError`` on empty input, negative weights, or an all-zero
    vector — those indicate an upstream bug, not something to paper over.
    """
    if not weights:
        raise ValueError("normalise_weights: empty weights dict")
    if any(w < -tol for w in weights.values()):
        raise ValueError(f"normalise_weights: negative weight in {weights}")

    total = sum(weights.values())
    if total <= tol:
        raise ValueError(f"normalise_weights: weights sum to ~0 ({total})")

    if abs(total - 1.0) <= tol:
        return dict(weights)
    return {k: v / total for k, v in weights.items()}


def is_finite_number(value) -> bool:
    """True if ``value`` is a real, finite number (rejects None/NaN/inf/str)."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
