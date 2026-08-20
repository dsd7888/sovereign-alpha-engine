"""
Price-derived raw technical features — momentum and low-volatility.

Deliberately kept separate from ``DataHarmonizer`` (which merges yfinance +
Screener.in *fundamental* data): these two factors need nothing but OHLCV,
which is both cheaper to obtain and, critically, genuinely point-in-time —
a historical daily close never changes, so a factor built only from prices
carries no lookahead risk. That property is what makes these two factors
usable in a long-horizon backtest when the fundamentals factors (which do
rely on point-in-time-uncertain scraped/derived data) are not.

Output feeds ``FundamentalScorer.score_momentum``/``score_low_vol``, which
percentile-rank these raw values across the universe exactly like every
other USHS input — this module only computes the raw numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings
from sovereign_alpha.utils.logger import logger


def compute_technical_features(
    ohlcv: pd.DataFrame, tickers: list[str], as_of: pd.Timestamp | None = None
) -> pd.DataFrame:
    """
    Computes 12-1 momentum and trailing low-volatility for each ticker from
    a long-format OHLCV DataFrame (the same shape ``DataIngestor.fetch_ohlcv``
    returns: date-indexed, with a ``ticker`` column and an ``Adj_Close``
    column).

    ``as_of``, when given, restricts every ticker's price series to rows
    with date <= as_of before computing anything — this is what makes the
    function safe to call from inside a historical backtest loop without
    leaking future prices into today's "current" momentum reading. Left
    as ``None`` for production use (all available history, i.e. "today").

    Returns a DataFrame indexed by ticker with columns ``momentum_12_1``
    (float, cumulative log-return) and ``low_vol_6m`` (float, annualised
    realised volatility) — NaN for any ticker with insufficient history,
    which downstream percentile-scoring treats as neutral (50), not as a
    disqualifying failure.
    """
    if as_of is not None:
        ohlcv = ohlcv[ohlcv.index <= as_of]

    records: dict[str, dict[str, float]] = {}
    for ticker in tickers:
        subset = ohlcv.loc[ohlcv["ticker"] == ticker, "Adj_Close"].dropna().sort_index()
        records[ticker] = _features_from_price_series(subset)

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "ticker"
    n_valid = df["momentum_12_1"].notna().sum()
    logger.info(
        f"[TECHNICAL] Momentum/low-vol computed for {n_valid}/{len(tickers)} tickers "
        f"(rest lack {settings.MIN_TECHNICAL_HISTORY_DAYS}+ days of history)"
    )
    return df


def _features_from_price_series(prices: pd.Series) -> dict[str, float]:
    if len(prices) < settings.MIN_TECHNICAL_HISTORY_DAYS:
        return {"momentum_12_1": np.nan, "low_vol_6m": np.nan}

    log_ret = np.log(prices / prices.shift(1)).dropna()

    # 12-1 momentum: cumulative return from t-252 to t-21, skipping the most
    # recent MOMENTUM_SKIP_DAYS to avoid the well-documented short-term
    # (1-month) reversal effect contaminating the momentum signal.
    window = log_ret.iloc[-settings.MOMENTUM_LOOKBACK_DAYS: -settings.MOMENTUM_SKIP_DAYS]
    momentum = float(window.sum()) if len(window) > 0 else np.nan

    vol_window = log_ret.iloc[-settings.LOW_VOL_LOOKBACK_DAYS:]
    low_vol = (
        float(vol_window.std(ddof=1) * np.sqrt(settings.TRADING_DAYS))
        if len(vol_window) >= 2
        else np.nan
    )

    return {"momentum_12_1": momentum, "low_vol_6m": low_vol}
