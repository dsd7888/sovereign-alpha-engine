"""
Point-in-time fundamentals reconstruction for the "full" backtest mode.

Screener.in has no historical API and no local snapshot history exists
beyond the last couple of weeks the engine has actually been running (see
``run_backtest.py`` module docstring for the full explanation of why), so
promoter-pledge and sentiment simply cannot be backtested honestly — both
are held neutral (50) here, exactly like production already does for
sentiment (a real gap, not a fudge).

What *is* honestly reconstructable: yfinance's annual balance
sheet/income/cashflow statements, typically ~5 fiscal years deep for NSE
large- and mid-caps. This module walks those periods and, for each one,
computes the exact same ``altman_z``/``beneish_m`` figures
``DataHarmonizer`` computes in production — by calling its static methods
directly, not reimplementing the formulas — so a backtest run and a live
run can never silently drift apart on what "Altman Z" means.

Reporting lag: a fiscal year's results are not knowable to a trader on the
fiscal year-end date, only some weeks after the company actually files them.
Every row here is indexed by "known_date" (period end +
``ANNUAL_RESULTS_REPORTING_LAG_DAYS``), and a backtest slicing on
``known_date <= as_of`` is what keeps this lookahead-safe.

PE/PB/FCF-yield need a market price and a share count, neither of which
has reliable point-in-time history from free sources. Share count is
approximated with the *current* ``sharesOutstanding`` applied across all
historical periods (a real approximation — share counts do drift via
buybacks/issuance/splits, but far more slowly than price or earnings, and
this is disclosed here rather than silently assumed). Price is genuinely
point-in-time (the caller supplies the as-of price from real OHLCV); only
the share count is approximated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from config import settings
from sovereign_alpha.ingestion.data_harmonizer import (
    _BS_CURRENT_ASSETS,
    _BS_CURRENT_LIAB,
    _BS_EQUITY,
    _BS_RECEIVABLES,
    _BS_RETAINED_EARNINGS,
    _BS_TOTAL_ASSETS,
    _BS_TOTAL_DEBT,
    _CF_CAPEX,
    _CF_OPERATING,
    _EBIT_BLOCKLIST,
    _IS_EBIT,
    _IS_GROSS_PROFIT,
    _IS_NET_INCOME,
    _IS_REVENUE,
    DataHarmonizer,
)
from sovereign_alpha.ingestion.data_ingestor import DataIngestor
from sovereign_alpha.utils.helpers import extract_statement_row
from sovereign_alpha.utils.logger import logger

_NEUTRAL_ONLY_COLUMNS = ["promoter_pledge_pct", "sentiment_score"]


def _row_at(row: pd.Series, i: int) -> float | None:
    return float(row.iloc[i]) if i < len(row) and pd.notna(row.iloc[i]) else None


def fetch_point_in_time_fundamentals(ticker: str, sector: str) -> pd.DataFrame:
    """
    One row per fiscal year yfinance has annual statements for, indexed by
    ``known_date`` (ascending). Columns: ``altman_z``, ``beneish_m``, ``roe``,
    ``debt_equity``, plus the raw ``net_income``/``book_equity`` needed to
    turn a later as-of price into PE/PB/fcf_yield. Empty DataFrame if no
    usable statement history exists for this ticker (never raises — a
    single ticker's missing history must not abort a backtest).
    """
    try:
        tk = yf.Ticker(ticker)
        balance_sheet = tk.balance_sheet
        income_stmt = tk.financials
        cashflow_stmt = tk.cashflow
    except Exception as e:
        logger.warning(f"[PIT] {ticker}: statement fetch failed: {e}")
        return pd.DataFrame()

    if balance_sheet is None or balance_sheet.empty:
        return pd.DataFrame()

    period_ends = list(balance_sheet.columns)
    n_periods = len(period_ends)
    if n_periods < 2:
        return pd.DataFrame()  # need at least curr+prev for Beneish M

    total_assets = extract_statement_row(balance_sheet, _BS_TOTAL_ASSETS)
    current_assets = extract_statement_row(balance_sheet, _BS_CURRENT_ASSETS)
    current_liab = extract_statement_row(balance_sheet, _BS_CURRENT_LIAB)
    retained_earnings = extract_statement_row(balance_sheet, _BS_RETAINED_EARNINGS)
    book_equity = extract_statement_row(balance_sheet, _BS_EQUITY)
    total_debt = extract_statement_row(balance_sheet, _BS_TOTAL_DEBT)
    receivables = extract_statement_row(balance_sheet, _BS_RECEIVABLES)

    revenue = extract_statement_row(income_stmt, _IS_REVENUE)
    gross_profit = extract_statement_row(income_stmt, _IS_GROSS_PROFIT)
    ebit = extract_statement_row(income_stmt, _IS_EBIT, blocklist=_EBIT_BLOCKLIST)
    net_income = extract_statement_row(income_stmt, _IS_NET_INCOME)

    operating_cf = extract_statement_row(cashflow_stmt, _CF_OPERATING)
    capex = extract_statement_row(cashflow_stmt, _CF_CAPEX)

    def build(i: int) -> dict:
        ta, ca, cl = _row_at(total_assets, i), _row_at(current_assets, i), _row_at(current_liab, i)
        wc = (ca - cl) if (ca is not None and cl is not None) else None
        ocf, cpx = _row_at(operating_cf, i), _row_at(capex, i)
        fcf = (ocf + cpx) if (ocf is not None and cpx is not None) else None
        return {
            "total_assets": ta, "working_capital": wc,
            "retained_earnings": _row_at(retained_earnings, i),
            "book_equity": _row_at(book_equity, i), "total_debt": _row_at(total_debt, i),
            "receivables": _row_at(receivables, i), "revenue": _row_at(revenue, i),
            "gross_profit": _row_at(gross_profit, i), "ebit": _row_at(ebit, i),
            "net_income": _row_at(net_income, i), "operating_cashflow": ocf,
            "current_assets": ca, "free_cashflow": fcf,
        }

    records = []
    lag = pd.Timedelta(days=settings.ANNUAL_RESULTS_REPORTING_LAG_DAYS)
    for i in range(n_periods - 1):  # need i+1 (prior year) for Beneish M
        curr, prev = build(i), build(i + 1)
        altman_z = DataHarmonizer.compute_altman_z(curr, sector=sector)
        beneish_m = DataHarmonizer.compute_beneish_m(curr, prev)

        roe = (
            curr["net_income"] / curr["book_equity"]
            if curr["net_income"] is not None and curr["book_equity"]
            else np.nan
        )
        debt_equity = (
            (curr["total_debt"] / curr["book_equity"]) * 100
            if curr["total_debt"] is not None and curr["book_equity"]
            else np.nan
        )  # x100 to match yfinance's debtToEquity convention (percentage, not ratio)

        records.append({
            "known_date": period_ends[i] + lag,
            "period_end": period_ends[i],
            "altman_z": altman_z,
            "beneish_m": beneish_m,
            "roe": roe,
            "debt_equity": debt_equity,
            "net_income": curr["net_income"],
            "book_equity": curr["book_equity"],
            "free_cashflow": curr["free_cashflow"],
        })

    df = pd.DataFrame(records).sort_values("known_date").reset_index(drop=True)
    return df


def as_of_fundamentals(
    pit_df: pd.DataFrame, as_of: pd.Timestamp, current_price: float, shares_outstanding: float | None
) -> dict:
    """
    The single fundamentals snapshot a trader would actually have known on
    ``as_of``: the most recent row whose ``known_date`` <= ``as_of``.
    Converts the stored raw fields into PE/PB/fcf_yield using ``as_of``'s
    real point-in-time price and an approximated (current) share count —
    see module docstring. Returns an all-NaN dict if no period was known
    yet by ``as_of`` (e.g. simulating before the company's first available
    statement).
    """
    empty = {"altman_z": np.nan, "beneish_m": np.nan, "roe": np.nan, "debt_equity": np.nan,
             "pe": np.nan, "pb": np.nan, "fcf_yield": np.nan}
    if pit_df.empty:
        return empty

    known = pit_df[pit_df["known_date"] <= as_of]
    if known.empty:
        return empty
    row = known.iloc[-1]

    market_cap = current_price * shares_outstanding if shares_outstanding else None
    pe = (
        market_cap / row["net_income"]
        if market_cap is not None and row["net_income"] and row["net_income"] > 0
        else np.nan
    )
    pb = market_cap / row["book_equity"] if market_cap is not None and row["book_equity"] else np.nan
    fcf_yield = (
        row["free_cashflow"] / market_cap
        if market_cap and row["free_cashflow"] is not None
        else np.nan
    )

    return {
        "altman_z": row["altman_z"], "beneish_m": row["beneish_m"],
        "roe": row["roe"], "debt_equity": row["debt_equity"],
        "pe": pe, "pb": pb, "fcf_yield": fcf_yield,
    }


def build_universe_pit_store(tickers: list[str], sectors: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Fetches ``fetch_point_in_time_fundamentals`` for every ticker once, up front."""
    store: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        store[ticker] = fetch_point_in_time_fundamentals(ticker, sectors.get(ticker, "Unknown"))
    n_usable = sum(1 for df in store.values() if not df.empty)
    logger.info(f"[PIT] Point-in-time fundamentals reconstructed for {n_usable}/{len(tickers)} tickers")
    return store


def fetch_current_shares_outstanding(tickers: list[str]) -> dict[str, float]:
    """Current share count per ticker, via the existing cached fundamentals fetch."""
    fund_df = DataIngestor().fetch_fundamentals(tickers)
    if "shares_outstanding" not in fund_df.columns:
        return {}
    return fund_df["shares_outstanding"].dropna().to_dict()
