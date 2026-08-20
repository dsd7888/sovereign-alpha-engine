"""
L1 — Data Ingestor.

Fetches raw market data from free sources (yfinance) and caches aggressively
on disk so that repeated runs on the same day don't hammer Yahoo Finance.
This is the only module in the engine that talks to the network for market
data; every other layer consumes its (validated) output.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import settings
from sovereign_alpha.utils.helpers import extract_statement_row, is_cache_valid
from sovereign_alpha.utils.logger import logger
from sovereign_alpha.utils.validators import clean_ticker_list

_OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "Adj_Close"]

# Transient network conditions worth retrying; a malformed response (e.g. a
# delisted ticker) is not — retrying that just burns time for no benefit.
_RETRYABLE = retry_if_exception_type((ConnectionError, TimeoutError, OSError))

# Candidate row labels for deriving ROE/FCF from statements when `.info`
# doesn't provide them — same yfinance naming drift as DataHarmonizer's
# statement candidates.
_BS_EQUITY = ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]
_IS_NET_INCOME = ["Net Income", "Net Income Common Stockholders"]
_CF_OPERATING = [
    "Operating Cash Flow", "Total Cash From Operating Activities",
    "Cash Flow From Continuing Operating Activities",
]
_CF_CAPEX = ["Capital Expenditure", "Purchase Of PP And E", "Capital Expenditure Reported"]


class DataIngestorError(RuntimeError):
    """Raised when no usable data could be obtained for a request."""


class DataIngestor:
    """Thin, caching wrapper around yfinance."""

    # ── OHLCV ────────────────────────────────────────────────────────────

    def fetch_ohlcv(
        self,
        tickers: list[str],
        start: str | None = None,
        end: str | None = None,
        interval: str = "1d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV for ``tickers``, one flat DataFrame with a ``ticker``
        column (long format), indexed by date. Each ticker is cached
        individually as Parquet and only re-fetched once the cache is older
        than ``settings.OHLCV_CACHE_HOURS``.

        Tickers that fail to fetch (delisted, network error after retries,
        no rows) are skipped with a warning rather than aborting the whole
        batch — a single bad symbol should never take down a 50-stock run.

        The cache is keyed by ticker and freshness only, NOT by the
        requested ``(start, end)`` range — safe in production, where every
        call uses the same implicit "today minus LOOKBACK_YEARS" window, so
        a cache hit is always the right data. It is NOT safe for a caller
        that requests genuinely different date ranges across calls (e.g. a
        backtest run repeated with different ``--years``/dates within the
        cache TTL) — a hit would silently return the *previous* call's
        window. Pass ``use_cache=False`` in that situation.
        """
        tickers = clean_ticker_list(tickers)
        if not tickers:
            raise ValueError("fetch_ohlcv: empty ticker list")

        if start is None:
            start = (datetime.now() - timedelta(days=365 * settings.LOOKBACK_YEARS)).strftime("%Y-%m-%d")
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        frames: list[pd.DataFrame] = []
        failed: list[str] = []

        for ticker in tickers:
            cache_path = self._ohlcv_cache_path(ticker)

            if use_cache and is_cache_valid(cache_path, settings.OHLCV_CACHE_HOURS):
                logger.debug(f"[CACHE HIT] {ticker} OHLCV")
                df = pd.read_parquet(cache_path)
            else:
                df = self._fetch_single_ohlcv(ticker, start, end, interval)
                if df is None or df.empty:
                    logger.warning(f"[SKIP] {ticker}: OHLCV fetch returned no data")
                    failed.append(ticker)
                    continue
                df.to_parquet(cache_path)
                logger.info(f"[FETCHED] {ticker}: {len(df)} rows")

            df = df.copy()
            df["ticker"] = ticker
            frames.append(df)

        if not frames:
            raise DataIngestorError(
                f"No OHLCV data fetched for any of {len(tickers)} tickers "
                f"(all failed: {failed})"
            )
        if failed:
            logger.warning(f"[OHLCV] {len(failed)}/{len(tickers)} tickers failed: {failed}")

        combined = pd.concat(frames)
        combined.index = pd.to_datetime(combined.index)
        return combined.sort_index()

    def _ohlcv_cache_path(self, ticker: str) -> Path:
        return settings.RAW_OHLCV_DIR / f"{ticker.replace('.', '_')}.parquet"

    @retry(
        retry=_RETRYABLE,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        reraise=False,
    )
    def _fetch_single_ohlcv(
        self, ticker: str, start: str, end: str, interval: str
    ) -> pd.DataFrame | None:
        """Fetch OHLCV for a single ticker. Normalises both old and new yfinance column layouts."""
        try:
            raw = yf.download(
                ticker, start=start, end=end, interval=interval,
                auto_adjust=True, progress=False, threads=False,
            )
        except Exception as e:
            logger.error(f"[ERROR] {ticker} OHLCV fetch: {e}")
            return None

        if raw is None or raw.empty:
            return None

        # yfinance >= 0.2.38 returns MultiIndex columns: (field, ticker)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw.columns = [str(c).replace(" ", "_") for c in raw.columns]
        raw = raw.rename(columns={"Adj_Close": "Adj_Close", "Adj Close": "Adj_Close"})
        if "Adj_Close" not in raw.columns and "Close" in raw.columns:
            raw["Adj_Close"] = raw["Close"]

        missing = [c for c in _OHLCV_COLUMNS if c not in raw.columns]
        if missing:
            logger.warning(f"[SCHEMA] {ticker}: missing columns {missing}, skipping")
            return None

        out = raw[_OHLCV_COLUMNS].dropna(how="all")
        # Non-positive prices are corrupt data (e.g. yfinance's occasional
        # 0.0 rows around corporate actions) — never let those reach pricing.
        out = out[(out["Close"] > 0) & (out["Adj_Close"] > 0)]
        return out if not out.empty else None

    # ── Fundamentals ─────────────────────────────────────────────────────

    def fetch_fundamentals(self, tickers: list[str]) -> pd.DataFrame:
        """
        Fetch fundamental ratios via yfinance's ``.info`` for each ticker,
        cached for ``settings.FUNDAMENTALS_CACHE_DAYS`` days. Returns a
        DataFrame indexed by ticker; tickers that fail entirely still get a
        row of all-NaN (downstream scoring treats missing fields as neutral,
        it must never silently drop a ticker here).
        """
        tickers = clean_ticker_list(tickers)
        if not tickers:
            return pd.DataFrame()

        records: list[dict] = []
        for ticker in tickers:
            cache_path = self._fundamentals_cache_path(ticker)

            if is_cache_valid(cache_path, settings.FUNDAMENTALS_CACHE_DAYS * 24):
                logger.debug(f"[CACHE HIT] {ticker} fundamentals")
                cached = pd.read_parquet(cache_path).to_dict(orient="records")
                rec = cached[0] if cached else {"ticker": ticker}
            else:
                rec = self._fetch_single_fundamentals(ticker)
                if rec is None:
                    rec = {"ticker": ticker}
                else:
                    pd.DataFrame([rec]).to_parquet(cache_path)
                    logger.info(f"[FETCHED] {ticker} fundamentals")
                time.sleep(0.3)  # gentle rate limiting between .info calls

            records.append(rec)

        return pd.DataFrame(records).set_index("ticker")

    def _fundamentals_cache_path(self, ticker: str) -> Path:
        return settings.RAW_FUND_DIR / f"{ticker.replace('.', '_')}_fund.parquet"

    @retry(
        retry=_RETRYABLE,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        reraise=False,
    )
    def _fetch_single_fundamentals(self, ticker: str) -> dict | None:
        """Pull key ratios from yfinance ``.info`` for one ticker."""
        try:
            info = yf.Ticker(ticker).get_info()
        except Exception as e:
            logger.error(f"[ERROR] {ticker} fundamentals: {e}")
            return None

        if not info or info.get("symbol") is None:
            return None

        roe = info.get("returnOnEquity")
        fcf = info.get("freeCashflow")
        roe_source = "info" if roe is not None else "none"
        fcf_source = "info" if fcf is not None else "none"

        if roe is None or fcf is None:
            derived = self._derive_fundamentals_from_statements(ticker)
            if roe is None and derived.get("roe_derived") is not None:
                roe = derived["roe_derived"]
                roe_source = "statements"
            if fcf is None and derived.get("fcf_derived") is not None:
                fcf = derived["fcf_derived"]
                fcf_source = "statements"

        return {
            "ticker": ticker,
            "pe": info.get("trailingPE"),
            "pb": info.get("priceToBook"),
            "roe": roe,
            "roe_source": roe_source,
            "debt_equity": info.get("debtToEquity"),
            "free_cashflow": fcf,
            "fcf_source": fcf_source,
            "market_cap": info.get("marketCap"),
            "trailing_eps": info.get("trailingEps"),
            "shares_outstanding": info.get("sharesOutstanding"),
            "current_ratio": info.get("currentRatio"),
            "total_debt": info.get("totalDebt"),
            "total_assets": None,  # not in `.info`; filled in by DataHarmonizer from the balance sheet
            "ebitda": info.get("ebitda"),
            "operating_cashflow": info.get("operatingCashflow"),
            "gross_profit": info.get("grossProfits"),
            "revenue": info.get("totalRevenue"),
            "sector": info.get("sector", "Unknown"),
            "beta": info.get("beta"),
            "avg_daily_volume_10d": info.get("averageDailyVolume10Day"),
            "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
        }

    def _derive_fundamentals_from_statements(self, ticker: str) -> dict:
        """
        Compute ROE and FCF from actual financial statements when ``.info``
        doesn't provide them. More reliable for Indian stocks — yfinance's
        ``.info`` frequently omits ``returnOnEquity``/``freeCashflow`` for
        NSE large-caps (RELIANCE, ITC, COALINDIA all return ``None``).

        ROE = Net Income (latest annual) / Stockholders Equity (latest)
        FCF = Operating Cash Flow - Capital Expenditure (latest annual)

        Returns ``{"roe_derived": ..., "fcf_derived": ...}``; values are
        ``None`` where the underlying statements aren't available.
        """
        result: dict = {"roe_derived": None, "fcf_derived": None}
        try:
            tk = yf.Ticker(ticker)
            balance_sheet = tk.balance_sheet
            income_stmt = tk.financials
            cashflow_stmt = tk.cashflow
        except Exception as e:
            logger.warning(f"[DERIVE] {ticker}: statement fetch failed: {e}")
            return result

        equity_row = extract_statement_row(balance_sheet, _BS_EQUITY)
        net_income_row = extract_statement_row(income_stmt, _IS_NET_INCOME)
        ocf_row = extract_statement_row(cashflow_stmt, _CF_OPERATING)
        capex_row = extract_statement_row(cashflow_stmt, _CF_CAPEX)

        equity = float(equity_row.iloc[0]) if not equity_row.empty and pd.notna(equity_row.iloc[0]) else None
        net_income = (
            float(net_income_row.iloc[0])
            if not net_income_row.empty and pd.notna(net_income_row.iloc[0])
            else None
        )
        ocf = float(ocf_row.iloc[0]) if not ocf_row.empty and pd.notna(ocf_row.iloc[0]) else None
        capex = float(capex_row.iloc[0]) if not capex_row.empty and pd.notna(capex_row.iloc[0]) else None

        if net_income is not None and equity:
            result["roe_derived"] = net_income / equity
        if ocf is not None and capex is not None:
            result["fcf_derived"] = ocf + capex  # yfinance reports capex as a negative outflow

        return result

    # ── India VIX ────────────────────────────────────────────────────────

    def fetch_india_vix(self) -> float:
        """
        Fetch India VIX (cached ``settings.VIX_CACHE_HOURS``). Falls back to
        ``settings.VIX_DEFAULT_FALLBACK`` (a neutral value, won't trip the
        circuit breaker) if every source fails — a missing VIX reading must
        never crash the run, but it also must never look like "everything is
        fine" by accident, hence the explicit critical log.
        """
        cache_path = settings.RAW_VIX_DIR / "vix_latest.json"

        if is_cache_valid(cache_path, settings.VIX_CACHE_HOURS):
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                vix = cached.get("vix")
                if vix is not None:
                    logger.debug(f"[CACHE HIT] VIX = {vix}")
                    return float(vix)
            except (json.JSONDecodeError, ValueError, TypeError):
                pass  # fall through to a live fetch

        vix = self._fetch_vix_yfinance()
        if vix is None:
            logger.critical(
                f"[VIX] All sources failed — using fallback {settings.VIX_DEFAULT_FALLBACK} "
                "(neutral; will NOT trigger circuit breakers). Verify network connectivity."
            )
            return settings.VIX_DEFAULT_FALLBACK

        cache_path.write_text(
            json.dumps({"vix": vix, "fetched_at": datetime.now().isoformat()}), encoding="utf-8"
        )
        logger.info(f"[VIX] {vix:.2f}")
        return vix

    def _fetch_vix_yfinance(self) -> float | None:
        for sym in ("^INDIAVIX", "INDIA-VIX.NS"):
            try:
                data = yf.download(
                    sym, period="5d", interval="1d",
                    auto_adjust=True, progress=False, threads=False,
                )
                if data is not None and not data.empty:
                    close = data["Close"].dropna()
                    if isinstance(close, pd.DataFrame):  # MultiIndex edge case
                        close = close.iloc[:, 0]
                    if not close.empty:
                        return float(close.iloc[-1])
            except Exception as e:
                logger.debug(f"[VIX] {sym} fetch failed: {e}")
                continue
        return None

    # ── Benchmark ────────────────────────────────────────────────────────

    def fetch_benchmark_returns(self, start: str | None = None) -> pd.Series:
        """Daily log-returns of the Nifty 50 index, for benchmark comparison."""
        if start is None:
            start = (datetime.now() - timedelta(days=365 * settings.LOOKBACK_YEARS)).strftime("%Y-%m-%d")
        try:
            raw = yf.download(
                settings.BENCHMARK_TICKER, start=start,
                auto_adjust=True, progress=False, threads=False,
            )
            if raw is None or raw.empty:
                return pd.Series(dtype=float)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            prices = raw["Close"].dropna()
            prices = prices[prices > 0]
            return np.log(prices / prices.shift(1)).dropna()
        except Exception as e:
            logger.error(f"[ERROR] Benchmark fetch: {e}")
            return pd.Series(dtype=float)


if __name__ == "__main__":
    ingestor = DataIngestor()
    print("Testing OHLCV fetch for 3 tickers...")
    df = ingestor.fetch_ohlcv(["RELIANCE.NS", "TCS.NS", "INFY.NS"])
    print(f"Shape: {df.shape}")
    print(df.tail(3))
    print(f"\nVIX: {ingestor.fetch_india_vix()}")
    print("DataIngestor OK")
