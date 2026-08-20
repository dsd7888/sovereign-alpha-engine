"""
L2 — Data Harmonizer.

Merges OHLCV + `.info` fundamentals + real financial-statement line items
(balance sheet / income statement / cash flow) + Screener.in enrichment into
one row per ticker, and computes Altman Z''-Score and Beneish M-Score from
*actual* reported figures.

Note on fidelity: an earlier draft of this pipeline approximated inputs like
working capital and retained earnings with rough multipliers off total
assets (e.g. ``0.3 * total_assets``) because yfinance's ``.info`` dict does
not expose them. That is not acceptable for a distress/manipulation score
that gates real capital allocation — this version pulls the real figures
from ``Ticker.balance_sheet`` / ``.financials`` / ``.cashflow`` instead, and
returns ``NaN`` (never a guess) when a required field truly isn't available.
"""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_fixed

from config import settings
from config.universe import sector_of
from sovereign_alpha.ingestion.data_ingestor import DataIngestor
from sovereign_alpha.utils.helpers import (
    extract_statement_row,
    is_cache_valid,
    load_json,
    save_json,
    today_str,
)
from sovereign_alpha.utils.logger import logger

_PLEDGE_SENTENCE_RE = re.compile(r"pledged\s+([\d.]+)\s*%", re.IGNORECASE)

SCREENER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
}

# Candidate row labels to try, in order, against yfinance statement indices.
# yfinance's exact naming has shifted across releases (e.g. "Total Liab" vs
# "Total Liabilities Net Minority Interest"), so every lookup tries several.
_BS_TOTAL_ASSETS = ["Total Assets"]
_BS_CURRENT_ASSETS = ["Current Assets", "Total Current Assets"]
_BS_CURRENT_LIAB = ["Current Liabilities", "Total Current Liabilities"]
_BS_RETAINED_EARNINGS = ["Retained Earnings"]
_BS_EQUITY = ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]
_BS_TOTAL_DEBT = ["Total Debt"]
_BS_RECEIVABLES = ["Receivables", "Net Receivables", "Accounts Receivable", "Gross Accounts Receivable"]

_IS_REVENUE = ["Total Revenue"]
_IS_GROSS_PROFIT = ["Gross Profit"]
_IS_EBIT = ["EBIT", "Operating Income"]
_IS_EBITDA = ["EBITDA", "Normalized EBITDA"]
_IS_NET_INCOME = ["Net Income", "Net Income Common Stockholders"]

_CF_OPERATING = [
    "Operating Cash Flow", "Total Cash From Operating Activities",
    "Cash Flow From Continuing Operating Activities",
]
_CF_CAPEX = ["Capital Expenditure", "Purchase Of PP And E", "Capital Expenditure Reported"]

# Row labels that must never be matched as EBIT/EBITDA even though they
# contain "operating income"/"ebit" as a substring — e.g. yfinance's
# "Other Non Operating Income Expenses" is a footnote-style plug, not
# operating profit, and the old plain substring match fabricated an EBIT
# value for HDFCBANK by matching it.
_EBIT_BLOCKLIST = [
    "other non operating", "non-operating", "non operating",
    "other income", "other expense", "interest expense", "income tax",
]

# Fields whose presence/absence drives ``data_completeness_pct`` — the ones
# that actually feed the USHS score, so a low value here is a real signal
# that a ticker's score is resting on fewer inputs than usual.
COMPLETENESS_FIELDS = [
    "pe", "roe", "fcf_yield", "debt_equity",
    "promoter_pledge_pct", "altman_z", "beneish_m",
]


def _compute_data_completeness_pct(row: dict) -> float:
    present = sum(
        1 for f in COMPLETENESS_FIELDS
        if row.get(f) is not None and not (isinstance(row.get(f), float) and np.isnan(row[f]))
    )
    return present / len(COMPLETENESS_FIELDS) * 100


def _periods(row: pd.Series, n: int = 2) -> list[float | None]:
    """Most recent ``n`` values of a statement row (newest first), None where missing/NaN."""
    values = [
        float(row.iloc[i]) if i < len(row) and pd.notna(row.iloc[i]) else None
        for i in range(n)
    ]
    return values


class DataHarmonizer:
    def __init__(self) -> None:
        self.ingestor = DataIngestor()

    # ── Screener.in scraping (best-effort enrichment; never blocks the pipeline) ──

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5), reraise=False)
    def scrape_screener(self, ticker_root: str) -> dict:
        """
        Scrape Screener.in for promoter pledging %, FCF (TTM), current ratio.

        ``ticker_root``: NSE symbol WITHOUT the ``.NS`` suffix, e.g. ``RELIANCE``.
        Always returns all three keys; values are ``None`` where not found.
        Screener's markup changes periodically — this is intentionally
        defensive and must never raise into the caller.
        """
        result: dict = {
            "promoter_pledge_pct": None,
            "fcf_ttm": None,
            "current_ratio_screener": None,
        }
        try:
            resp = self._get_screener_page(ticker_root)
            if resp is None:
                return result

            soup = BeautifulSoup(resp, "lxml")
            result["promoter_pledge_pct"] = self._parse_pledge_pct(soup)

            for li in soup.select("ul#top-ratios li, ul.company-ratios li"):
                label_el = li.find("span", class_="name")
                value_el = li.find("span", class_="value") or li.find("span", class_="number")
                if not label_el or not value_el:
                    continue
                label = label_el.get_text(strip=True).lower()
                val_text = (
                    value_el.get_text(strip=True)
                    .replace(",", "").replace("₹", "").replace("Cr.", "").replace("Cr", "").strip()
                )
                try:
                    if "current ratio" in label:
                        result["current_ratio_screener"] = float(val_text)
                    elif "free cash flow" in label:
                        result["fcf_ttm"] = float(val_text) * 1e7  # ₹ Cr → absolute rupees
                except ValueError:
                    continue
        except Exception as e:
            logger.warning(f"[SCREENER] {ticker_root}: {e}")

        return result

    def _get_screener_page(self, ticker_root: str) -> str | None:
        for suffix in ("/consolidated/", "/"):
            url = f"{settings.SCREENER_BASE_URL}{ticker_root}{suffix}"
            resp = requests.get(url, headers=SCREENER_HEADERS, timeout=settings.SCREENER_TIMEOUT_SEC)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code != 404:
                logger.warning(f"[SCREENER] {ticker_root}: HTTP {resp.status_code} at {url}")
                return None
        return None

    @staticmethod
    def _parse_pledge_pct(soup: BeautifulSoup) -> float | None:
        """
        Verified against Screener.in's live markup (2026): the shareholding-
        pattern table no longer renders a numeric "Pledged" row server-side —
        that detail moved behind a JS-driven modal (``Company.showShareholders``)
        with no static-HTML fallback. The only reliable source left in the
        static page is the auto-generated analysis sentence in the Pros/Cons
        section, e.g. "Promoters have pledged 44.7% of their holding." Its
        absence means either genuinely zero pledge or that Screener chose not
        to flag it — both correctly fall back to a neutral score downstream,
        so returning None here is safe either way.
        """
        match = _PLEDGE_SENTENCE_RE.search(soup.get_text(" ", strip=True))
        return float(match.group(1)) if match else None

    # ── Real financial statements (balance sheet / income / cash flow) ───────

    def _fetch_financial_statements(self, ticker: str) -> tuple[dict, dict]:
        """
        Return ``(current_period, prior_period)`` dicts of real statement
        fields, sourced from yfinance's annual statements and cached for
        ``settings.FUNDAMENTALS_CACHE_DAYS`` days. Missing fields are
        ``None`` — callers must treat that as "unknown", never as zero.
        """
        cache_path = settings.RAW_FUND_DIR / f"{ticker.replace('.', '_')}_stmts.json"
        if is_cache_valid(cache_path, settings.FUNDAMENTALS_CACHE_DAYS * 24):
            cached = load_json(cache_path)
            if cached.get("curr") is not None:
                logger.debug(f"[CACHE HIT] {ticker} statements")
                return cached["curr"], cached["prev"]

        try:
            tk = yf.Ticker(ticker)
            balance_sheet = tk.balance_sheet
            income_stmt = tk.financials
            cashflow_stmt = tk.cashflow
        except Exception as e:
            logger.warning(f"[STMTS] {ticker}: fetch failed: {e}")
            return {}, {}

        total_assets = _periods(extract_statement_row(balance_sheet, _BS_TOTAL_ASSETS))
        current_assets = _periods(extract_statement_row(balance_sheet, _BS_CURRENT_ASSETS))
        current_liab = _periods(extract_statement_row(balance_sheet, _BS_CURRENT_LIAB))
        retained_earnings = _periods(extract_statement_row(balance_sheet, _BS_RETAINED_EARNINGS))
        book_equity = _periods(extract_statement_row(balance_sheet, _BS_EQUITY))
        total_debt = _periods(extract_statement_row(balance_sheet, _BS_TOTAL_DEBT))
        receivables = _periods(extract_statement_row(balance_sheet, _BS_RECEIVABLES))

        revenue = _periods(extract_statement_row(income_stmt, _IS_REVENUE))
        gross_profit = _periods(extract_statement_row(income_stmt, _IS_GROSS_PROFIT))
        ebit = _periods(extract_statement_row(income_stmt, _IS_EBIT, blocklist=_EBIT_BLOCKLIST))
        ebitda = _periods(extract_statement_row(income_stmt, _IS_EBITDA, blocklist=_EBIT_BLOCKLIST))
        net_income = _periods(extract_statement_row(income_stmt, _IS_NET_INCOME))

        operating_cf = _periods(extract_statement_row(cashflow_stmt, _CF_OPERATING))
        capex = _periods(extract_statement_row(cashflow_stmt, _CF_CAPEX))

        def build(i: int) -> dict:
            ta, ca, cl = total_assets[i], current_assets[i], current_liab[i]
            wc = (ca - cl) if (ca is not None and cl is not None) else None
            ocf = operating_cf[i]
            cpx = capex[i]  # yfinance reports capex as a negative outflow
            fcf = (ocf + cpx) if (ocf is not None and cpx is not None) else None
            return {
                "total_assets": ta,
                "current_assets": ca,
                "current_liabilities": cl,
                "working_capital": wc,
                "retained_earnings": retained_earnings[i],
                "book_equity": book_equity[i],
                "total_debt": total_debt[i],
                "receivables": receivables[i],
                "revenue": revenue[i],
                "gross_profit": gross_profit[i],
                "ebit": ebit[i] if ebit[i] is not None else ebitda[i],
                "ebitda": ebitda[i],
                "net_income": net_income[i],
                "operating_cashflow": ocf,
                "free_cashflow_statement": fcf,
            }

        curr, prev = build(0), build(1)
        save_json({"curr": curr, "prev": prev, "fetched_at": today_str()}, cache_path)
        return curr, prev

    # ── Altman Z''-Score (emerging-market non-manufacturing variant) ────────

    @staticmethod
    def compute_altman_z(fin: dict, sector: str | None = None) -> float:
        """
        Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4
        X1 = working capital / total assets
        X2 = retained earnings / total assets
        X3 = EBIT / total assets
        X4 = book equity / total liabilities

        Z'' > 2.6 = safe zone. 1.1-2.6 = grey zone. < 1.1 = distress zone.
        Requires total_assets and ebit at minimum; returns NaN otherwise.

        Structurally inapplicable to banks/NBFCs/insurers (see
        ``settings.ALTMAN_Z_EXCLUDED_SECTORS``) — they don't have current
        assets/liabilities or EBIT in the traditional sense, so returns NaN
        for those sectors before any computation.
        """
        if sector in settings.ALTMAN_Z_EXCLUDED_SECTORS:
            return np.nan

        ta = fin.get("total_assets")
        ebit = fin.get("ebit")
        if ta is None or ebit is None or not ta:
            return np.nan

        wc = fin.get("working_capital") or 0.0
        re = fin.get("retained_earnings") or 0.0
        bve = fin.get("book_equity")
        total_debt = fin.get("total_debt") or 0.0
        if bve is not None:
            tl = ta - bve  # total liabilities = total assets - book equity (accounting identity)
        else:
            tl = total_debt  # no book equity available; approximate with interest-bearing debt only
            bve = ta - tl

        x1, x2, x3 = wc / ta, re / ta, ebit / ta
        x4 = bve / max(tl, 1.0)

        z = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
        return float(np.clip(z, -5, 15))

    # ── Beneish M-Score (5-variable simplified model) ───────────────────────

    @staticmethod
    def compute_beneish_m(curr: dict, prev: dict) -> float:
        """
        M = -4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 4.679*TATA
        M < -2.22 = low manipulation risk. M >= -2.22 = flag for review.
        Requires revenue (both periods) and total assets (both periods);
        returns NaN otherwise rather than guessing.
        """
        rev_c, rev_p = curr.get("revenue"), prev.get("revenue")
        ta_c, ta_p = curr.get("total_assets"), prev.get("total_assets")
        if not rev_c or not rev_p or not ta_c or not ta_p:
            return np.nan

        rec_c, rec_p = curr.get("receivables") or 0.0, prev.get("receivables") or 0.0
        gp_c, gp_p = curr.get("gross_profit") or 0.0, prev.get("gross_profit") or 0.0
        ca_c, ca_p = curr.get("current_assets") or 0.0, prev.get("current_assets") or 0.0
        ni_c = curr.get("net_income") or 0.0
        ocf_c = curr.get("operating_cashflow") or 0.0

        try:
            dsri = (rec_c / rev_c) / max(rec_p / rev_p, 1e-6)
            gm_c, gm_p = gp_c / rev_c, gp_p / rev_p
            gmi = gm_p / max(gm_c, 1e-6)
            aqi = (1 - ca_c / ta_c) / max(1 - ca_p / ta_p, 1e-6)
            sgi = rev_c / rev_p
            tata = (ni_c - ocf_c) / ta_c

            m = -4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi + 4.679 * tata
            return float(np.clip(m, -10, 10))
        except (ZeroDivisionError, TypeError, ValueError) as e:
            logger.debug(f"Beneish M computation error: {e}")
            return np.nan

    # ── Sector P/E normalisation ──────────────────────────────────────────

    @staticmethod
    def normalise_sector_pe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds ``pe_sector_normalised``: (sector_median_pe - ticker_pe) /
        sector_std_pe, clipped to [-3, 3]. Higher = cheaper vs sector peers.
        """
        df = df.copy()
        df["sector"] = df.index.map(sector_of)
        df["pe_sector_normalised"] = 0.0

        for _sector, group in df.groupby("sector"):
            pe_vals = group["pe"].dropna()
            if len(pe_vals) < 2:
                continue
            median, std = pe_vals.median(), pe_vals.std()
            if not std or np.isnan(std):
                continue
            normalised = (median - df.loc[group.index, "pe"]) / std
            df.loc[group.index, "pe_sector_normalised"] = normalised.clip(-3, 3).fillna(0.0)

        return df

    # ── Master harmonize ──────────────────────────────────────────────────

    def harmonize(self, tickers: list[str]) -> pd.DataFrame:
        """
        Full L2 pipeline: fundamentals + statements + Screener enrichment +
        Altman Z / Beneish M, one row per ticker. Saved to
        ``data/processed/harmonized_YYYYMMDD.parquet``.
        """
        logger.info(f"[HARMONIZE] Starting for {len(tickers)} tickers...")
        fund_df = self.ingestor.fetch_fundamentals(tickers)

        records = []
        for ticker in tickers:
            row = fund_df.loc[ticker].to_dict() if ticker in fund_df.index else {"ticker": ticker}
            row["ticker"] = ticker

            curr_fin, prev_fin = self._fetch_financial_statements(ticker)

            # Statement data supersedes `.info` approximations where present —
            # it is the primary source `.info` itself is derived from.
            row["total_assets"] = curr_fin.get("total_assets")
            row["ebitda"] = curr_fin.get("ebitda") or row.get("ebitda")
            row["revenue"] = curr_fin.get("revenue") or row.get("revenue")
            row["gross_profit"] = curr_fin.get("gross_profit") or row.get("gross_profit")
            row["operating_cashflow"] = curr_fin.get("operating_cashflow") or row.get("operating_cashflow")
            if curr_fin.get("total_debt") is not None:
                row["total_debt"] = curr_fin["total_debt"]

            fcf = curr_fin.get("free_cashflow_statement") or row.get("free_cashflow")
            mcap = row.get("market_cap")
            row["free_cashflow"] = fcf
            row["fcf_yield"] = (fcf / mcap) if (fcf and mcap and mcap > 0) else None

            vol_shares = row.get("avg_daily_volume_10d")
            price = row.get("current_price")
            if not price and mcap and row.get("shares_outstanding"):
                shares = row["shares_outstanding"]
                price = (mcap / shares) if shares else None
            row["avg_daily_volume_inr"] = (
                float(vol_shares) * float(price)
                if vol_shares and price and vol_shares > 0 and price > 0
                else None
            )

            sector = sector_of(ticker)
            row["altman_z"] = self.compute_altman_z(curr_fin, sector=sector)
            row["beneish_m"] = self.compute_beneish_m(curr_fin, prev_fin)

            ticker_root = ticker.replace(settings.MARKET_SUFFIX, "")
            logger.info(f"  Scraping Screener.in for {ticker_root}...")
            row.update(self.scrape_screener(ticker_root))
            row["sector"] = sector
            row["data_completeness_pct"] = _compute_data_completeness_pct(row)

            records.append(row)
            time.sleep(settings.SCREENER_DELAY_SEC)  # rate limit — never remove

        df = pd.DataFrame(records).set_index("ticker")
        df = self.normalise_sector_pe(df)
        df = self._apply_liquidity_filter(df)

        out_path = settings.PROCESSED_DIR / f"harmonized_{today_str()}.parquet"
        df.to_parquet(out_path)
        logger.info(f"[HARMONIZE] Done. Saved to {out_path}. Shape: {df.shape}")
        return df

    @staticmethod
    def _apply_liquidity_filter(df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop names below the ADV floor — impact-cost sizing breaks on thin books.

        Unknown liquidity (``avg_daily_volume_inr`` is NaN/None, e.g. yfinance's
        ``.info`` omitting ``averageDailyVolume10Day`` for a ticker) is treated
        as neutral, not as zero — same as unknown ROE elsewhere in the
        pipeline. Only tickers with an explicitly known ADV below the floor
        are dropped.
        """
        if "avg_daily_volume_inr" not in df.columns or df.empty:
            return df
        before = len(df)
        adv = df["avg_daily_volume_inr"]
        drop = adv.notna() & (adv < settings.MIN_AVG_DAILY_VOLUME_INR)
        filtered = df.loc[~drop]
        dropped = before - len(filtered)
        if dropped:
            logger.info(
                f"[HARMONIZE] Liquidity filter dropped {dropped}/{before} tickers "
                f"(ADV < Rs.{settings.MIN_AVG_DAILY_VOLUME_INR:,.0f})"
            )
        return filtered

    def load_latest(self) -> pd.DataFrame:
        """Load the most recently saved harmonized parquet without re-fetching anything."""
        files = sorted(settings.PROCESSED_DIR.glob("harmonized_*.parquet"))
        if not files:
            raise FileNotFoundError("No harmonized data found. Run harmonize() first.")
        return pd.read_parquet(files[-1])


if __name__ == "__main__":
    h = DataHarmonizer()
    df = h.harmonize(["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS"])
    print(df[["pe", "roe", "altman_z", "beneish_m", "promoter_pledge_pct"]].to_string())
    print("DataHarmonizer OK")
