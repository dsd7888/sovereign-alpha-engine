"""
Builds the trading universe from real, periodically-refreshed NSE index
constituent data, instead of a hand-maintained static list.

Why this exists: a hardcoded ticker list is permanently behind the market
by construction — it can never contain a stock that IPO'd or got promoted
into an index after the list was last edited, and nobody's job is to keep
it current. Emerging-theme names (defence, semiconductors/electronics
manufacturing, EV/battery) mostly live in the Smallcap tier and NSE's
dedicated "Nifty India Defence" index, neither of which the old
Nifty 50 + Midcap 100 list touched at all.

Runs periodically (see ``reconstitute_universe.py`` /
``.github/workflows/universe_reconstitution.yml`` — monthly, since real
index reconstitution only happens quarterly/semi-annually), NOT on the
daily trading engine's critical path. The daily engine just reads
whatever the last reconstitution wrote to ``config/universe_data.json``.
That keeps the live trading day's live-dependency surface at yfinance +
Screener.in — it does not add a third live scrape of a notoriously
bot-hostile NSE endpoint to the thing that has to succeed every weekday
at 17:33.

Sector: NSE's own "Industry" column is used directly rather than
remapped through a hand-picked taxonomy — it already buckets banks/NBFCs/
insurers under "Financial Services", which is one of the exact strings in
``settings.ALTMAN_Z_EXCLUDED_SECTORS``, so the exclusion this project
already relies on keeps working with zero translation, and one fewer
hand-maintained table that can silently go stale.
"""
from __future__ import annotations

import random
from datetime import datetime
from io import StringIO

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_fixed

from config import settings
from sovereign_alpha.utils.logger import logger

# NSE's constituent CSVs are sorted alphabetically by company name, not by
# weight/market-cap/liquidity — nothing in the source data ranks
# constituents meaningfully. Taking the first N rows of a per-tier cap
# would therefore just mean "whichever companies' names start with A-D",
# a completely arbitrary bias, not a selection. A fixed-seed shuffle before
# capping is unbiased and reproducible — the actual quality filtering
# happens downstream anyway, via USHS scoring and the existing
# MIN_AVG_DAILY_VOLUME_INR liquidity gate in get_eligible_tickers.
UNIVERSE_SAMPLE_SEED = 42

_NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/csv,application/csv,*/*",
}
_NSE_ARCHIVE_BASE = "https://archives.nseindia.com/content/indices/"

# (slug, tier, per-source cap). Nifty50/Next50/Defence/Midcap150 are kept
# in full — each is already a deliberately bounded, curated set (liquidity
# anchor + the exact thematic gap this module exists to close + a
# well-curated 150 names, not an oversized pool). Only Smallcap250 gets a
# cap — 250 names is genuinely too many for the Screener.in rate limit to
# absorb in full alongside everything else, but it's also exactly where
# names like Kaynes Technology, Syrma SGS and Data Patterns actually live,
# so it gets a substantial (not token) share of the budget.
INDEX_SOURCES: list[tuple[str, str, int | None]] = [
    ("ind_nifty50list.csv", "large", None),
    ("ind_niftyindiadefence_list.csv", "thematic", None),
    ("ind_niftynext50list.csv", "large", None),
    ("ind_niftymidcap150list.csv", "mid", None),
    ("ind_niftysmallcap250list.csv", "small", 120),
]

# Overall safety ceiling regardless of per-tier caps summing higher (caps
# above sum to at most 50+19+50+150+120=389 before de-duplication) —
# Screener.in scraping is rate-limited to one request per
# settings.SCREENER_DELAY_SEC, so universe size directly sets the daily
# run's wall-clock time. See .github/workflows/daily_run.yml's
# timeout-minutes, bumped alongside this.
DEFAULT_MAX_UNIVERSE_SIZE = 400


class UniverseBuilderError(RuntimeError):
    pass


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5), reraise=False)
def fetch_index_constituents(slug: str) -> pd.DataFrame | None:
    """One NSE index constituent CSV as a DataFrame, or None on any failure — never raises."""
    url = f"{_NSE_ARCHIVE_BASE}{slug}"
    try:
        resp = requests.get(url, headers=_NSE_HEADERS, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"[UNIVERSE] {slug}: HTTP {resp.status_code}")
            return None
        df = pd.read_csv(StringIO(resp.text))
        df.columns = [c.strip() for c in df.columns]
        if "Symbol" not in df.columns:
            logger.warning(f"[UNIVERSE] {slug}: unexpected schema, columns={list(df.columns)}")
            return None
        return df
    except Exception as e:
        logger.warning(f"[UNIVERSE] {slug}: fetch failed: {e}")
        return None


def build_universe(max_size: int = DEFAULT_MAX_UNIVERSE_SIZE) -> dict:
    """
    Fetches every source in ``INDEX_SOURCES`` in priority order and merges
    them into one universe, capped at ``max_size`` total tickers (a real
    constraint: Screener.in scraping is rate-limited to one request per
    ``settings.SCREENER_DELAY_SEC``, so universe size directly sets the
    daily run's wall-clock time).

    Returns a dict — never raises unless *every* source failed, since a
    genuinely empty universe is the one outcome that must stop the
    reconstitution run rather than silently writing nothing useful.
    """
    tickers: list[str] = []
    tier_of: dict[str, str] = {}
    sector_map: dict[str, str] = {}
    sources_status: dict[str, str] = {}

    for slug, tier, per_source_cap in INDEX_SOURCES:
        if len(tickers) >= max_size:
            sources_status[slug] = "SKIPPED (universe cap already reached)"
            continue

        df = fetch_index_constituents(slug)
        if df is None or df.empty:
            sources_status[slug] = "FAILED"
            continue
        sources_status[slug] = f"OK ({len(df)} rows)"

        if per_source_cap is not None and len(df) > per_source_cap:
            df = df.sample(frac=1.0, random_state=random.Random(UNIVERSE_SAMPLE_SEED).randint(0, 2**31))

        added_from_source = 0
        for _, row in df.iterrows():
            if len(tickers) >= max_size:
                break
            if per_source_cap is not None and added_from_source >= per_source_cap:
                break
            symbol = str(row["Symbol"]).strip()
            if not symbol:
                continue
            ticker = f"{symbol}{settings.MARKET_SUFFIX}"
            if ticker in tier_of:
                continue  # already captured at a higher-priority tier — doesn't count against this tier's cap
            tickers.append(ticker)
            tier_of[ticker] = tier
            industry = str(row.get("Industry", "")).strip()
            sector_map[ticker] = industry or "Unknown"
            added_from_source += 1

    if not tickers:
        raise UniverseBuilderError("Every NSE index source failed — refusing to write an empty universe.")

    large_cap = sorted(t for t, tier in tier_of.items() if tier == "large")
    non_large_cap = sorted(t for t, tier in tier_of.items() if tier != "large")

    n_ok = sum(1 for v in sources_status.values() if v.startswith("OK"))
    logger.info(
        f"[UNIVERSE] Built {len(tickers)} tickers from {n_ok}/{len(INDEX_SOURCES)} sources "
        f"({len(large_cap)} large-cap, {len(non_large_cap)} mid/small/thematic)"
    )
    if n_ok < len(INDEX_SOURCES):
        logger.warning(f"[UNIVERSE] Source failures: {sources_status}")

    return {
        "tickers": tickers,
        "sector_map": sector_map,
        "large_cap_tickers": large_cap,
        "non_large_cap_tickers": non_large_cap,
        "sources": sources_status,
        "built_at": datetime.now().isoformat(),
    }
