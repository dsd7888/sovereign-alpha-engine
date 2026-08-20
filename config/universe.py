"""
Ticker universe and sector mapping. yfinance requires the ``.NS`` suffix
for NSE symbols.

The real universe/sector data lives in ``data/universe/universe_data.json``,
built periodically by ``sovereign_alpha.ingestion.universe_builder`` from
real NSE index constituent files (see ``reconstitute_universe.py`` /
``.github/workflows/universe_reconstitution.yml`` — monthly). This module
loads that file at import time.

Why not just hardcode the list here, as before: a hand-typed list is
permanently behind the market by construction — it can never contain a
stock that IPO'd or got promoted into an index after the list was last
edited, and nobody's job is to keep it current. It also silently mislabels
itself over time — the list this replaced was named ``NIFTY_MIDCAP_100``
but only actually contained 65 tickers, not 100.

The constants below (``LARGE_CAP_UNIVERSE``, ``NON_LARGECAP_UNIVERSE``,
``NON_LARGECAP_SET``, ``PILOT_UNIVERSE``, ``SECTOR_MAP``, ``sector_of``)
are the same public API every other module already imports — this file
changed where the data comes from, not the interface, so nothing
downstream needed to change.

If ``universe_data.json`` doesn't exist yet (first run before any
reconstitution) or is corrupt, this falls back to a small bundled seed
list rather than failing import entirely — a wrong-but-working universe
beats an engine that can't start.
"""
from __future__ import annotations

import json

from config import settings

_UNIVERSE_DATA_PATH = settings.BASE_DIR / "data" / "universe" / "universe_data.json"

# Bundled fallback seed — used only if universe_data.json is missing or
# corrupt. Deliberately small and static; the real, current universe comes
# from the dynamic fetch above once reconstitution has run at least once.
_FALLBACK_NIFTY_50: list[str] = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "BAJFINANCE.NS",
    "HCLTECH.NS", "WIPRO.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "TITAN.NS",
    "SUNPHARMA.NS", "TATAMOTORS.NS", "M&M.NS", "POWERGRID.NS", "NTPC.NS",
    "BAJAJFINSV.NS", "TECHM.NS", "ONGC.NS", "JSWSTEEL.NS", "TATASTEEL.NS",
    "HDFCLIFE.NS", "INDUSINDBK.NS", "GRASIM.NS", "DRREDDY.NS", "CIPLA.NS",
    "ADANIPORTS.NS", "COALINDIA.NS", "EICHERMOT.NS", "APOLLOHOSP.NS",
    "BRITANNIA.NS", "DIVISLAB.NS", "HEROMOTOCO.NS", "BPCL.NS", "TATACONSUM.NS",
    "SBILIFE.NS", "HINDALCO.NS", "SHRIRAMFIN.NS", "ADANIENT.NS", "BEL.NS",
    "BAJAJ-AUTO.NS",
]

_FALLBACK_NON_LARGECAP: list[str] = [
    "ABCAPITAL.NS", "ABFRL.NS", "AUROPHARMA.NS", "BALKRISIND.NS", "BANDHANBNK.NS",
    "BANKBARODA.NS", "BERGEPAINT.NS", "BHEL.NS", "BOSCHLTD.NS", "CANBK.NS",
    "CHOLAFIN.NS", "COLPAL.NS", "CONCOR.NS", "COFORGE.NS", "CROMPTON.NS",
    "DABUR.NS", "DALBHARAT.NS", "DEEPAKNTR.NS", "DLF.NS", "FEDERALBNK.NS",
    "GLENMARK.NS", "GODREJCP.NS", "GODREJPROP.NS", "GUJGASLTD.NS", "HAL.NS",
    "HAVELLS.NS", "HFCL.NS", "ICICIGI.NS", "ICICIPRULI.NS", "IDFCFIRSTB.NS",
    "INDHOTEL.NS", "INDUSTOWER.NS", "IOC.NS", "IRCTC.NS", "JUBLFOOD.NS",
    "KAJARIACER.NS", "LICHSGFIN.NS", "LUPIN.NS", "MCDOWELL-N.NS", "MFSL.NS",
    "MPHASIS.NS", "MRF.NS", "NAUKRI.NS", "NMDC.NS", "OBEROIRLTY.NS",
    "OFSS.NS", "PAGEIND.NS", "PERSISTENT.NS", "PETRONET.NS", "PIDILITIND.NS",
    "PIIND.NS", "PNB.NS", "POLYCAB.NS", "RECLTD.NS", "SAIL.NS",
    "SIEMENS.NS", "SRF.NS", "SUPREMEIND.NS", "TORNTPHARM.NS", "TRENT.NS",
    "TVSMOTOR.NS", "UBL.NS", "VEDL.NS", "VOLTAS.NS", "ZOMATO.NS",
]

_FALLBACK_SECTOR_MAP: dict[str, str] = {
    "RELIANCE.NS": "Energy", "ONGC.NS": "Energy", "BPCL.NS": "Energy",
    "TCS.NS": "IT", "INFY.NS": "IT", "HCLTECH.NS": "IT",
    "WIPRO.NS": "IT", "TECHM.NS": "IT",
    "HDFCBANK.NS": "Banking", "ICICIBANK.NS": "Banking", "SBIN.NS": "Banking",
    "KOTAKBANK.NS": "Banking", "AXISBANK.NS": "Banking", "INDUSINDBK.NS": "Banking",
    "HINDUNILVR.NS": "FMCG", "ITC.NS": "FMCG", "NESTLEIND.NS": "FMCG",
    "BRITANNIA.NS": "FMCG", "TATACONSUM.NS": "FMCG",
    "MARUTI.NS": "Auto", "TATAMOTORS.NS": "Auto", "BAJAJ-AUTO.NS": "Auto",
    "EICHERMOT.NS": "Auto", "HEROMOTOCO.NS": "Auto", "M&M.NS": "Auto",
    "SUNPHARMA.NS": "Pharma", "DRREDDY.NS": "Pharma", "CIPLA.NS": "Pharma",
    "DIVISLAB.NS": "Pharma", "APOLLOHOSP.NS": "Pharma",
    "LT.NS": "Infra", "POWERGRID.NS": "Infra", "NTPC.NS": "Infra",
    "ADANIPORTS.NS": "Infra", "BEL.NS": "Infra", "COALINDIA.NS": "Infra",
    "BAJFINANCE.NS": "NBFC", "BAJAJFINSV.NS": "NBFC", "SHRIRAMFIN.NS": "NBFC",
    "HDFCLIFE.NS": "Insurance", "SBILIFE.NS": "Insurance",
    "ASIANPAINT.NS": "Paints",
    "TITAN.NS": "Consumer", "ULTRACEMCO.NS": "Cement",
    "JSWSTEEL.NS": "Metals", "TATASTEEL.NS": "Metals", "HINDALCO.NS": "Metals",
    "ADANIENT.NS": "Diversified", "GRASIM.NS": "Diversified",
    "BANDHANBNK.NS": "Banking", "BANKBARODA.NS": "Banking", "CANBK.NS": "Banking",
    "FEDERALBNK.NS": "Banking", "IDFCFIRSTB.NS": "Banking",
    "ICICIGI.NS": "Insurance", "ICICIPRULI.NS": "Insurance", "MFSL.NS": "Insurance",
    "ABCAPITAL.NS": "NBFC", "CHOLAFIN.NS": "NBFC", "LICHSGFIN.NS": "NBFC",
}


def _load_universe_data() -> dict:
    try:
        raw = json.loads(_UNIVERSE_DATA_PATH.read_text(encoding="utf-8"))
        if not raw.get("tickers") or not raw.get("large_cap_tickers"):
            raise ValueError("universe_data.json missing required keys")
        return raw
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return {
            "tickers": list(dict.fromkeys(_FALLBACK_NIFTY_50 + _FALLBACK_NON_LARGECAP)),
            "large_cap_tickers": _FALLBACK_NIFTY_50,
            "non_large_cap_tickers": _FALLBACK_NON_LARGECAP,
            "sector_map": _FALLBACK_SECTOR_MAP,
        }


_DATA = _load_universe_data()

# Nifty 50 + Nifty Next 50 — the large-cap liquidity anchor. Named
# generically because, unlike the constant this replaced, it's no longer
# only Nifty 50's own membership.
LARGE_CAP_UNIVERSE: list[str] = _DATA["large_cap_tickers"]
# Everything in PILOT_UNIVERSE that isn't in the large-cap tier — mid,
# small, and thematic (e.g. defence) constituents. Named generically
# because, unlike the old NIFTY_MIDCAP_100 list, this is no longer just
# one index's membership.
NON_LARGECAP_UNIVERSE: list[str] = _DATA["non_large_cap_tickers"]
NON_LARGECAP_SET: frozenset[str] = frozenset(NON_LARGECAP_UNIVERSE)

PILOT_UNIVERSE: list[str] = list(dict.fromkeys(_DATA["tickers"]))

# Sector mapping — used to normalise P/E within its peer group, and to
# gate Altman Z off financials (settings.ALTMAN_Z_EXCLUDED_SECTORS).
SECTOR_MAP: dict[str, str] = _DATA["sector_map"]


def sector_of(ticker: str) -> str:
    """Sector for a ticker, defaulting to ``"Unknown"`` if unmapped."""
    return SECTOR_MAP.get(ticker, "Unknown")
