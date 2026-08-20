"""Ticker universe and sector mapping. yfinance requires the ``.NS`` suffix for NSE symbols."""
from __future__ import annotations

NIFTY_50: list[str] = [
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

NIFTY_MIDCAP_100: list[str] = [
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

NIFTY_MIDCAP_100_SET: frozenset[str] = frozenset(NIFTY_MIDCAP_100)

# Large-cap liquidity + midcap alpha sleeve; Nifty 500 once the expanded pilot validates.
PILOT_UNIVERSE: list[str] = list(dict.fromkeys(NIFTY_50 + NIFTY_MIDCAP_100))

# Sector mapping — used to normalise P/E within its peer group rather than
# across the whole (highly heterogeneous) index.
SECTOR_MAP: dict[str, str] = {
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


def sector_of(ticker: str) -> str:
    """Sector for a ticker, defaulting to ``"Unknown"`` if unmapped."""
    return SECTOR_MAP.get(ticker, "Unknown")
