"""
Central configuration. All tunable constants live here — never hardcode
a magic number inside a `sovereign_alpha` module; import it from here instead.
"""
from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ── Data paths ────────────────────────────────────────────────────────────
RAW_OHLCV_DIR = BASE_DIR / "data" / "raw" / "ohlcv"
RAW_FUND_DIR = BASE_DIR / "data" / "raw" / "fundamentals"
RAW_VIX_DIR = BASE_DIR / "data" / "raw" / "vix"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"
PAPER_PORTFOLIO_DIR = BASE_DIR / "data" / "paper_portfolio"
REPORTS_DIR = BASE_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"

for _d in (
    RAW_OHLCV_DIR, RAW_FUND_DIR, RAW_VIX_DIR, PROCESSED_DIR,
    OUTPUT_DIR, PAPER_PORTFOLIO_DIR, REPORTS_DIR, LOG_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

# ── Market ────────────────────────────────────────────────────────────────
MARKET_SUFFIX = ".NS"
TRADING_DAYS = 252
RISK_FREE_RATE = 0.068  # 10-yr GSec yield — update quarterly
BENCHMARK_TICKER = "^NSEI"  # Nifty 50 index
LOOKBACK_YEARS = 3
MIN_OHLCV_HISTORY_DAYS = 100  # below this a ticker is excluded from optimisation

# ── Cache durations ───────────────────────────────────────────────────────
OHLCV_CACHE_HOURS = 18
FUNDAMENTALS_CACHE_DAYS = 7
VIX_CACHE_HOURS = 4

# ── USHS (Unified Stock Health Score) weights — must sum to 1.0 ───────────
USHS_WEIGHTS = {
    "pe_score": 0.12,
    "pb_score": 0.08,
    "roe_score": 0.15,
    "fcf_yield_score": 0.15,
    "debt_equity_score": 0.10,
    "promoter_pledge_score": 0.10,
    "altman_z_score": 0.12,
    "beneish_m_score": 0.10,
    "sentiment_score": 0.08,
}
assert abs(sum(USHS_WEIGHTS.values()) - 1.0) < 1e-9, "USHS_WEIGHTS must sum to 1.0"

# ── Groww brokerage cost model — Equity Delivery, post 21-Jun-2025 rates ──
# BUY:  brokerage = clip(0.1% of value, min=5, max=20)
#       STT 0.1% + exchange 0.00297% + SEBI ₹10/crore + stamp duty 0.015% (buy only)
#       GST 18% on (brokerage + exchange + SEBI)
# SELL: same brokerage/STT/exchange/SEBI, no stamp duty, + DP charge ₹20/scrip
#       GST 18% on (brokerage + exchange + SEBI + DP)
# Round-trip on a ₹1,00,000 delivery trade ≈ ₹295 (0.295%).
GROWW_BROKERAGE_MAX = 20.00
GROWW_BROKERAGE_PCT = 0.001
GROWW_BROKERAGE_MIN = 5.00
STT_RATE = 0.001
EXCHANGE_TC_RATE = 0.0000297
SEBI_RATE = 1e-7
STAMP_DUTY_RATE = 0.00015
GST_RATE = 0.18
DP_CHARGE_PER_SELL = 20.00

# ── Impact cost model (for position sizing) ───────────────────────────────
IMPACT_COST_ALPHA = 0.10
IMPACT_COST_BETA = 0.60
IMPACT_COST_CAP = 0.02  # anything above this means the order is too large
MIN_AVG_DAILY_VOLUME_INR = 5_00_00_000  # Rs.5 crore minimum daily turnover

# ── Paper trading ─────────────────────────────────────────────────────────
INITIAL_CAPITAL_INR = 100_000.0
PAPER_STATE_FILE = PAPER_PORTFOLIO_DIR / "portfolio_state.json"
MAX_STOCKS_IN_PORTFOLIO = 12
MIN_USHS_THRESHOLD = 60.0
MIN_POSITION_VALUE_INR = 2_000.0
REBALANCE_SELL_BAND = 0.05   # trim if overweight by more than this
REBALANCE_BUY_BAND = 0.03    # top up if underweight by more than this

# ── Circuit breaker thresholds ────────────────────────────────────────────
DRAWDOWN_REDUCE_50 = 0.08
DRAWDOWN_FULL_EXIT = 0.15
VIX_ELEVATED = 20.0
VIX_HIGH_ALERT = 25.0
VIX_CRISIS = 35.0
SINGLE_STOCK_STOP_LOSS = 0.12
SINGLE_STOCK_STOP_LOSS_MIDCAP = 0.15  # allow slightly more volatility than large cap
VIX_DEFAULT_FALLBACK = 18.0  # neutral value used only if all VIX sources fail

# ── Monte Carlo ────────────────────────────────────────────────────────────
N_SIMULATIONS = 5_000
CVAR_CONFIDENCE = 0.95
MC_RANDOM_SEED = 42
MU_SHRINKAGE = True  # James-Stein shrinkage on drift estimates — set False to disable for debugging

# ── Screener.in scraping ───────────────────────────────────────────────────
SCREENER_BASE_URL = "https://www.screener.in/company/"
SCREENER_DELAY_SEC = 3.0  # non-negotiable — avoids IP ban
SCREENER_TIMEOUT_SEC = 15

# ── Altman Z''-Score applicability ─────────────────────────────────────────
# Banks/NBFCs/insurers don't have current assets/liabilities in the
# traditional sense, don't report EBIT, and have fundamentally different
# capital structures — Z'' is meaningless for them.
ALTMAN_Z_EXCLUDED_SECTORS = {
    "Banking", "NBFC", "Insurance", "Financial Services",
}
