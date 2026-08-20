# 🏛️ Sovereign Alpha Engine

Automated quantitative paper trading engine for Indian equities.

| | |
|---|---|
| **Universe** | Nifty 50 + Midcap 100 (115 tickers) |
| **Capital** | ₹1,00,000 virtual |
| **Cost model** | Groww equity delivery (2025–26 rates) |
| **Runs** | Every weekday at 17:33 IST via GitHub Actions |

## 📊 Daily Report

**[→ View today's report](https://dsd7888.github.io/sovereign-alpha-engine/)**

Updated automatically every weekday evening.
[Report archive](https://dsd7888.github.io/sovereign-alpha-engine/archive.html)

## Manual Run

Trigger anytime without waiting for 17:33:
**Actions → Sovereign Alpha Engine — Daily Run → Run workflow**

## Architecture

L1 DataIngestor → yfinance OHLCV + fundamentals
L2 DataHarmonizer → Altman Z, Beneish M, Screener.in pledge data
L3 FundamentalScorer → Unified Stock Health Score (USHS 0–100), incl. momentum + low-vol
L4 PortfolioOptimizer → Ledoit-Wolf + James-Stein shrinkage, Monte Carlo + Cholesky + CVaR frontier
L5 RiskGuardian → Circuit breakers (VIX + drawdown), per-position stop-loss
L6 PaperBroker → Virtual execution with exact Groww costs

## 📈 Backtesting & Walk-Forward Validation

**[→ Technical (momentum/low-vol) backtest](https://dsd7888.github.io/sovereign-alpha-engine/backtest/technical.html)** — point-in-time-safe over the full ~10yr price history.
**[→ Full-strategy backtest incl. fundamentals](https://dsd7888.github.io/sovereign-alpha-engine/backtest/full.html)** — annual resolution, ~5yr depth (limited by real fundamentals history availability, not a cache setting — see `sovereign_alpha/backtesting/point_in_time.py`).

Runs weekly via `.github/workflows/backtest.yml` (Sunday), reusing the exact same
`FundamentalScorer`/`PortfolioOptimizer`/`PaperBroker`/`RiskGuardian` modules the daily
engine uses — a backtest and a live run can never silently diverge on what "the strategy"
actually does. Results accumulate in `data/backtest_history/`.

```
python run_backtest.py                          # technical mode, max available history
python run_backtest.py --mode full               # fundamentals-included, annual resolution
python run_backtest.py --mode both --years 5      # both, capped lookback
```

Runtime note: a full-universe run does one Monte Carlo optimisation per rebalance day —
budget tens of minutes to a few hours depending on `--rebalance-frequency-days` and
`--n-simulations`. This is a batch job, not something to run on every commit.
