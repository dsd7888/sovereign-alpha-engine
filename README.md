# 🏛️ Sovereign Alpha Engine

Automated quantitative paper trading engine for Indian equities.

| | |
|---|---|
| **Universe** | Nifty 50 + Midcap 100 (115 tickers) |
| **Capital** | ₹1,00,000 virtual |
| **Cost model** | Groww equity delivery (2025–26 rates) |
| **Runs** | Every weekday at 17:30 IST via GitHub Actions |

## 📊 Daily Report

**[→ View today's report](https://dhruvdakhara.github.io/sovereign-alpha-engine/)**

Updated automatically every weekday evening.
[Report archive](https://dhruvdakhara.github.io/sovereign-alpha-engine/archive.html)

## Manual Run

Trigger anytime without waiting for 17:30:
**Actions → Sovereign Alpha Engine — Daily Run → Run workflow**

## Architecture

L1 DataIngestor → yfinance OHLCV + fundamentals
L2 DataHarmonizer → Altman Z, Beneish M, Screener.in pledge data
L3 FundamentalScorer → Unified Stock Health Score (USHS 0–100)
L4 PortfolioOptimizer → Monte Carlo + Cholesky + CVaR frontier
L5 RiskGuardian → Circuit breakers (VIX + drawdown)
L6 PaperBroker → Virtual execution with exact Groww costs
