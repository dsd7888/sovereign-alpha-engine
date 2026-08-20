"""
Master orchestrator — the single entry point for every run, manual or
cron-triggered.

Usage:
  python run_engine.py                  # full run: score + optimise + trade
  python run_engine.py --mode score     # score stocks only, no optimisation/trading
  python run_engine.py --open-report    # open the HTML report in a browser afterwards
  python run_engine.py --force          # run even if today is a weekend (for testing)
"""
from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime

from config import settings
from config.universe import PILOT_UNIVERSE
from sovereign_alpha.utils.helpers import is_market_day
from sovereign_alpha.utils.logger import logger


def run_full_pipeline(open_report: bool = False, mode: str = "full", force: bool = False) -> None:
    start_time = time.time()
    logger.info("=" * 60)
    logger.info(f"SOVEREIGN ALPHA ENGINE — {datetime.now().strftime('%A %d %B %Y %H:%M IST')}")
    logger.info(f"Mode: {mode} | Universe: {len(PILOT_UNIVERSE)} tickers")
    logger.info("=" * 60)

    if not force and not is_market_day():
        logger.warning("Today is a weekend — skipping (markets closed). Pass --force to override.")
        return

    # ── L1 + L2: ingestion & harmonisation ──────────────────────────────
    logger.info("[STEP 1/6] Data ingestion + harmonisation...")
    from sovereign_alpha.ingestion.data_harmonizer import DataHarmonizer

    harmonizer = DataHarmonizer()
    try:
        harmonized_df = harmonizer.harmonize(PILOT_UNIVERSE)
    except Exception as e:
        logger.error(f"[STEP 1 FAILED] {e} — falling back to last cached harmonized data...")
        try:
            harmonized_df = harmonizer.load_latest()
        except FileNotFoundError as e2:
            logger.critical(f"No data available at all: {e2}. Aborting run.")
            return

    # Momentum + low-vol are computed from price history alone (no Screener
    # dependency), so a failure here must never abort the run — it just
    # means those two factors fall back to neutral (50) for this run, same
    # as any other USHS input with missing data.
    from sovereign_alpha.ingestion.data_ingestor import DataIngestor
    from sovereign_alpha.ingestion.technical_features import compute_technical_features

    try:
        technical_ohlcv = DataIngestor().fetch_ohlcv(PILOT_UNIVERSE)
        technical_df = compute_technical_features(technical_ohlcv, PILOT_UNIVERSE)
        harmonized_df = harmonized_df.join(technical_df, how="left")
    except Exception as e:
        logger.warning(f"[TECHNICAL] Momentum/low-vol unavailable this run: {e}")

    # ── L3: scoring ──────────────────────────────────────────────────────
    logger.info("[STEP 2/6] Computing USHS scores...")
    from sovereign_alpha.scoring.fundamental_scorer import FundamentalScorer

    scorer = FundamentalScorer()
    try:
        scored_df = scorer.compute_ushs(harmonized_df)
        eligible_tickers = scorer.get_eligible_tickers(scored_df)
    except Exception as e:
        logger.error(f"[STEP 2 FAILED] {e}")
        return

    if mode == "score":
        logger.info("[SCORE MODE] Done — skipping optimisation and trading.")
        return

    if len(eligible_tickers) < 3:
        logger.warning(f"Only {len(eligible_tickers)} eligible tickers — using top 10 by USHS instead.")
        eligible_tickers = scored_df.sort_values("ushs", ascending=False).head(10).index.tolist()
    eligible_tickers = eligible_tickers[: settings.MAX_STOCKS_IN_PORTFOLIO]

    # ── L4: portfolio optimisation ────────────────────────────────────────
    logger.info(f"[STEP 3/6] Monte Carlo optimisation ({len(eligible_tickers)} stocks)...")
    from sovereign_alpha.optimization.portfolio_optimizer import PortfolioOptimizer

    optimizer = PortfolioOptimizer()
    frontier_chart_path = None
    try:
        returns_df = optimizer.load_returns(eligible_tickers)
        cov, mu = optimizer.compute_covariance_matrix(returns_df)
        sim_df = optimizer.cholesky_monte_carlo(mu, cov, list(returns_df.columns))
        optimal = optimizer.find_optimal_portfolios(sim_df)
        target_weights = optimal["max_sharpe"]["weights"]
        frontier_chart_path = optimizer.plot_frontier(optimal["all_sims"], optimal["max_sharpe"], optimal["min_cvar"])
        logger.info(
            f"[OPT] Max Sharpe (net): {optimal['max_sharpe']['sharpe_net']:.2f} | "
            f"Expected return: {optimal['max_sharpe']['exp_return_net']:.1%}"
        )
    except Exception as e:
        logger.error(f"[STEP 3 FAILED] {e}. Falling back to equal-weight allocation.")
        n = len(eligible_tickers)
        target_weights = {t: 1.0 / n for t in eligible_tickers}

    # ── Current prices + VIX ─────────────────────────────────────────────
    # Prices must cover target_weights *and* every currently-held ticker —
    # not just target_weights. A holding that fell out of today's target
    # (target_w == 0) is exactly what paper_broker.rebalance() needs to
    # exit, but it can only do so if current_prices actually has a price
    # for it; otherwise that position is stuck at its stale last_price
    # forever, un-marked and un-sellable.
    logger.info("[STEP 4/6] Fetching current prices + India VIX...")
    from sovereign_alpha.execution.paper_broker import PaperBroker
    from sovereign_alpha.ingestion.data_ingestor import DataIngestor

    broker = PaperBroker()
    held_tickers = {t for t, h in broker.state["holdings"].items() if h.get("qty", 0) > 0}
    price_universe = set(target_weights) | held_tickers

    ingestor = DataIngestor()
    try:
        ohlcv = ingestor.fetch_ohlcv(list(price_universe))
    except Exception as e:
        logger.critical(f"[STEP 4 FAILED] Could not fetch current prices: {e}. Aborting run.")
        return

    current_prices = {}
    for ticker in price_universe:
        subset = ohlcv.loc[ohlcv["ticker"] == ticker, "Adj_Close"].dropna()
        if len(subset) > 0:
            current_prices[ticker] = float(subset.iloc[-1])
    vix = ingestor.fetch_india_vix()

    # ── L5: risk check ────────────────────────────────────────────────────
    logger.info(f"[STEP 5/6] Risk Guardian check (VIX: {vix:.1f})...")
    import pandas as pd

    from sovereign_alpha.risk.risk_guardian import Action, RiskGuardian
    from sovereign_alpha.utils.validators import is_valid_price

    portfolio_value = broker.update_prices(current_prices)
    peak_value = broker.state["peak_value"]

    risk_guard = RiskGuardian()
    circuit_action = risk_guard.check_circuit_breaker(portfolio_value, peak_value, vix)
    logger.info(f"[RISK] Circuit action: {circuit_action.value}")

    # Per-position overrides: a per-stock stop-loss or a promoter-pledge
    # surge exits that one holding immediately, ahead of and independent
    # from today's target weights, so the freed cash is available for the
    # rebalance below. These were previously computed for the report only
    # and never actually executed a sell.
    for ticker in list(held_tickers):
        holding = broker.state["holdings"].get(ticker)
        if not holding or holding.get("qty", 0) <= 0:
            continue
        price = current_prices.get(ticker)
        if not is_valid_price(price):
            continue

        action = risk_guard.check_single_position(ticker, holding.get("avg_cost", 0.0), price)
        if action == Action.HOLD and ticker in scored_df.index:
            new_pledge = scored_df.loc[ticker, "promoter_pledge_pct"]
            prev_pledge = holding.get("promoter_pledge_pct")
            if pd.notna(new_pledge):
                if prev_pledge is not None:
                    action = risk_guard.check_promoter_pledge(ticker, float(new_pledge), prev_pledge)
                holding["promoter_pledge_pct"] = float(new_pledge)

        if action == Action.FULL_EXIT:
            broker.sell_all(ticker, price)

    # ── L6: execute rebalance ─────────────────────────────────────────────
    logger.info("[STEP 6/6] Rebalancing paper portfolio...")
    trades = broker.rebalance(target_weights, current_prices, circuit_action)
    final_value = broker.update_prices(current_prices)

    # ── Reporting ─────────────────────────────────────────────────────────
    summary = broker.get_summary()
    logger.info("=" * 50)
    logger.info("PORTFOLIO SUMMARY")
    logger.info(f"  Current Value:     Rs.{summary['current_value']:,.2f}")
    logger.info(f"  Total P&L:         Rs.{summary['total_pnl']:+,.2f} ({summary['total_pnl_pct']:+.2f}%)")
    logger.info(f"  Annualised Return: {summary['annualised_return']:.1f}%")
    logger.info(f"  Sharpe Ratio:      {summary['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown:      {summary['max_drawdown_pct']:.1f}%")
    logger.info(f"  Total Costs Paid:  Rs.{summary['total_costs_paid']:,.2f}")
    logger.info(f"  Trades today:      {len(trades)}")
    logger.info("=" * 50)

    risk_report = risk_guard.generate_risk_report(broker.state["holdings"], final_value, broker.state["peak_value"], vix)
    logger.info("\n" + risk_report)

    try:
        from sovereign_alpha.utils.report_generator import generate_daily_report

        report_path = generate_daily_report(
            summary=summary, scored_df=scored_df, portfolio_state=broker.state,
            vix=vix, frontier_chart_path=frontier_chart_path,
        )
        if open_report:
            subprocess.run(["open", str(report_path)], check=False)
    except Exception as e:
        logger.error(f"[REPORT] Failed: {e}")

    logger.info(f"\nEngine run complete in {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sovereign Alpha Engine")
    parser.add_argument("--mode", choices=["full", "score"], default="full",
                         help="'score' = score stocks only | 'full' = optimise + trade")
    parser.add_argument("--open-report", action="store_true", help="Open the HTML report after the run")
    parser.add_argument("--force", action="store_true", help="Run even on a non-market day (testing)")
    args = parser.parse_args()
    run_full_pipeline(open_report=args.open_report, mode=args.mode, force=args.force)
