"""
Backtest / walk-forward validation entry point.

Usage:
  python run_backtest.py                                  # technical mode, max available history
  python run_backtest.py --mode full                       # fundamentals-included, annual resolution
  python run_backtest.py --mode both                       # run both modes back to back
  python run_backtest.py --years 5                          # cap the lookback window
  python run_backtest.py --tickers RELIANCE.NS TCS.NS       # override the universe (default: PILOT_UNIVERSE)

Runtime: a full ~10-year daily-rebalance "technical" run over the full
115-ticker universe runs a 5,000-simulation Monte Carlo optimisation on
every rebalance day — expect this to take a while (tens of minutes to a
few hours depending on --rebalance-frequency-days and --n-simulations).
This is a batch job, not something to run on every commit; see
.github/workflows/backtest.yml for the scheduled (weekly) version.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta

from config import settings
from config.universe import PILOT_UNIVERSE
from sovereign_alpha.backtesting.engine import BacktestEngine
from sovereign_alpha.backtesting.report_generator import generate_backtest_report
from sovereign_alpha.backtesting.walk_forward import WalkForwardValidator
from sovereign_alpha.utils.logger import logger


def _run_one_mode(
    mode: str, tickers: list[str], start: str, end: str,
    initial_capital: float, rebalance_frequency_days: int, n_simulations: int | None, fold_months: int,
) -> dict:
    logger.info("=" * 60)
    logger.info(f"BACKTEST — mode={mode} | {start} -> {end} | {len(tickers)} tickers")
    logger.info("=" * 60)
    t0 = time.time()

    engine = BacktestEngine(
        mode=mode, initial_capital=initial_capital,
        rebalance_frequency_days=rebalance_frequency_days, n_simulations=n_simulations,
    )
    validator = WalkForwardValidator(engine=engine, fold_months=fold_months)
    result = validator.run(tickers, start=start, end=end)

    elapsed = time.time() - t0
    logger.info(f"[BACKTEST] {mode} complete in {elapsed / 60:.1f} min")
    logger.info(f"  CAGR:          {result.overall.get('cagr_pct', float('nan')):.1f}%")
    logger.info(f"  Sharpe:        {result.overall.get('sharpe_ratio', float('nan')):.2f}")
    logger.info(f"  Sortino:       {result.overall.get('sortino_ratio', float('nan')):.2f}")
    logger.info(f"  Max Drawdown:  {result.overall.get('max_drawdown_pct', float('nan')):.1f}%")
    logger.info(f"  Win Rate:      {result.overall.get('win_rate_pct', float('nan')):.0f}%")
    logger.info(f"  Trades:        {result.overall.get('n_trades', 0)}")

    report_path = generate_backtest_report(result.overall, result.folds, result.backtest.nav, mode)

    payload = json.dumps({
        "mode": mode, "start": start, "end": end, "tickers": tickers,
        "elapsed_seconds": elapsed, "overall": result.overall, "folds": result.folds,
    }, indent=2, default=str)

    out_dir = settings.OUTPUT_DIR / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"backtest_{mode}_{datetime.now().strftime('%Y%m%d')}.json"
    json_path.write_text(payload, encoding="utf-8")
    logger.info(f"[BACKTEST] Results saved: {json_path}")
    logger.info(f"[BACKTEST] Report saved: {report_path}")

    # A second, git-tracked copy (settings.OUTPUT_DIR is gitignored — this
    # one isn't, see .gitignore) — a small, versioned record of how each
    # scheduled run assessed the strategy, independent of the ephemeral
    # report/chart artifacts.
    history_dir = settings.BASE_DIR / "data" / "backtest_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    (history_dir / f"backtest_{mode}_{datetime.now().strftime('%Y%m%d')}.json").write_text(
        payload, encoding="utf-8"
    )

    return result.overall


def main() -> None:
    parser = argparse.ArgumentParser(description="Sovereign Alpha Engine — Backtest")
    parser.add_argument("--mode", choices=["technical", "full", "both"], default="technical")
    parser.add_argument("--years", type=int, default=settings.BACKTEST_MAX_LOOKBACK_YEARS,
                         help="Lookback window in years (bounded by whatever yfinance actually returns)")
    parser.add_argument("--tickers", nargs="+", default=None, help="Override the universe (default: PILOT_UNIVERSE)")
    parser.add_argument("--initial-capital", type=float, default=settings.INITIAL_CAPITAL_INR)
    parser.add_argument("--rebalance-frequency-days", type=int, default=settings.BACKTEST_REBALANCE_FREQUENCY_DAYS)
    parser.add_argument("--n-simulations", type=int, default=None,
                         help="Monte Carlo draws per rebalance (default: settings.N_SIMULATIONS)")
    parser.add_argument("--fold-months", type=int, default=settings.BACKTEST_WALK_FORWARD_FOLD_MONTHS)
    args = parser.parse_args()

    tickers = args.tickers or PILOT_UNIVERSE
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=365 * args.years)).strftime("%Y-%m-%d")

    modes = ["technical", "full"] if args.mode == "both" else [args.mode]
    for mode in modes:
        try:
            _run_one_mode(
                mode, tickers, start, end, args.initial_capital,
                args.rebalance_frequency_days, args.n_simulations, args.fold_months,
            )
        except Exception as e:
            logger.error(f"[BACKTEST] mode={mode} failed: {e}")
            if len(modes) == 1:
                raise


if __name__ == "__main__":
    main()
