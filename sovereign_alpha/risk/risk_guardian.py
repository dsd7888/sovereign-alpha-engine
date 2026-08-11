"""
L5 — Risk Guardian.

Always-on supervisory layer. Its verdicts override whatever the optimiser
wants to do — a circuit breaker firing means the portfolio de-risks
regardless of how attractive the Monte Carlo frontier looked this morning.
"""
from __future__ import annotations

from enum import Enum

import numpy as np

from config import settings
from config.universe import NIFTY_MIDCAP_100_SET
from sovereign_alpha.utils.logger import logger


class Action(Enum):
    HOLD = "HOLD"
    REDUCE_50 = "REDUCE_50"  # liquidate 50% of equity positions
    FULL_EXIT = "FULL_EXIT"  # 100% liquidation to cash
    HEDGE_PUT = "HEDGE_PUT"  # defensive posture — flagged for human review (no options in paper mode)
    HALT = "HALT"  # stop all new entries, flag for review


class RiskGuardian:
    """Circuit breakers, per-position stop losses, and Kelly position sizing."""

    def check_circuit_breaker(self, portfolio_value: float, peak_value: float, vix: float) -> Action:
        """
        Priority order: VIX crisis > VIX high alert > drawdown full-exit >
        (drawdown reduce / VIX elevated) > hold. VIX triggers are checked
        first because a volatility spike can precede a drawdown by hours —
        waiting for the drawdown to also breach would be too late.
        """
        drawdown = (peak_value - portfolio_value) / max(peak_value, 1e-9)

        if vix >= settings.VIX_CRISIS:
            logger.critical(f"[CIRCUIT BREAKER] VIX CRISIS ({vix:.1f} >= {settings.VIX_CRISIS}) -> HALT")
            return Action.HALT

        if vix >= settings.VIX_HIGH_ALERT:
            logger.warning(f"[CIRCUIT BREAKER] VIX HIGH ALERT ({vix:.1f}) -> HEDGE_PUT")
            return Action.HEDGE_PUT

        if drawdown >= settings.DRAWDOWN_FULL_EXIT:
            logger.critical(
                f"[CIRCUIT BREAKER] Drawdown {drawdown:.1%} >= {settings.DRAWDOWN_FULL_EXIT:.0%} -> FULL_EXIT"
            )
            return Action.FULL_EXIT

        if drawdown >= settings.DRAWDOWN_REDUCE_50 or vix >= settings.VIX_ELEVATED:
            logger.warning(f"[CIRCUIT BREAKER] Drawdown {drawdown:.1%} / VIX {vix:.1f} -> REDUCE_50")
            return Action.REDUCE_50

        logger.debug(f"[CIRCUIT BREAKER] HOLD | Drawdown: {drawdown:.1%} | VIX: {vix:.1f}")
        return Action.HOLD

    def check_single_position(self, ticker: str, entry_price: float, current_price: float) -> Action:
        """Per-position stop loss: exit a holding that's down more than the configured threshold."""
        if entry_price <= 0:
            return Action.HOLD
        loss_pct = (entry_price - current_price) / entry_price
        stop = (
            settings.SINGLE_STOCK_STOP_LOSS_MIDCAP
            if ticker in NIFTY_MIDCAP_100_SET
            else settings.SINGLE_STOCK_STOP_LOSS
        )
        if loss_pct >= stop:
            logger.warning(
                f"[STOP-LOSS] {ticker}: {loss_pct:.1%} loss from Rs.{entry_price:.2f} "
                f"(threshold {stop:.0%}) -> FULL_EXIT"
            )
            return Action.FULL_EXIT
        return Action.HOLD

    def check_promoter_pledge(self, ticker: str, new_pct: float, prev_pct: float) -> Action:
        """Exit if promoter pledging exceeds 50% AND jumped more than 10 percentage points."""
        if new_pct > 50 and (new_pct - prev_pct) > 10:
            logger.warning(
                f"[PLEDGE ALERT] {ticker}: pledge surged {prev_pct:.1f}% -> {new_pct:.1f}% -> FULL_EXIT"
            )
            return Action.FULL_EXIT
        return Action.HOLD

    def compute_position_size(
        self,
        ticker: str,
        total_capital: float,
        mu: float,
        sigma: float,
        avg_daily_volume_inr: float = 1e8,
    ) -> dict:
        """
        Half-Kelly position sizing: f = 0.5 * (mu - rf) / sigma^2, clipped to
        [0, 25%] of capital, then trimmed further if the resulting order's
        market-impact cost would exceed 0.5%.
        """
        rf = settings.RISK_FREE_RATE
        sigma = max(sigma, 0.01)  # guard against division by ~0 on a near-zero-vol series

        f_kelly = (mu - rf) / (sigma ** 2)
        f_actual = min(max(f_kelly * 0.5, 0.0), 0.25)
        position_inr = f_actual * total_capital

        from sovereign_alpha.optimization.transaction_costs import GrowwCostModel

        daily_vol = sigma / np.sqrt(settings.TRADING_DAYS)
        ic = GrowwCostModel.compute_impact_cost(position_inr, avg_daily_volume_inr, daily_vol)
        if ic > 0.005:
            reduction_factor = 0.005 / ic
            position_inr *= reduction_factor
            f_actual *= reduction_factor
            logger.debug(f"[KELLY] {ticker}: impact cost {ic:.2%} too high, reduced to {f_actual:.2%}")

        return {
            "ticker": ticker,
            "kelly_raw": f_kelly,
            "position_fraction": f_actual,
            "position_inr": position_inr,
            "impact_cost_pct": ic * 100,
        }

    def generate_risk_report(
        self,
        holdings: dict[str, dict],
        current_value: float,
        peak_value: float,
        vix: float,
    ) -> str:
        """Human-readable risk summary for the daily report/log. ``holdings`` is ``PaperBroker.state["holdings"]``."""
        lines = ["=" * 50, f"RISK REPORT | VIX: {vix:.1f}", "=" * 50]

        drawdown = (peak_value - current_value) / max(peak_value, 1e-9)
        overall = self.check_circuit_breaker(current_value, peak_value, vix)
        lines.append(f"Portfolio Value: Rs.{current_value:,.2f}")
        lines.append(f"Drawdown: {drawdown:.2%}")
        lines.append(f"Overall Action: {overall.value}")
        lines.append("")

        for ticker, holding in holdings.items():
            entry = holding.get("avg_cost", 0)
            curr = holding.get("last_price", entry)
            qty = holding.get("qty", 0)
            if qty == 0 or entry == 0:
                continue
            action = self.check_single_position(ticker, entry, curr)
            pnl_pct = (curr - entry) / entry * 100
            lines.append(f"  {ticker:20s}: {qty} shares @ Rs.{curr:.2f} ({pnl_pct:+.1f}%)  [{action.value}]")

        return "\n".join(lines)


if __name__ == "__main__":
    rg = RiskGuardian()
    print("Scenario 1 — VIX=27, drawdown=5%:", rg.check_circuit_breaker(95000, 100000, 27.0).value)
    print("Scenario 2 — VIX=15, drawdown=10%:", rg.check_circuit_breaker(90000, 100000, 15.0).value)
    print("Scenario 3 — VIX=36, drawdown=3%:", rg.check_circuit_breaker(97000, 100000, 36.0).value)
    print("Scenario 4 — Normal market:", rg.check_circuit_breaker(99000, 100000, 17.0).value)
    print("RiskGuardian OK")
