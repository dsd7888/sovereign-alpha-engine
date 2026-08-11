"""
Virtual execution engine — simulates Groww equity-delivery paper trades.

Tracks positions, cash, and P&L with the exact same cost model used by the
optimiser (``GrowwCostModel``), so the numbers the optimiser expects are the
numbers actually charged. State persists atomically to
``data/paper_portfolio/portfolio_state.json`` (see ``helpers.save_json``).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import settings
from sovereign_alpha.optimization.transaction_costs import GrowwCostModel
from sovereign_alpha.risk.risk_guardian import Action
from sovereign_alpha.utils.helpers import load_json, save_json, today_str
from sovereign_alpha.utils.logger import logger
from sovereign_alpha.utils.validators import is_valid_price


class PaperBroker:
    """
    State schema (``portfolio_state.json``):

    ```
    {
      "initial_capital": 100000.0,
      "cash": 85000.0,
      "peak_value": 103000.0,
      "holdings": {
        "RELIANCE.NS": {"qty": 2, "avg_cost": 2450.0, "last_price": 2480.0, "entry_date": "20260720"}
      },
      "trade_log": [{...}],
      "daily_nav": {"20260720": 100000.0, "20260721": 101200.0},
      "last_updated": "20260724"
    }
    ```
    """

    def __init__(self, state_path: Path | None = None) -> None:
        self.cost_model = GrowwCostModel()
        self.state_path = state_path or settings.PAPER_STATE_FILE
        self.state = self._load_or_init_state()

    # ── State management ────────────────────────────────────────────────

    def _load_or_init_state(self) -> dict:
        state = load_json(self.state_path)
        if not state or "initial_capital" not in state:
            logger.info(f"[PAPER] Initialising new portfolio with Rs.{settings.INITIAL_CAPITAL_INR:,.0f}")
            state = {
                "initial_capital": settings.INITIAL_CAPITAL_INR,
                "cash": settings.INITIAL_CAPITAL_INR,
                "peak_value": settings.INITIAL_CAPITAL_INR,
                "holdings": {},
                "trade_log": [],
                "daily_nav": {},
                "last_updated": today_str(),
            }
            save_json(state, self.state_path)
        return state

    def _save_state(self) -> None:
        self.state["last_updated"] = today_str()
        save_json(self.state, self.state_path)

    # ── Valuation ────────────────────────────────────────────────────────

    def update_prices(self, current_prices: dict[str, float]) -> float:
        """
        Marks every holding to ``current_prices`` where available, falling
        back to the last known price otherwise (a missed price fetch must
        never make a position silently vanish from the NAV). Returns total
        portfolio value and records it as today's NAV point.
        """
        holdings_value = 0.0
        for ticker, holding in self.state["holdings"].items():
            if holding["qty"] <= 0:
                continue
            fresh = current_prices.get(ticker)
            if is_valid_price(fresh):
                holding["last_price"] = fresh
            else:
                logger.warning(f"[PAPER] No fresh price for {ticker}; using stale last_price for valuation")
            holdings_value += holding["qty"] * holding.get("last_price", 0.0)

        total_value = self.state["cash"] + holdings_value
        self.state["daily_nav"][today_str()] = round(total_value, 2)
        self.state["peak_value"] = max(self.state["peak_value"], total_value)
        self._save_state()
        return total_value

    # ── Trade execution ──────────────────────────────────────────────────

    def buy(self, ticker: str, qty: int, price: float) -> dict | None:
        """Simulates a BUY: deducts trade value + all Groww buy-side costs from cash."""
        if qty <= 0 or not is_valid_price(price):
            return None

        trade_value = qty * price
        costs = self.cost_model.compute_buy_cost(trade_value)
        total_outflow = trade_value + costs.total

        if total_outflow > self.state["cash"]:
            logger.warning(
                f"[PAPER] Insufficient cash for {ticker}: need Rs.{total_outflow:,.2f}, "
                f"have Rs.{self.state['cash']:,.2f}"
            )
            return None

        self.state["cash"] -= total_outflow

        existing = self.state["holdings"].get(ticker)
        if existing and existing["qty"] > 0:
            new_qty = existing["qty"] + qty
            new_avg = (existing["avg_cost"] * existing["qty"] + price * qty) / new_qty
            existing.update({"qty": new_qty, "avg_cost": round(new_avg, 4), "last_price": price})
        else:
            self.state["holdings"][ticker] = {
                "qty": qty, "avg_cost": price, "last_price": price, "entry_date": today_str(),
            }

        trade_record = {
            "date": today_str(), "ticker": ticker, "action": "BUY",
            "qty": qty, "price": price, "value": trade_value,
            "cost_breakdown": {
                "brokerage": costs.brokerage, "stt": costs.stt, "exchange_tc": costs.exchange_tc,
                "sebi_charge": costs.sebi_charge, "stamp_duty": costs.stamp_duty, "gst": costs.gst,
            },
            "total_cost": round(costs.total, 2),
            "cash_after": round(self.state["cash"], 2),
        }
        self.state["trade_log"].append(trade_record)
        logger.info(
            f"[PAPER BUY]  {ticker}: {qty} x Rs.{price:.2f} = Rs.{trade_value:,.2f} "
            f"+ Rs.{costs.total:.2f} costs | Cash left: Rs.{self.state['cash']:,.2f}"
        )
        self._save_state()
        return trade_record

    def sell(self, ticker: str, qty: int, price: float) -> dict | None:
        """Simulates a SELL: credits proceeds minus all Groww sell-side costs."""
        if qty <= 0 or not is_valid_price(price):
            return None

        holding = self.state["holdings"].get(ticker)
        if not holding or holding["qty"] < qty:
            have = holding["qty"] if holding else 0
            logger.warning(f"[PAPER] Cannot sell {qty} of {ticker} — only holding {have}")
            return None

        trade_value = qty * price
        costs = self.cost_model.compute_sell_cost(trade_value)
        proceeds = trade_value - costs.total

        self.state["cash"] += proceeds
        holding["qty"] -= qty
        holding["last_price"] = price

        realised_pnl = (price - holding["avg_cost"]) * qty - costs.total

        trade_record = {
            "date": today_str(), "ticker": ticker, "action": "SELL",
            "qty": qty, "price": price, "value": trade_value,
            "cost_breakdown": {
                "brokerage": costs.brokerage, "stt": costs.stt, "exchange_tc": costs.exchange_tc,
                "sebi_charge": costs.sebi_charge, "dp_charge": costs.dp_charge, "gst": costs.gst,
            },
            "total_cost": round(costs.total, 2),
            "realised_pnl": round(realised_pnl, 2),
            "cash_after": round(self.state["cash"], 2),
        }
        self.state["trade_log"].append(trade_record)
        logger.info(
            f"[PAPER SELL] {ticker}: {qty} x Rs.{price:.2f} = Rs.{trade_value:,.2f} "
            f"- Rs.{costs.total:.2f} costs | PnL: Rs.{realised_pnl:,.2f} | Cash: Rs.{self.state['cash']:,.2f}"
        )
        self._save_state()
        return trade_record

    def sell_all(self, ticker: str, price: float) -> dict | None:
        """Liquidates the entire position in one ticker."""
        qty = self.state["holdings"].get(ticker, {}).get("qty", 0)
        return self.sell(ticker, qty, price) if qty > 0 else None

    def _affordable_qty(self, price: float, desired_qty: int) -> int:
        """
        Largest quantity <= ``desired_qty`` whose full buy cost (trade value
        + Groww fees) fits available cash, found by binary search (buy cost
        is monotonically non-decreasing in trade value).

        Needed because a rebalance target for one ticker is sized against
        total portfolio value, not against what's left in cash *after*
        buying the tickers processed earlier in the same pass — without
        this, the first ticker in iteration order can exhaust the cash
        buffer and leave every later ticker at zero, instead of every
        ticker landing close to its intended weight.
        """
        if desired_qty <= 0 or price <= 0:
            return 0

        def outflow(qty: int) -> float:
            trade_value = qty * price
            return trade_value + self.cost_model.compute_buy_cost(trade_value).total

        if outflow(desired_qty) <= self.state["cash"]:
            return desired_qty

        lo, hi = 0, desired_qty
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if outflow(mid) <= self.state["cash"]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    # ── Rebalancing ──────────────────────────────────────────────────────

    def rebalance(
        self,
        target_weights: dict[str, float],
        current_prices: dict[str, float],
        circuit_action: Action = Action.HOLD,
    ) -> list[dict]:
        """
        Moves current holdings toward ``target_weights`` with the minimal
        set of trades (sells first to free up cash, then buys), unless
        ``circuit_action`` overrides normal rebalancing entirely.
        """
        trades: list[dict] = []
        total_value = self.update_prices(current_prices)

        if circuit_action == Action.HALT:
            logger.warning("[REBALANCE] HALT — no trades executed.")
            return trades

        if circuit_action == Action.FULL_EXIT:
            logger.warning("[REBALANCE] FULL EXIT — liquidating all positions.")
            for ticker, holding in list(self.state["holdings"].items()):
                if holding["qty"] > 0 and ticker in current_prices:
                    t = self.sell_all(ticker, current_prices[ticker])
                    if t:
                        trades.append(t)
            return trades

        if circuit_action == Action.REDUCE_50:
            # Halve every position and STOP — do not fall through to the
            # normal target-weight buy logic below. The optimiser's target
            # weights reflect yesterday's regime; buying back toward them in
            # the same pass would immediately undo the de-risking this
            # circuit breaker exists to enforce. The next HOLD-state run
            # will pick up rebalancing from the reduced book.
            logger.warning("[REBALANCE] REDUCE_50 — halving all positions, no further trades this run.")
            for ticker, holding in list(self.state["holdings"].items()):
                if holding["qty"] > 0 and ticker in current_prices:
                    sell_qty = holding["qty"] // 2
                    if sell_qty > 0:
                        t = self.sell(ticker, sell_qty, current_prices[ticker])
                        if t:
                            trades.append(t)
            self.update_prices(current_prices)
            return trades

        # Normal rebalance.
        current_weights = {
            t: (h["qty"] * current_prices.get(t, h["last_price"])) / total_value
            for t, h in self.state["holdings"].items() if h["qty"] > 0
        }

        # Sells first — trim anything over target by more than the sell band,
        # or fully exit anything no longer in the target portfolio at all.
        for ticker, curr_w in current_weights.items():
            target_w = target_weights.get(ticker, 0.0)
            price = current_prices.get(ticker)
            if not is_valid_price(price):
                continue
            if target_w == 0 or (curr_w - target_w) > settings.REBALANCE_SELL_BAND:
                target_value = target_w * total_value
                current_value = self.state["holdings"][ticker]["qty"] * price
                sell_qty = max(0, int((current_value - target_value) // price))
                if sell_qty > 0:
                    t = self.sell(ticker, sell_qty, price)
                    if t:
                        trades.append(t)

        # Buys — top up anything under target by more than the buy band.
        total_value = self.update_prices(current_prices)
        for ticker, target_w in target_weights.items():
            price = current_prices.get(ticker)
            if not is_valid_price(price):
                continue
            curr_holding = self.state["holdings"].get(ticker, {})
            curr_value = curr_holding.get("qty", 0) * price
            target_value = target_w * total_value
            gap = target_value - curr_value
            if gap > settings.MIN_POSITION_VALUE_INR:
                desired_qty = int(gap // price)
                buy_qty = self._affordable_qty(price, desired_qty)
                if buy_qty > 0:
                    t = self.buy(ticker, buy_qty, price)
                    if t:
                        trades.append(t)

        logger.info(f"[REBALANCE] {len(trades)} trades executed. Portfolio value: Rs.{total_value:,.2f}")
        return trades

    # ── Summary ──────────────────────────────────────────────────────────

    def get_summary(self, benchmark_return: float | None = None) -> dict:
        """Key performance metrics derived from the daily NAV history and trade log."""
        initial = self.state["initial_capital"]
        peak = self.state["peak_value"]

        nav_series = pd.Series(self.state["daily_nav"], dtype=float)
        if nav_series.empty:
            current = initial
            ann_return, sharpe = 0.0, 0.0
        else:
            nav_series.index = pd.to_datetime(nav_series.index, format="%Y%m%d")
            nav_series = nav_series.sort_index()
            current = float(nav_series.iloc[-1])

            if len(nav_series) > 1:
                days = (nav_series.index[-1] - nav_series.index[0]).days
                years = max(days / 365.25, 1 / 365.25)
                ann_return = (current / initial) ** (1 / years) - 1
            else:
                ann_return = 0.0

            if len(nav_series) > 2:
                daily_returns = nav_series.pct_change().dropna()
                daily_std = daily_returns.std()
                sharpe = (
                    (daily_returns.mean() - settings.RISK_FREE_RATE / settings.TRADING_DAYS)
                    / daily_std * (settings.TRADING_DAYS ** 0.5)
                    if daily_std > 0 else 0.0
                )
            else:
                sharpe = 0.0

        drawdown = (peak - current) / max(peak, 1e-9)
        total_pnl = current - initial
        total_costs = sum(t["total_cost"] for t in self.state["trade_log"])

        return {
            "initial_capital": initial,
            "current_value": current,
            "cash": self.state["cash"],
            "holdings_value": current - self.state["cash"],
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl / initial * 100,
            "annualised_return": ann_return * 100,
            "max_drawdown_pct": drawdown * 100,
            "sharpe_ratio": sharpe,
            "total_costs_paid": total_costs,
            "total_trades": len(self.state["trade_log"]),
            "benchmark_return": benchmark_return,
        }
