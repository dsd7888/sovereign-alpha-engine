"""
Exact Groww equity-delivery cost model (post 21-Jun-2025 rate revision).
Used both to penalise expected returns in the optimiser and to deduct real
costs from virtual cash in the PaperBroker — the same numbers must be used
in both places, which is exactly why this lives in one shared module.
"""
from __future__ import annotations

from dataclasses import dataclass

from config import settings


@dataclass(frozen=True)
class CostBreakdown:
    brokerage: float
    stt: float
    exchange_tc: float
    sebi_charge: float
    stamp_duty: float
    dp_charge: float
    gst: float
    total: float
    total_pct: float  # total / trade_value


_ZERO_COST = CostBreakdown(0, 0, 0, 0, 0, 0, 0, 0, 0.0)


class GrowwCostModel:
    """
    Groww equity delivery charges:
      BUY:  brokerage = clip(0.1% of value, min=5, max=20)
            + STT 0.1% + exchange 0.00297% + SEBI 1e-7*value + stamp duty 0.015%
            + GST 18% on (brokerage + exchange + SEBI)
      SELL: same brokerage/STT/exchange/SEBI, no stamp duty, + DP charge ₹20/scrip
            + GST 18% on (brokerage + exchange + SEBI + DP)
    """

    @staticmethod
    def _brokerage(trade_value: float) -> float:
        return max(
            settings.GROWW_BROKERAGE_MIN,
            min(settings.GROWW_BROKERAGE_MAX, settings.GROWW_BROKERAGE_PCT * trade_value),
        )

    @classmethod
    def compute_buy_cost(cls, trade_value: float) -> CostBreakdown:
        if trade_value <= 0:
            return _ZERO_COST

        brokerage = cls._brokerage(trade_value)
        stt = settings.STT_RATE * trade_value
        exchange_tc = settings.EXCHANGE_TC_RATE * trade_value
        sebi = settings.SEBI_RATE * trade_value
        stamp_duty = settings.STAMP_DUTY_RATE * trade_value
        gst = settings.GST_RATE * (brokerage + exchange_tc + sebi)
        total = brokerage + stt + exchange_tc + sebi + stamp_duty + gst

        return CostBreakdown(
            brokerage=brokerage, stt=stt, exchange_tc=exchange_tc, sebi_charge=sebi,
            stamp_duty=stamp_duty, dp_charge=0.0, gst=gst, total=total,
            total_pct=total / trade_value,
        )

    @classmethod
    def compute_sell_cost(cls, trade_value: float) -> CostBreakdown:
        if trade_value <= 0:
            return _ZERO_COST

        brokerage = cls._brokerage(trade_value)
        stt = settings.STT_RATE * trade_value
        exchange_tc = settings.EXCHANGE_TC_RATE * trade_value
        sebi = settings.SEBI_RATE * trade_value
        dp_charge = settings.DP_CHARGE_PER_SELL
        gst = settings.GST_RATE * (brokerage + exchange_tc + sebi + dp_charge)
        total = brokerage + stt + exchange_tc + sebi + dp_charge + gst

        return CostBreakdown(
            brokerage=brokerage, stt=stt, exchange_tc=exchange_tc, sebi_charge=sebi,
            stamp_duty=0.0, dp_charge=dp_charge, gst=gst, total=total,
            total_pct=total / trade_value,
        )

    @staticmethod
    def compute_impact_cost(
        order_value: float, avg_daily_volume_inr: float, daily_volatility: float
    ) -> float:
        """
        Impact cost for large orders that move the market:
        IC = alpha * (order_value / avg_daily_volume)^beta * daily_volatility.
        Returns a fraction, e.g. 0.003 == 0.3%. Capped at
        ``settings.IMPACT_COST_CAP`` — beyond that the position is simply
        too large for the stock's liquidity and must be trimmed upstream.
        """
        if order_value <= 0:
            return 0.0
        if avg_daily_volume_inr <= 0:
            return 0.005  # no volume data — assume a conservative 0.5%

        q_v_ratio = order_value / avg_daily_volume_inr
        ic = settings.IMPACT_COST_ALPHA * (q_v_ratio ** settings.IMPACT_COST_BETA) * max(daily_volatility, 0.0)
        return float(min(ic, settings.IMPACT_COST_CAP))

    @classmethod
    def round_trip_cost_fraction(
        cls,
        trade_value: float,
        avg_daily_volume_inr: float = 1e8,
        daily_volatility: float = 0.015,
    ) -> float:
        """Total cost fraction for buy + sell of the same value (used to penalise turnover)."""
        buy = cls.compute_buy_cost(trade_value)
        sell = cls.compute_sell_cost(trade_value)
        ic = cls.compute_impact_cost(trade_value, avg_daily_volume_inr, daily_volatility)
        return buy.total_pct + sell.total_pct + ic

    @classmethod
    def penalise_return(
        cls,
        expected_annual_return: float,
        annual_turnover: float = 1.0,
        avg_trade_value: float = 50_000,
    ) -> float:
        """Cost-adjusted return = gross return - (round-trip cost * annual turnover)."""
        rt_cost = cls.round_trip_cost_fraction(avg_trade_value)
        return expected_annual_return - (rt_cost * annual_turnover)


if __name__ == "__main__":
    tv = 100_000
    buy = GrowwCostModel.compute_buy_cost(tv)
    sell = GrowwCostModel.compute_sell_cost(tv)
    print(f"BUY  ₹{tv:,.0f}: total = ₹{buy.total:.2f} ({buy.total_pct * 100:.3f}%)")
    print(f"  brokerage ₹{buy.brokerage:.2f} | stt ₹{buy.stt:.2f} | exch ₹{buy.exchange_tc:.2f} "
          f"| stamp ₹{buy.stamp_duty:.2f} | gst ₹{buy.gst:.2f}")
    print(f"SELL ₹{tv:,.0f}: total = ₹{sell.total:.2f} ({sell.total_pct * 100:.3f}%)")
    print(f"  brokerage ₹{sell.brokerage:.2f} | stt ₹{sell.stt:.2f} | dp ₹{sell.dp_charge:.2f} "
          f"| gst ₹{sell.gst:.2f}")
    rt = GrowwCostModel.round_trip_cost_fraction(tv)
    print(f"Round-trip cost: {rt * 100:.3f}% (~₹{rt * tv:.0f} on ₹{tv:,.0f})")
    print("GrowwCostModel OK")
