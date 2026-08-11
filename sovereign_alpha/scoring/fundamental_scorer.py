"""
L3 — Fundamental Scorer.

Transforms harmonized financial metrics into the Unified Stock Health Score
(USHS), 0-100. Every step is vectorised pandas — no Python loops over rows,
which matters once the universe grows past Nifty 50 to Nifty 500.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import settings
from sovereign_alpha.utils.helpers import today_str
from sovereign_alpha.utils.logger import logger

_SCORE_COLUMNS = [
    "pe_score", "pb_score", "roe_score", "fcf_yield_score", "debt_equity_score",
    "promoter_pledge_score", "altman_z_score", "beneish_m_score", "sentiment_score",
]


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """
    ``df[name]`` if the column exists, else an all-NaN series aligned to
    ``df.index``. Using ``df.get(name, pd.Series(dtype=float))`` (as a naive
    implementation would) returns an *empty* series with no index when the
    column is absent — every downstream vectorised op then silently
    misaligns and produces all-NaN garbage instead of raising. This keeps
    the index correct in that edge case.
    """
    if name in df.columns:
        return df[name]
    return pd.Series(np.nan, index=df.index)


class FundamentalScorer:
    """Computes USHS and the eligible-for-trading filter from it."""

    # ── Scoring primitive ──────────────────────────────────────────────────

    @staticmethod
    def _percentile_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
        """Rank each value 0-100 by percentile within the universe. NaN -> 50 (neutral)."""
        if series.empty or series.dropna().empty:
            return pd.Series(50.0, index=series.index)
        pct = series.rank(pct=True, na_option="keep") * 100
        pct = pct.fillna(50.0)
        return pct if higher_is_better else (100 - pct)

    # ── Individual factor scorers ──────────────────────────────────────────

    def score_pe(self, df: pd.DataFrame) -> pd.Series:
        """Sector-normalised P/E; higher normalised value = cheaper vs peers = better."""
        col = _col(df, "pe_sector_normalised")
        if col.dropna().empty:
            col = _col(df, "pe")
        return self._percentile_score(col, higher_is_better=True)

    def score_pb(self, df: pd.DataFrame) -> pd.Series:
        """Lower P/B = better. Outliers above 20x capped before ranking."""
        pb = _col(df, "pb").clip(upper=20)
        return self._percentile_score(pb, higher_is_better=False)

    def score_roe(self, df: pd.DataFrame) -> pd.Series:
        """Higher ROE = better. Clipped to [-50%, 100%] to blunt extreme outliers."""
        roe = _col(df, "roe").clip(-0.5, 1.0)
        return self._percentile_score(roe, higher_is_better=True)

    def score_fcf_yield(self, df: pd.DataFrame) -> pd.Series:
        return self._percentile_score(_col(df, "fcf_yield"), higher_is_better=True)

    def score_debt_equity(self, df: pd.DataFrame) -> pd.Series:
        """
        Lower D/E = better. yfinance reports ``debtToEquity`` as a percentage
        (234 == 2.34x), so D/E > 5x (value > 500) or negative equity
        (value < -100) are hard-overridden to 0 regardless of percentile rank.
        """
        de = _col(df, "debt_equity")
        scored = self._percentile_score(de, higher_is_better=False)
        scored = scored.where(~(de > 500), 0.0)
        scored = scored.where(~(de < -100), 0.0)
        return scored

    def score_promoter_pledge(self, df: pd.DataFrame) -> pd.Series:
        """0% pledge -> 100. 50%+ pledge -> 0, linear between. Missing -> neutral (50)."""
        pledge = _col(df, "promoter_pledge_pct")
        scored = 100.0 - (pledge.clip(0, 50) * 2.0)
        return scored.fillna(50.0)

    def score_altman_z(self, df: pd.DataFrame) -> pd.Series:
        """Z > 2.6 -> 100 (safe). 1.1 < Z <= 2.6 -> 40-80 (grey, linear). Z <= 1.1 -> 0. NaN -> 40."""
        z = _col(df, "altman_z")

        def _to_score(val: float) -> float:
            if pd.isna(val):
                return 40.0
            if val > 2.6:
                return 100.0
            if val > 1.1:
                return 40.0 + (val - 1.1) / (2.6 - 1.1) * 40.0
            return 0.0

        return z.map(_to_score)

    def score_beneish_m(self, df: pd.DataFrame) -> pd.Series:
        """M < -2.22 -> 100 (safe). -2.22 <= M < -1.78 -> 50 (caution). M >= -1.78 -> 0. NaN -> 50."""
        m = _col(df, "beneish_m")

        def _to_score(val: float) -> float:
            if pd.isna(val):
                return 50.0
            if val < -2.22:
                return 100.0
            if val < -1.78:
                return 50.0
            return 0.0

        return m.map(_to_score)

    # ── Master USHS ─────────────────────────────────────────────────────────

    def compute_ushs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds all ``*_score`` columns plus ``ushs``, ``ushs_rank``,
        ``ushs_grade`` to a copy of ``df``. Saves a ranked CSV to
        ``data/outputs/ushs_scores_YYYYMMDD.csv``.
        """
        if df.empty:
            raise ValueError("compute_ushs: input DataFrame is empty")

        df = df.copy()
        scores = {
            "pe_score": self.score_pe(df),
            "pb_score": self.score_pb(df),
            "roe_score": self.score_roe(df),
            "fcf_yield_score": self.score_fcf_yield(df),
            "debt_equity_score": self.score_debt_equity(df),
            "promoter_pledge_score": self.score_promoter_pledge(df),
            "altman_z_score": self.score_altman_z(df),
            "beneish_m_score": self.score_beneish_m(df),
            "sentiment_score": pd.Series(50.0, index=df.index),  # neutral — no sentiment agent live yet
        }
        for col, values in scores.items():
            df[col] = values.reindex(df.index)

        w = settings.USHS_WEIGHTS
        df["ushs"] = sum(df[col] * weight for col, weight in
                          ((k, w[k]) for k in _SCORE_COLUMNS)).round(2)

        df["ushs_rank"] = df["ushs"].rank(ascending=False, method="min").astype(int)
        df["ushs_grade"] = pd.cut(
            df["ushs"], bins=[0, 40, 60, 80, 100], labels=["D", "C", "B", "A"], include_lowest=True,
        )

        df_sorted = df.sort_values("ushs_rank")
        out_path = settings.OUTPUT_DIR / f"ushs_scores_{today_str()}.csv"
        cols_to_save = [c for c in (_SCORE_COLUMNS + ["ushs", "ushs_rank", "ushs_grade", "sector"])
                         if c in df_sorted.columns]
        df_sorted[cols_to_save].to_csv(out_path)
        logger.info(f"[USHS] Saved to {out_path}")
        logger.info("\n" + df_sorted[["ushs", "ushs_rank", "ushs_grade", "sector"]].head(15).to_string())
        return df

    def get_eligible_tickers(self, scored_df: pd.DataFrame) -> list[str]:
        """
        Investable universe: USHS >= threshold AND not in Altman distress AND
        not flagged by Beneish M. Tickers with unknown Altman/Beneish
        (NaN, scored neutrally above) are *not* excluded by this filter —
        only a confirmed bad reading disqualifies a stock.
        """
        altman = _col(scored_df, "altman_z")
        beneish = _col(scored_df, "beneish_m")

        mask = (
            (scored_df["ushs"] >= settings.MIN_USHS_THRESHOLD)
            & ~(altman <= 1.1)
            & ~(beneish >= -1.78)
        )
        # Drop illiquid stocks — impact cost model breaks below this threshold
        if "avg_daily_volume_inr" in scored_df.columns:
            mask &= scored_df["avg_daily_volume_inr"] >= settings.MIN_AVG_DAILY_VOLUME_INR
        eligible = scored_df[mask].sort_values("ushs", ascending=False)
        tickers = eligible.index.tolist()
        logger.info(f"[ELIGIBLE] {len(tickers)} tickers pass all filters: {tickers}")
        return tickers


if __name__ == "__main__":
    from sovereign_alpha.ingestion.data_harmonizer import DataHarmonizer

    df = DataHarmonizer().load_latest()
    scorer = FundamentalScorer()
    scored = scorer.compute_ushs(df)
    print("\nTop 10 by USHS:")
    print(scored[["ushs", "ushs_rank", "ushs_grade"]].head(10))
    print("FundamentalScorer OK")
