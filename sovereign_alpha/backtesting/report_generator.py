"""Generates the backtest/walk-forward HTML report — same visual language as the daily report."""
from __future__ import annotations

import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import settings
from sovereign_alpha.utils.helpers import today_str
from sovereign_alpha.utils.logger import logger

_MODE_NOTES = {
    "technical": (
        "Stock selection driven ONLY by momentum + low-volatility — the two factors "
        "computable from real, point-in-time daily prices over the full window below. "
        "Every fundamentals factor (PE, PB, ROE, Altman Z, Beneish M, promoter pledge) is "
        "absent from this run and scored neutral, exactly as documented in "
        "sovereign_alpha/backtesting/engine.py. This is the statistically meaningful "
        "long-horizon result."
    ),
    "full": (
        "Replays the complete USHS including fundamentals, reconstructed at ANNUAL "
        "resolution from real statements (~5yr depth for most NSE names) with a "
        "75-day reporting lag to stay lookahead-safe. Promoter pledge and sentiment "
        "have no historical source at all and are held neutral throughout — see "
        "sovereign_alpha/backtesting/point_in_time.py for the full reasoning. PE/PB/FCF "
        "yield use the CURRENT share count applied across history (an approximation; "
        "share counts drift far more slowly than price or earnings)."
    ),
}


def _nav_chart_b64(nav: pd.Series, benchmark_nav: pd.Series | None = None) -> str:
    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.plot(nav.index, nav.values, color="#FFD700", linewidth=1.5, label="Strategy")
    if benchmark_nav is not None and not benchmark_nav.empty:
        ax.plot(benchmark_nav.index, benchmark_nav.values, color="#00CED1", linewidth=1.2,
                 linestyle="--", label="Nifty 50 (rebased)", alpha=0.8)
    ax.set_ylabel("NAV (Rs.)", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#2a2a3e", labelcolor="white")
    ax.grid(alpha=0.15)
    for spine in ax.spines.values():
        spine.set_color("#555")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _is_number(v) -> bool:
    return v is not None and not pd.isna(v)


def _metric_cards(overall: dict) -> str:
    def color(v: float | None) -> str:
        return "#2ecc71" if _is_number(v) and v >= 0 else "#e74c3c"

    cagr = overall.get("cagr_pct", float("nan"))
    sharpe = overall.get("sharpe_ratio", float("nan"))
    dd = overall.get("max_drawdown_pct", float("nan"))
    alpha = overall.get("alpha_annualised")
    alpha_pct = alpha * 100 if _is_number(alpha) else None

    cards = [
        ("Total Return", f"{overall.get('total_return_pct', 0):+.1f}%", color(overall.get("total_return_pct"))),
        ("CAGR", f"{cagr:.1f}%" if _is_number(cagr) else "N/A", color(cagr)),
        ("Sharpe Ratio", f"{sharpe:.2f}" if _is_number(sharpe) else "N/A", "#e0e0e0"),
        ("Sortino Ratio", f"{overall.get('sortino_ratio', float('nan')):.2f}", "#e0e0e0"),
        ("Max Drawdown", f"{dd:.1f}%" if _is_number(dd) else "N/A", "#e74c3c"),
        ("Calmar Ratio", f"{overall.get('calmar_ratio', float('nan')):.2f}", "#e0e0e0"),
        ("Win Rate", f"{overall.get('win_rate_pct', float('nan')):.0f}%", "#e0e0e0"),
        ("Ann. Turnover", f"{overall.get('annualised_turnover', float('nan')):.1f}x", "#e0e0e0"),
        ("Total Costs Paid", f"Rs.{overall.get('total_costs_paid', 0):,.0f}", "#888"),
        ("Alpha (ann., vs Nifty 50)", f"{alpha_pct:+.1f}%" if alpha_pct is not None else "N/A", color(alpha_pct)),
        ("Beta (vs Nifty 50)", f"{overall.get('beta', float('nan')):.2f}", "#e0e0e0"),
        ("Trades Executed", f"{overall.get('n_trades', 0):,}", "#888"),
    ]
    return "".join(
        f'<div class="card"><div class="label">{label}</div>'
        f'<div class="value" style="color:{c}">{val}</div></div>'
        for label, val, c in cards
    )


def _fold_rows(folds: list[dict]) -> str:
    rows = ""
    for f in folds:
        cagr = f.get("cagr_pct", float("nan"))
        color = "#2ecc71" if _is_number(cagr) and cagr >= 0 else "#e74c3c"
        rows += (
            f"<tr><td>{f['fold_start']} &rarr; {f['fold_end']}</td>"
            f"<td style='color:{color}'>{cagr:+.1f}%</td>"
            f"<td>{f.get('sharpe_ratio', float('nan')):.2f}</td>"
            f"<td>{f.get('max_drawdown_pct', float('nan')):.1f}%</td>"
            f"<td>{f.get('win_rate_pct', float('nan')):.0f}%</td>"
            f"<td>{f.get('n_trades', 0)}</td></tr>"
        )
    return rows


def generate_backtest_report(overall: dict, folds: list[dict], nav: pd.Series, mode: str) -> Path:
    """Builds and saves the backtest/walk-forward HTML report. Returns the path written."""
    chart_b64 = _nav_chart_b64(nav)
    cards = _metric_cards(overall)
    fold_rows = _fold_rows(folds)
    note = _MODE_NOTES.get(mode, "")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sovereign Alpha Engine — Backtest ({mode})</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f0f1a; color: #e0e0e0; padding: 24px; }}
  h1 {{ color: #FFD700; font-size: 1.8rem; margin-bottom: 4px; }}
  .subtitle {{ color: #888; margin-bottom: 24px; font-size: 0.9rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 32px; }}
  .card {{ background: #1a1a2e; border-radius: 10px; padding: 20px; border: 1px solid #2a2a4e; }}
  .card .label {{ font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }}
  .card .value {{ font-size: 1.5rem; font-weight: 700; }}
  h2 {{ color: #FFD700; font-size: 1.1rem; margin: 24px 0 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ background: #1a1a2e; color: #888; text-align: left; padding: 10px 12px; border-bottom: 1px solid #2a2a4e; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #151528; }}
  tr:hover td {{ background: #1f1f38; }}
  .section {{ background: #1a1a2e; border-radius: 10px; padding: 20px; margin-bottom: 24px; border: 1px solid #2a2a4e; }}
  .note {{ background: #241f0f; border: 1px solid #4a3f1a; border-radius: 8px; padding: 16px; font-size: 0.85rem; color: #d4af37; margin-bottom: 24px; }}
</style>
</head>
<body>
<h1>Sovereign Alpha Engine — Backtest</h1>
<div class="subtitle">Mode: {mode} &middot; {nav.index.min().strftime("%Y-%m-%d")} &rarr; {nav.index.max().strftime("%Y-%m-%d")} &middot;
Generated {datetime.now().strftime("%A, %d %B %Y %H:%M IST")}</div>

<div class="note"><strong>Data-availability note:</strong> {note}</div>

<div class="grid">
{cards}
</div>

<div class="section">
  <h2>NAV Over Time</h2>
  <img src="data:image/png;base64,{chart_b64}" style="max-width:100%;border-radius:8px;" />
</div>

<div class="section">
  <h2>Walk-Forward Folds (out-of-sample by calendar period)</h2>
  <table>
    <tr><th>Period</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Win Rate</th><th>Trades</th></tr>
    {fold_rows}
  </table>
</div>

<div class="section" style="font-size:0.75rem; color:#555;">
  <p>Paper-trading simulation &middot; Groww delivery costs applied &middot; Not financial advice.</p>
</div>
</body>
</html>"""

    out_dir = settings.OUTPUT_DIR / "backtests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"backtest_{mode}_{today_str()}.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info(f"[BACKTEST REPORT] Saved: {out_path}")
    return out_path
