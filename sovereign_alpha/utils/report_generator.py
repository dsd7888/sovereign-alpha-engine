"""Generates the daily HTML report — a single self-contained file, no external assets."""
from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import settings
from sovereign_alpha.utils.helpers import today_str
from sovereign_alpha.utils.logger import logger

_GRADE_COLORS = {"A": "#2ecc71", "B": "#f39c12", "C": "#e67e22", "D": "#e74c3c"}


def _holdings_rows(holdings: dict[str, dict]) -> str:
    rows = ""
    for ticker, h in holdings.items():
        if h.get("qty", 0) == 0:
            continue
        entry = h.get("avg_cost", 0)
        curr = h.get("last_price", entry)
        pnl_pct = (curr - entry) / max(entry, 1e-9) * 100
        color = "#2ecc71" if pnl_pct >= 0 else "#e74c3c"
        rows += (
            f"<tr><td>{ticker.replace('.NS', '')}</td><td>{h.get('qty', '')}</td>"
            f"<td>Rs.{entry:,.2f}</td><td>Rs.{curr:,.2f}</td>"
            f"<td style='color:{color}'>{pnl_pct:+.2f}%</td>"
            f"<td>Rs.{h.get('qty', 0) * curr:,.2f}</td></tr>"
        )
    return rows


def _completeness_color(pct: float) -> str:
    if pct >= 80:
        return "#2ecc71"
    if pct >= 50:
        return "#f39c12"
    return "#e74c3c"


def _top_stocks_rows(scored_df: pd.DataFrame, n: int = 15) -> str:
    cols = [
        c for c in ("ushs", "ushs_grade", "sector", "pe", "roe", "altman_z", "beneish_m", "data_completeness_pct")
        if c in scored_df.columns
    ]
    top = scored_df.sort_values("ushs", ascending=False)[cols].head(n).round(2)

    rows = ""
    for ticker, row in top.iterrows():
        grade = str(row.get("ushs_grade", "C"))
        color = _GRADE_COLORS.get(grade, "#aaa")
        roe = row.get("roe")
        roe_str = f"{roe * 100:.1f}%" if pd.notna(roe) else "N/A"
        completeness = row.get("data_completeness_pct")
        if pd.notna(completeness):
            comp_color = _completeness_color(completeness)
            comp_str = f"<td style='color:{comp_color}; font-weight:bold'>{completeness:.0f}%</td>"
        else:
            comp_str = "<td>N/A</td>"
        rows += (
            f"<tr><td>{ticker.replace('.NS', '')}</td>"
            f"<td style='color:{color}; font-weight:bold'>{grade} ({row.get('ushs', 0):.1f})</td>"
            f"<td>{row.get('sector', '')}</td>"
            f"<td>{row.get('pe', 'N/A')}</td>"
            f"<td>{roe_str}</td>"
            f"<td>{row.get('altman_z', 'N/A')}</td>"
            f"{comp_str}</tr>"
        )
    return rows


def generate_daily_report(
    summary: dict,
    scored_df: pd.DataFrame,
    portfolio_state: dict,
    vix: float,
    frontier_chart_path: Path | None = None,
) -> Path:
    """Builds and saves the daily HTML report. Returns the path written."""
    holdings_rows = _holdings_rows(portfolio_state.get("holdings", {}))
    stocks_rows = _top_stocks_rows(scored_df)

    vix_color = "#2ecc71" if vix < settings.VIX_ELEVATED else (
        "#f39c12" if vix < settings.VIX_HIGH_ALERT else "#e74c3c"
    )
    vix_label = "Low" if vix < settings.VIX_ELEVATED else (
        "Elevated" if vix < settings.VIX_HIGH_ALERT else "HIGH ALERT"
    )
    pnl_color = "#2ecc71" if summary.get("total_pnl_pct", 0) >= 0 else "#e74c3c"

    frontier_img = ""
    if frontier_chart_path and Path(frontier_chart_path).exists():
        img_bytes = Path(frontier_chart_path).read_bytes()
        img_b64   = base64.b64encode(img_bytes).decode("utf-8")
        frontier_img = (
            '<img src="data:image/png;base64,' + img_b64 + '" '
            'style="max-width:100%;border-radius:8px;margin-top:16px;" />'
        )

    holdings_block = (
        "<p style='color:#888'>No positions yet.</p>" if not holdings_rows else
        f"<table><tr><th>Ticker</th><th>Qty</th><th>Avg Cost</th><th>LTP</th>"
        f"<th>P&amp;L %</th><th>Value</th></tr>{holdings_rows}</table>"
    )
    frontier_block = (
        f'<div class="section"><h2>Efficient Frontier</h2>{frontier_img}</div>' if frontier_img else ""
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sovereign Alpha Engine — {today_str()}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f0f1a; color: #e0e0e0; padding: 24px; }}
  h1 {{ color: #FFD700; font-size: 1.8rem; margin-bottom: 4px; }}
  .subtitle {{ color: #888; margin-bottom: 24px; font-size: 0.9rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 32px; }}
  .card {{ background: #1a1a2e; border-radius: 10px; padding: 20px; border: 1px solid #2a2a4e; }}
  .card .label {{ font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }}
  .card .value {{ font-size: 1.5rem; font-weight: 700; }}
  .card .sub {{ font-size: 0.8rem; color: #888; margin-top: 4px; }}
  h2 {{ color: #FFD700; font-size: 1.1rem; margin: 24px 0 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ background: #1a1a2e; color: #888; text-align: left; padding: 10px 12px; border-bottom: 1px solid #2a2a4e; }}
  td {{ padding: 9px 12px; border-bottom: 1px solid #151528; }}
  tr:hover td {{ background: #1f1f38; }}
  .section {{ background: #1a1a2e; border-radius: 10px; padding: 20px; margin-bottom: 24px; border: 1px solid #2a2a4e; }}
</style>
</head>
<body>
<h1>Sovereign Alpha Engine</h1>
<div class="subtitle">Daily Report &middot; {datetime.now().strftime("%A, %d %B %Y %H:%M IST")} &middot; Universe: Nifty 50 Pilot</div>

<div class="grid">
  <div class="card">
    <div class="label">Portfolio Value</div>
    <div class="value">Rs.{summary.get('current_value', 0):,.0f}</div>
    <div class="sub">Initial: Rs.{summary.get('initial_capital', 0):,.0f}</div>
  </div>
  <div class="card">
    <div class="label">Total P&amp;L</div>
    <div class="value" style="color:{pnl_color}">Rs.{summary.get('total_pnl', 0):+,.0f}</div>
    <div class="sub" style="color:{pnl_color}">{summary.get('total_pnl_pct', 0):+.2f}%</div>
  </div>
  <div class="card">
    <div class="label">Annualised Return</div>
    <div class="value">{summary.get('annualised_return', 0):.1f}%</div>
    <div class="sub">Sharpe: {summary.get('sharpe_ratio', 0):.2f}</div>
  </div>
  <div class="card">
    <div class="label">Max Drawdown</div>
    <div class="value" style="color:#e74c3c">{summary.get('max_drawdown_pct', 0):.1f}%</div>
    <div class="sub">Circuit @ {settings.DRAWDOWN_REDUCE_50:.0%} / {settings.DRAWDOWN_FULL_EXIT:.0%}</div>
  </div>
  <div class="card">
    <div class="label">India VIX</div>
    <div class="value" style="color:{vix_color}">{vix:.1f}</div>
    <div class="sub" style="color:{vix_color}">{vix_label}</div>
  </div>
  <div class="card">
    <div class="label">Cash Available</div>
    <div class="value">Rs.{summary.get('cash', 0):,.0f}</div>
    <div class="sub">Total costs paid: Rs.{summary.get('total_costs_paid', 0):,.2f}</div>
  </div>
</div>

<div class="section">
  <h2>Current Holdings</h2>
  {holdings_block}
</div>

<div class="section">
  <h2>Top 15 Stocks by USHS Score</h2>
  <table>
    <tr><th>Ticker</th><th>USHS Grade</th><th>Sector</th><th>P/E</th><th>ROE</th><th>Altman Z</th><th>Data Completeness</th></tr>
    {stocks_rows}
  </table>
</div>

{frontier_block}

<div class="section" style="font-size:0.75rem; color:#555;">
  <p>Generated by Sovereign Alpha Engine &middot; Paper trading mode &middot; Groww delivery costs applied &middot;
  Round-trip cost &asymp; 0.30% per trade &middot; Not financial advice.</p>
</div>
</body>
</html>"""

    out_path = settings.REPORTS_DIR / f"report_{today_str()}.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info(f"[REPORT] Saved: {out_path}")
    return out_path
