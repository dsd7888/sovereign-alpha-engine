"""Small, dependency-light utility functions shared across layers."""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from sovereign_alpha.utils.logger import logger


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file. Returns ``{}`` if missing, empty, or corrupt (never raises)."""
    try:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return {}
        return json.loads(text)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"[STATE] Corrupt JSON at {path}: {e}. Treating as empty.")
        return {}


def save_json(data: dict[str, Any], path: Path) -> None:
    """
    Atomically write ``data`` as pretty JSON to ``path``.

    Writes to a temp file in the same directory then ``os.replace``s it into
    place, so a crash or kill mid-write can never leave a truncated/corrupt
    state file behind (this file is the source of truth for the paper
    portfolio — corruption there is unacceptable).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def is_cache_valid(cache_path: Path, max_age_hours: float) -> bool:
    """True if ``cache_path`` exists and is younger than ``max_age_hours``."""
    if not cache_path.exists():
        return False
    age_hours = (datetime.now().timestamp() - cache_path.stat().st_mtime) / 3600
    return age_hours < max_age_hours


def today_str() -> str:
    return datetime.now().strftime("%Y%m%d")


def extract_statement_row(
    df: pd.DataFrame | None, candidates: list[str], blocklist: list[str] | None = None
) -> pd.Series:
    """First matching row (case-insensitive) from a yfinance statement DataFrame.

    Tries an exact label match first, then falls back to a word-boundary
    match against the start of the row label (e.g. "Operating Income" matches
    "Operating Income Loss" but not "Other Non Operating Income Expenses").
    If ``blocklist`` is given, rows whose label contains any of its phrases
    are never matched via the fallback — used for EBIT/EBITDA lookups, where
    a plain substring match previously fabricated an EBIT value for HDFCBANK
    by matching "Other Non Operating Income Expenses" against "Operating
    Income". Shared by ``DataHarmonizer`` and ``DataIngestor`` so both stay
    fixed together rather than drifting into two copies of this bug.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)
    index_lower = {str(i).strip().lower(): i for i in df.index}
    for cand in candidates:
        cand_l = cand.strip().lower()
        if cand_l in index_lower:
            return df.loc[index_lower[cand_l]]
    for cand in candidates:
        cand_l = cand.strip().lower()
        for lower_name, orig_name in index_lower.items():
            if blocklist and any(blocked in lower_name for blocked in blocklist):
                continue
            if lower_name == cand_l or re.match(rf"\b{re.escape(cand_l)}\b", lower_name):
                return df.loc[orig_name]
    return pd.Series(dtype=float)


def is_market_day() -> bool:
    """
    True if today is a weekday (Mon-Fri).

    Does NOT account for NSE trading holidays (Republic Day, Diwali, etc.) —
    acceptable for the pilot; the engine will simply attempt a run on those
    days and fall back to the last cached close (see DataIngestor caching).
    """
    return datetime.now().weekday() < 5
