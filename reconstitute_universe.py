"""
Universe reconstitution entry point.

Fetches current NSE index constituent data (Nifty 50, Next 50, Midcap 150,
Smallcap 250, India Defence) and writes ``data/universe/universe_data.json``
— the file ``config/universe.py`` loads at import time. Meant to run
periodically (see .github/workflows/universe_reconstitution.yml — monthly,
since real index reconstitution only happens quarterly/semi-annually), not
on the daily trading engine's critical path.

Usage:
  python reconstitute_universe.py
  python reconstitute_universe.py --max-size 300
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

from config import settings
from sovereign_alpha.ingestion.universe_builder import DEFAULT_MAX_UNIVERSE_SIZE, build_universe
from sovereign_alpha.utils.logger import logger


def main() -> None:
    parser = argparse.ArgumentParser(description="Sovereign Alpha Engine — Universe Reconstitution")
    parser.add_argument("--max-size", type=int, default=DEFAULT_MAX_UNIVERSE_SIZE)
    args = parser.parse_args()

    result = build_universe(max_size=args.max_size)

    out_path = settings.BASE_DIR / "data" / "universe" / "universe_data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "tickers": result["tickers"],
        "large_cap_tickers": result["large_cap_tickers"],
        "non_large_cap_tickers": result["non_large_cap_tickers"],
        "sector_map": result["sector_map"],
        "sources": result["sources"],
        "built_at": result["built_at"],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    logger.info(f"[RECONSTITUTE] Wrote {len(result['tickers'])} tickers to {out_path}")
    logger.info(f"  Large-cap:      {len(result['large_cap_tickers'])}")
    logger.info(f"  Mid/small/thematic: {len(result['non_large_cap_tickers'])}")
    logger.info(f"  Timestamp:      {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
