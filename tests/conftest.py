"""
Shared test fixtures.

Several modules write to disk as a documented side effect of normal
operation (USHS CSVs, harmonized parquet, the efficient-frontier PNG,
cached OHLCV/fundamentals). Every one of them resolves its path through
``config.settings`` at call time via ``from config import settings`` (not
``from config.settings import X``, which would bind the value at import
time and be immune to monkeypatching) — so redirecting these attributes
here, before every test runs, is enough to guarantee the suite never reads
from or writes into the project's real ``data/`` or ``reports/``
directories, regardless of which module a test happens to exercise.
"""
import pytest

from config import settings


@pytest.fixture(autouse=True)
def _isolate_data_directories(tmp_path, monkeypatch):
    redirected = {
        "RAW_OHLCV_DIR": tmp_path / "raw" / "ohlcv",
        "RAW_FUND_DIR": tmp_path / "raw" / "fundamentals",
        "RAW_VIX_DIR": tmp_path / "raw" / "vix",
        "PROCESSED_DIR": tmp_path / "processed",
        "OUTPUT_DIR": tmp_path / "outputs",
        "PAPER_PORTFOLIO_DIR": tmp_path / "paper_portfolio",
        "REPORTS_DIR": tmp_path / "reports",
    }
    for name, path in redirected.items():
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(settings, name, path)
    monkeypatch.setattr(settings, "PAPER_STATE_FILE", redirected["PAPER_PORTFOLIO_DIR"] / "portfolio_state.json")
    yield
