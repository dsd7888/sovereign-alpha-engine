import json

import config.universe as universe_module


class TestLoadUniverseData:
    def test_missing_file_falls_back_to_seed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(universe_module, "_UNIVERSE_DATA_PATH", tmp_path / "does_not_exist.json")
        data = universe_module._load_universe_data()
        assert data["tickers"] == list(
            dict.fromkeys(universe_module._FALLBACK_NIFTY_50 + universe_module._FALLBACK_NON_LARGECAP)
        )

    def test_corrupt_json_falls_back_to_seed(self, tmp_path, monkeypatch):
        path = tmp_path / "universe_data.json"
        path.write_text("{not valid json", encoding="utf-8")
        monkeypatch.setattr(universe_module, "_UNIVERSE_DATA_PATH", path)
        data = universe_module._load_universe_data()
        assert data["large_cap_tickers"] == universe_module._FALLBACK_NIFTY_50

    def test_missing_required_keys_falls_back_to_seed(self, tmp_path, monkeypatch):
        path = tmp_path / "universe_data.json"
        path.write_text(json.dumps({"tickers": ["A.NS"]}), encoding="utf-8")  # no large_cap_tickers
        monkeypatch.setattr(universe_module, "_UNIVERSE_DATA_PATH", path)
        data = universe_module._load_universe_data()
        assert data["large_cap_tickers"] == universe_module._FALLBACK_NIFTY_50

    def test_valid_file_is_used_as_is(self, tmp_path, monkeypatch):
        path = tmp_path / "universe_data.json"
        payload = {
            "tickers": ["X.NS", "Y.NS"],
            "large_cap_tickers": ["X.NS"],
            "non_large_cap_tickers": ["Y.NS"],
            "sector_map": {"X.NS": "IT", "Y.NS": "Auto"},
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setattr(universe_module, "_UNIVERSE_DATA_PATH", path)
        data = universe_module._load_universe_data()
        assert data["tickers"] == ["X.NS", "Y.NS"]
        assert data["sector_map"]["X.NS"] == "IT"

# NOTE: deliberately not testing the module-level constants (PILOT_UNIVERSE,
# NIFTY_50, etc.) via importlib.reload(). Other already-imported modules
# did `from config.universe import sector_of` at their own import time;
# reload() only replaces config.universe's own namespace, so those other
# modules keep pointing at pre-reload function/data objects — and a second
# "restore" reload doesn't fix that either, it just creates yet another
# orphaned object. That mismatch caused a real, hard-to-debug failure in
# test_data_harmonizer.py during development of this test. _load_universe_data
# above is the actual interesting logic (fallback/corrupt-file handling);
# "assigning its return value to module constants" is trivial Python
# semantics that doesn't need its own test.
