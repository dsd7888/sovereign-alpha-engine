import pytest

from sovereign_alpha.utils.helpers import is_cache_valid, load_json, save_json
from sovereign_alpha.utils.validators import (
    clean_ticker_list,
    is_finite_number,
    is_valid_price,
    normalise_weights,
)


class TestJsonRoundTrip:
    def test_save_then_load_roundtrips(self, tmp_path):
        path = tmp_path / "sub" / "state.json"
        data = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}
        save_json(data, path)
        assert load_json(path) == data

    def test_load_missing_file_returns_empty_dict(self, tmp_path):
        assert load_json(tmp_path / "does_not_exist.json") == {}

    def test_load_corrupt_json_returns_empty_dict_not_raise(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("{not valid json", encoding="utf-8")
        assert load_json(path) == {}

    def test_save_leaves_no_temp_file_behind(self, tmp_path):
        target_dir = tmp_path / "state_dir"
        path = target_dir / "state.json"
        save_json({"x": 1}, path)
        assert [p.name for p in target_dir.iterdir()] == ["state.json"]


class TestCacheValidity:
    def test_missing_file_is_invalid(self, tmp_path):
        assert is_cache_valid(tmp_path / "nope.json", max_age_hours=1) is False

    def test_fresh_file_is_valid(self, tmp_path):
        path = tmp_path / "fresh.json"
        path.write_text("{}", encoding="utf-8")
        assert is_cache_valid(path, max_age_hours=1) is True


class TestCleanTickerList:
    def test_dedupes_preserving_order(self):
        assert clean_ticker_list(["A.NS", "b.ns", "A.NS"]) == ["A.NS", "B.NS"]

    def test_strips_whitespace_and_drops_empties(self):
        assert clean_ticker_list([" A.NS ", "", "  "]) == ["A.NS"]


class TestPriceValidation:
    @pytest.mark.parametrize("price,expected", [
        (100.0, True), (0.0, False), (-5.0, False), (None, False), (float("nan"), False), (float("inf"), False),
    ])
    def test_is_valid_price(self, price, expected):
        assert is_valid_price(price) is expected


class TestFiniteNumber:
    def test_rejects_none_and_nan_and_strings(self):
        assert is_finite_number(None) is False
        assert is_finite_number(float("nan")) is False
        assert is_finite_number("not a number") is False

    def test_accepts_ints_and_floats(self):
        assert is_finite_number(5) is True
        assert is_finite_number(5.5) is True


class TestNormaliseWeights:
    def test_already_normalised_returned_unchanged(self):
        w = {"A": 0.5, "B": 0.5}
        assert normalise_weights(w) == w

    def test_rescales_to_sum_one(self):
        result = normalise_weights({"A": 0.2, "B": 0.2})
        assert abs(sum(result.values()) - 1.0) < 1e-9
        assert abs(result["A"] - 0.5) < 1e-9

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            normalise_weights({})

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            normalise_weights({"A": -0.1, "B": 1.1})

    def test_all_zero_raises(self):
        with pytest.raises(ValueError):
            normalise_weights({"A": 0.0, "B": 0.0})
