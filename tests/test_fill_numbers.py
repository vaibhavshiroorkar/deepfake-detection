import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("fill_numbers", Path(__file__).parents[1] / "scripts" / "fill_numbers.py")
fill_numbers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fill_numbers)

NUMBERS = {"a.auc": {"value": 0.80912, "source": "x"}, "a.n": {"value": 20284, "source": "x"},
           "a.fpr": {"value": 0.267, "source": "x"}}


def test_placeholders_are_filled_with_styles() -> None:
    text = "AUC {{a.auc}} on {{a.n|int}} clips, {{a.fpr|pct}} false alarms"
    assert fill_numbers.fill(text, NUMBERS) == "AUC 0.809 on 20,284 clips, 27% false alarms"


def test_a_missing_number_stops_the_build() -> None:
    with pytest.raises(SystemExit, match="missing numbers: b.auc"):
        fill_numbers.fill("{{a.auc}} and {{b.auc}}", NUMBERS)
