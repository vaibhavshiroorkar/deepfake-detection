"""The method figure: every label stays inside its box."""

import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "make_figures", Path(__file__).parents[1] / "scripts" / "make_figures.py"
)
figures = importlib.util.module_from_spec(spec)
sys.modules["make_figures"] = figures
spec.loader.exec_module(figures)


def test_no_label_of_the_method_figure_leaves_its_box() -> None:
    fig, pairs = figures.draw_pipeline()
    assert pairs, "the figure registers its labels with their boxes"
    assert figures.overflowing(fig, pairs) == []
