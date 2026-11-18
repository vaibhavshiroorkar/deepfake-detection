import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "run_a16", Path(__file__).parents[1] / "scripts" / "run_a16.py"
)
a16 = importlib.util.module_from_spec(spec)
sys.modules["run_a16"] = a16
spec.loader.exec_module(a16)


def test_every_swapped_audio_comes_from_another_person() -> None:
    sources = ["a", "a", "b", "c", "c", "d", "e", "e"]
    order = a16.derangement(sources)
    assert sorted(order) == list(range(len(sources)))  # each audio used once
    assert all(sources[i] != sources[j] for i, j in enumerate(order))
    assert order == a16.derangement(sources)  # fixed by the seed


def test_mean_auc_is_one_when_every_moved_window_scores_higher() -> None:
    pairs = [a16._Pair(f"id{i}", [0.1 * i] * 3, [0.1 * i + 5] * 3) for i in range(5)]
    assert a16.mean_auc(pairs) == 1.0
