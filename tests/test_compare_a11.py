"""compare_a11.py: each A11 model against its P-001 path, paired on the same clips."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_a11.py"
HEADER = "clip_id,label,probability,source,manipulation_type,method\n"


def _module():
    spec = importlib.util.spec_from_file_location("compare_a11", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + "".join(f"{c},{y},{p},id{c},t,m\n" for c, y, p in rows))


def test_pairs_by_clip_id_whatever_the_row_order(tmp_path):
    module = _module()
    new = [("a", 0, 0.1), ("b", 0, 0.2), ("c", 1, 0.9), ("d", 1, 0.8)]
    old = [("d", 1, 0.1), ("c", 1, 0.3), ("b", 0, 0.2), ("a", 0, 0.4)]
    _write(tmp_path / "new" / "clip-seed17-dfdc-predictions.csv", new)
    _write(tmp_path / "old" / "visual-seed17-dfdc-predictions.csv", old)
    result = module.compare_part(
        tmp_path / "new", tmp_path / "old", "clip", "visual", "dfdc", samples=50
    )
    assert result["seeds"] == [17]
    assert result["auc_new"] == [1.0]
    assert result["auc_old"] == [0.25]
    assert abs(result["difference"]["estimate"] - 0.75) < 1e-9


def test_one_class_set_is_skipped_with_a_reason(tmp_path):
    module = _module()
    rows = [("a", 0, 0.1), ("b", 0, 0.2)]
    _write(tmp_path / "new" / "xlsr-seed17-dfdc-predictions.csv", rows)
    _write(tmp_path / "old" / "audio-seed17-dfdc-predictions.csv", rows)
    result = module.compare_part(
        tmp_path / "new", tmp_path / "old", "xlsr", "audio", "dfdc", samples=50
    )
    assert result == {"skipped": "one class in the set"}


def test_a_seed_with_no_new_model_is_left_out(tmp_path):
    module = _module()
    rows = [("a", 0, 0.1), ("b", 1, 0.9)]
    for seed in (17, 29):
        _write(tmp_path / "old" / f"visual-seed{seed}-dfdc-predictions.csv", rows)
    _write(tmp_path / "new" / "clip-seed17-dfdc-predictions.csv", rows)
    result = module.compare_part(
        tmp_path / "new", tmp_path / "old", "clip", "visual", "dfdc", samples=50
    )
    assert result["seeds"] == [17]
