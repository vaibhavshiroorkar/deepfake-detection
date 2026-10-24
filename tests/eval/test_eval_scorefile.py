"""Score file reading and validation."""

import numpy as np
import pytest

from dfd.eval.calibration import logit
from dfd.eval.scorefile import ScoreFileError, read_scores, write_scores

BASE = dict(run_id=["r"] * 4, iteration=["I-0"] * 4, seed=[1] * 4, modality=["visual"] * 4,
            dataset=["d"] * 4, split=["val"] * 4, sample_id=["a", "b", "c", "d"],
            group_id=["g1", "g1", "g2", "g2"], label=[0, 1, 0, 1], score=[0.1, 0.9, 0.2, 0.8])


def write(tmp_path, **over):
    cols = {**BASE, **over}
    p = tmp_path / "s.csv"
    write_scores(p, cols)
    return p


def test_reads_valid_file_and_fills_logit(tmp_path):
    t = read_scores(write(tmp_path))
    assert len(t) == 4 and len(t.sha256) == 64
    assert np.allclose(t.logit, logit(t.score))
    assert t.label.tolist() == [0, 1, 0, 1]


def test_given_logit_is_used_and_blank_filled(tmp_path):
    t = read_scores(write(tmp_path, logit=["-2", "", "1.5", ""]))
    assert t.logit[0] == -2.0 and t.logit[2] == 1.5
    assert t.logit[1] == pytest.approx(logit(0.9))
    assert t.n_logit_given == 2
    assert any("logit" in w for w in t.warnings)


def test_missing_column_and_bad_values(tmp_path):
    cols = {k: v for k, v in BASE.items() if k != "group_id"}
    p = tmp_path / "s.csv"
    write_scores(p, cols)
    with pytest.raises(ScoreFileError, match="group_id"):
        read_scores(p)
    for bad in (dict(label=[0, 1, 2, 1]), dict(score=[0.1, 1.2, 0.2, 0.8]), dict(score=[0.1, "x", 0.2, 0.8]),
                dict(split=["val", "val", "valid", "val"]), dict(modality=["video"] * 4),
                dict(group_id=["g1", "", "g2", "g2"]), dict(seed=[1, 1, 1.5, 1]),
                dict(score=[0.1, "nan", 0.2, 0.8])):
        with pytest.raises(ScoreFileError):
            read_scores(write(tmp_path, **bad))


def test_run_has_one_seed_and_one_iteration(tmp_path):
    with pytest.raises(ScoreFileError, match="seed"):
        read_scores(write(tmp_path, seed=[1, 1, 2, 2]))
    with pytest.raises(ScoreFileError, match="iteration"):
        read_scores(write(tmp_path, iteration=["I-0", "I-0", "I-1", "I-1"]))


def test_frames_are_aggregated_by_mean(tmp_path):
    cols = dict(BASE)
    cols = {k: v + [v[0]] for k, v in cols.items()}  # a second frame of sample "a"
    cols["score"] = [0.1, 0.9, 0.2, 0.8, 0.3]
    p = tmp_path / "f.csv"
    write_scores(p, cols)
    t = read_scores(p)
    assert len(t) == 4 and t.n_frame_rows == 1
    assert t.score[0] == pytest.approx(0.2)
    assert t.logit[0] == pytest.approx(logit(0.2))
    cols["label"] = [0, 1, 0, 1, 1]
    write_scores(p, cols)
    with pytest.raises(ScoreFileError, match="labels"):
        read_scores(p)


def test_empty_and_header_only(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ScoreFileError):
        read_scores(p)
    p.write_text(",".join(BASE) + "\n", encoding="utf-8")
    with pytest.raises(ScoreFileError, match="no data"):
        read_scores(p)
