import csv
from pathlib import Path

import cv2
import pytest

from dfd.prep import face as F
from dfd.prep import pipeline as P
from prep_test_helpers import frame_number, make_video


class FakeDetector:
    """Finds a 20x20 face at (22, 14) (plus a tiny one) in every frame whose source index is not a multiple of 3."""
    calls = 0

    def detect(self, frame):
        FakeDetector.calls += 1
        n = frame_number(frame)
        if n % 3 == 0:
            return []
        return [F.Face(22.0, 14.0, 20.0, 20.0, 0.9), F.Face(0.0, 0.0, 5.0, 5.0, 0.5)]


@pytest.fixture
def setup(tmp_path, ffmpeg_ok, monkeypatch):
    FakeDetector.calls = 0
    monkeypatch.setattr(P, "_get_detector", lambda p: FakeDetector())
    vids = []
    for k in range(3):
        make_video(tmp_path / f"v{k}.mp4", n_frames=40)
        vids.append(tmp_path / f"v{k}.mp4")
    (tmp_path / "x.mp4.part").write_bytes(b"half")
    man = tmp_path / "m.csv"
    with open(man, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "path", "label", "group_id"])
        for k, v in enumerate(vids):
            w.writerow([f"s{k}", str(v), k % 2, f"g{k}"])
        w.writerow(["partial", str(tmp_path / "x.mp4.part"), 0, "gp"])
        (tmp_path / "junk.mp4").write_bytes(b"not a video")
        w.writerow(["broken", str(tmp_path / "junk.mp4"), 0, "gb"])
    return tmp_path, man


PARAMS = P.Params(n_frames=8, size=32, min_age_s=0.0)


def read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_index_csv_contents_and_crops(setup):
    tmp, man = setup
    out = tmp / "crops"
    summ = P.run(man, out, workers=0, params=PARAMS, quiet=True, root=".")
    assert summ["processed_ok"] == 3
    assert summ["errors"] == 1 and summ["skipped_unstable"] == 1  # broken -> error, .part -> unstable
    idx = read(out / "index.csv")
    assert list(idx[0].keys()) == P.INDEX_COLUMNS
    assert len(idx) == 3 * 8                      # every sampled frame has a row, detected or not
    wanted = [int((i + 0.5) * 40 // 8) for i in range(8)]
    rows0 = [r for r in idx if r["sample_id"] == "s0"]
    assert [int(r["frame_index"]) for r in rows0] == list(range(8))
    assert [int(r["source_frame_index"]) for r in rows0] == wanted
    for r in rows0:
        src = int(r["source_frame_index"])
        if src % 3 == 0:                           # no detection: recorded, not dropped
            assert r["detected"] == "0" and r["box"] == "" and r["crop_path"] == ""
        else:
            assert r["detected"] == "1" and r["n_faces"] == "2"
            assert r["box"].startswith("22.0,14.0,20.0,20.0")
            assert r["crop_box"] == "19,11,26"     # centre (32,24), side round(1.3*20)=26
            img = cv2.imread(str(out / r["crop_path"]))
            assert img.shape == (32, 32, 3)
            assert r["crop_path"] == f"s0/{r['frame_index']}.jpg"
    assert any(r["detected"] == "0" for r in idx) and any(r["detected"] == "1" for r in idx)
    status = {r["sample_id"]: r["status"] for r in read(out / "videos.csv")}
    assert status["s0"] == "ok" and status["partial"] == "skipped_unstable" and status["broken"] == "error"


def test_resume_skips_finished_videos(setup):
    tmp, man = setup
    out = tmp / "crops"
    P.run(man, out, workers=0, params=PARAMS, quiet=True, root=".")
    first_calls = FakeDetector.calls
    assert first_calls > 0
    s2 = P.run(man, out, workers=0, params=PARAMS, quiet=True, root=".")
    assert s2["skipped_already_done"] == 3 and s2["processed_ok"] == 0
    assert FakeDetector.calls == first_calls       # finished videos did no detector work
    assert len(read(out / "index.csv")) == 24      # index still covers the earlier videos
    # an interrupted video (no _done.json) is redone alone
    (out / "s1" / "_done.json").unlink()
    before = FakeDetector.calls
    s3 = P.run(man, out, workers=0, params=PARAMS, quiet=True, root=".")
    assert s3["processed_ok"] == 1 and FakeDetector.calls - before == 8
    assert len(read(out / "index.csv")) == 24


def test_changed_settings_need_a_new_output_folder(setup):
    tmp, man = setup
    out = tmp / "crops"
    P.run(man, out, workers=0, params=PARAMS, quiet=True, root=".")
    other = P.Params(n_frames=8, size=64, min_age_s=0.0)
    with pytest.raises(SystemExit):
        P.run(man, out, workers=0, params=other, quiet=True, root=".")


def test_recent_file_is_skipped_by_min_age(setup):
    tmp, man = setup
    p = P.Params(n_frames=8, size=32, min_age_s=3600.0)   # everything was just written
    s = P.run(man, tmp / "c2", workers=0, params=p, quiet=True, root=".")
    assert s["processed_ok"] == 0 and s["skipped_unstable"] >= 3


def test_manifest_needs_columns_and_unique_ids(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("id,file\n1,a\n")
    with pytest.raises(SystemExit):
        P.read_manifest(bad)
    dup = tmp_path / "dup.csv"
    dup.write_text("sample_id,path\na,x\na,y\n")
    with pytest.raises(SystemExit):
        P.read_manifest(dup)
    assert P.folder_name("ffpp:Deepfakes:123_456") == "ffpp_Deepfakes_123_456"
    assert P.folder_name("a/b:c*d") == "a_b_c_d"
    with pytest.raises(ValueError):
        P.folder_name("..")
    clash = tmp_path / "clash.csv"
    clash.write_text("sample_id,path\nx:1,a\nx_1,b\n")
    with pytest.raises(SystemExit):
        P.read_manifest(clash)
    mixed = tmp_path / "mixed.csv"
    mixed.write_text("sample_id,path,modality\nv,a.mp4,video\nau,b.flac,audio\n")
    assert P.read_manifest(mixed, "R") == [("v", str(Path("R") / "a.mp4"))]
