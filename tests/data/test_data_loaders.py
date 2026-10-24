import csv
import json

import cv2
import numpy as np
import pytest
import soundfile as sf

from dfd.data import loaders as L
from dfd.data.manifest import Record, write_manifest
from dfd.prep.pipeline import INDEX_COLUMNS, folder_name


def _rec(sid, modality, path, label, group, **kw):
    return Record(sample_id=sid, dataset="toy", family="toy", path=path, modality=modality, label=label,
                  method="real" if label == 0 else "fake", group_id=group, group_basis="source_video",
                  identity_known=False, **kw)


def _write_clip(crops, sid, detected, size=32):
    folder = crops / folder_name(sid)
    folder.mkdir(parents=True)
    rows = []
    for k in range(4):
        row = dict.fromkeys(INDEX_COLUMNS, "")
        row.update(sample_id=sid, frame_index=k, source_frame_index=k * 5, detected=int(k in detected), n_faces=1)
        if k in detected:
            img = np.full((size, size, 3), 10 * (k + 1), np.uint8)
            img[:, :, 2] = 200  # BGR red channel
            cv2.imwrite(str(folder / f"{k}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
            row["crop_path"] = f"{folder.name}/{k}.jpg"
        rows.append(row)
    with open(folder / "index.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, INDEX_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    (folder / "_done.json").write_text(json.dumps({"params": "p"}))


@pytest.fixture
def toy(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    crops = tmp_path / "crops"
    crops.mkdir()
    (crops / "params.json").write_text('{"n_frames": 4}')
    recs = [_rec("toy:a:1", "video", "v/a1.mp4", 0, "toy:g1"), _rec("toy:b:2", "video", "v/b2.mp4", 1, "toy:g2"),
            _rec("toy:c:3", "video", "v/c3.mp4", 1, "toy:g3"), _rec("toy:d:4", "video", "v/d4.mp4", 0, "toy:g4")]
    manifest = tmp_path / "m.csv"
    write_manifest(manifest, recs)
    split = tmp_path / "s.csv"
    split.write_text("sample_id,split\ntoy:a:1,train\ntoy:b:2,train\ntoy:c:3,val\ntoy:d:4,test_x\n")
    _write_clip(crops, "toy:a:1", {0, 1, 2, 3})
    _write_clip(crops, "toy:b:2", {1, 3})
    _write_clip(crops, "toy:c:3", set())
    _write_clip(crops, "toy:d:4", {0})
    return dict(data=data, crops=crops, manifest=manifest, split=split, recs=recs)


def test_select_records_filters_split_in_manifest_order(toy):
    r = L.select_records(toy["manifest"], toy["split"], "train")
    assert [x.sample_id for x in r] == ["toy:a:1", "toy:b:2"]


def test_test_splits_refused_unless_final(toy):
    with pytest.raises(L.LoaderError, match="final-only"):
        L.select_records(toy["manifest"], toy["split"], "test_x")
    assert len(L.select_records(toy["manifest"], toy["split"], "test_x", final=True)) == 1


def test_unknown_split_and_empty_split(toy):
    with pytest.raises(L.LoaderError):
        L.select_records(toy["manifest"], toy["split"], "bogus")
    with pytest.raises(L.LoaderError):
        L.select_records(toy["manifest"], toy["split"], "dev_x")


def test_face_clip_masks_missing_frames(toy):
    ds = L.FaceClipDataset(L.select_records(toy["manifest"], toy["split"], "train"), toy["crops"],
                           n_frames=4, size=32, data_root=toy["data"])
    assert len(ds) == 2 and ds.labels().tolist() == [0, 1]
    b = ds[1]
    assert b["x"].shape == (4, 3, 32, 32) and b["x"].dtype.is_floating_point is False
    assert b["mask"].tolist() == [False, True, False, True] and not b["no_face"]
    assert int(b["x"][0].sum()) == 0                      # masked frame is zero
    assert int(b["x"][1, 0].float().mean()) in range(195, 205)  # channel 0 is RGB red, written as BGR red
    assert (b["label"], b["group_id"], b["sample_id"]) == (1, "toy:g2", "toy:b:2")


def test_face_clip_no_face_flag_and_preprocess_id(toy):
    ds = L.FaceClipDataset(L.select_records(toy["manifest"], toy["split"], "val"), toy["crops"],
                           n_frames=4, size=32, data_root=toy["data"])
    assert ds[0]["no_face"] is True and not ds[0]["mask"].any()
    assert len(ds.preprocess_id) == 12 and ds.preprocess_id == L.preprocess_id(toy["crops"])


def test_face_clip_rejects_unfinished_crops(toy):
    (toy["crops"] / folder_name("toy:a:1") / "_done.json").unlink()
    with pytest.raises(L.LoaderError, match="no finished crops"):
        L.FaceClipDataset(L.select_records(toy["manifest"], toy["split"], "train"), toy["crops"], data_root=toy["data"])


def test_audio_fixed_length_centre_crop_and_padding(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    sr = 16000
    sf.write(data / "long.wav", np.arange(sr * 2, dtype=np.float32) / (sr * 2), sr)
    sf.write(data / "short.wav", np.full(sr // 2, 0.5, np.float32), sr)
    recs = [_rec("t:long", "audio", "long.wav", 0, "toy:g1"), _rec("t:short", "audio", "short.wav", 1, "toy:g2")]
    ds = L.AudioDataset(recs, seconds=1.0, data_root=data)
    long_, short = ds[0], ds[1]
    assert long_["x"].shape == (sr,) and long_["length"] == sr
    assert abs(float(long_["x"][0]) - 0.25) < 1e-3          # centre crop starts at 0.5 s of 2 s
    assert short["x"].shape == (sr,) and short["length"] == sr // 2 and float(short["x"][-1]) == 0.0


def test_audio_train_crop_is_seeded_per_epoch(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    sr = 16000
    sf.write(data / "a.wav", np.random.default_rng(0).uniform(-1, 1, sr * 3).astype(np.float32), sr)
    recs = [_rec("t:a", "audio", "a.wav", 0, "toy:g1")]
    ds = L.AudioDataset(recs, seconds=1.0, data_root=data, train_crop=True, seed=7)
    a1, a2 = ds[0]["x"].clone(), ds[0]["x"].clone()
    assert (a1 == a2).all()
    ds.set_epoch(1)
    assert not (ds[0]["x"] == a1).all()


def test_image_dataset_whole_image(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    cv2.imwrite(str(data / "i.png"), np.full((50, 80, 3), 90, np.uint8))
    ds = L.ImageDataset([_rec("t:i", "image", "i.png", 1, "toy:g")], size=64, data_root=data)
    assert ds[0]["x"].shape == (3, 64, 64) and int(ds[0]["x"].float().mean()) == 90
