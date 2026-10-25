"""FakeAVCeleb builder helpers and the identity-separated split of D-070 (synthetic data, no archive needed)."""
import random

from dfd.data.builders.fakeavceleb import member_path, target_ids
from dfd.data.fakeavceleb_split import DEV, SALT, SEALED, assign, half_of, identity_cells, verify
from dfd.data.manifest import Record


def test_target_ids_ignores_non_identity_tokens_and_source():
    assert target_ids("id00076", "00109_10_id00476_wavtolip.mp4", "10", "-") == ["id00476"]
    assert target_ids("id00076", "00109_9_id00076_wavtolip.mp4", "9", "-") == []          # self-target (Wav2Lip on the same person)
    assert target_ids("id00001", "00001_fake.mp4", "fake.mp4", "faceswap") == []
    assert target_ids("id00001", "x_id00002.mp4", "id00003", "id00002") == ["id00002", "id00003"]


def test_member_path_maps_metadata_prefix():
    assert member_path("FakeAVCeleb/RealVideo-RealAudio/African/men/id00076", "00109.mp4") == "RealVideo-RealAudio/African/men/id00076/00109.mp4"


def make(n_per_cell=20, cells=("A/men", "B/women"), seed=0):
    rnd = random.Random(seed)
    ids = {c: [f"id{ci * 1000 + i:05d}" for i in range(n_per_cell)] for ci, c in enumerate(cells)}
    recs = []
    for c, lst in ids.items():
        for s in lst:
            recs.append(Record(f"d:{s}/real", "d", "d", f"d/{s}/real.mp4", "video", 0, "real", f"d:{s}", "identity", True, "", f"category=RVRA;ids={s};cell={c}"))
            for t in rnd.sample([x for x in lst if x != s], 4):
                recs.append(Record(f"d:{s}/{t}", "d", "d", f"d/{s}/{t}.mp4", "video", 1, "fsgan", f"d:{s}", "identity", True, "",
                                   f"category=FVRA;ids={s}|{t};cell={c}"))
    return recs


def test_halves_balanced_inside_each_cell_and_deterministic():
    recs = make()
    h = half_of(identity_cells(recs))
    for c, lst in identity_cells(recs).items():
        assert sum(h[i] for i in lst) == len(lst) // 2
    assert h == half_of(identity_cells(make()))
    assert half_of(identity_cells(recs), salt="other") != h


def test_split_never_shares_an_identity_and_excludes_cross_clips():
    recs = make()
    kept, excluded, half = assign(recs)
    assert verify(recs, kept)["ok"]
    assert len(kept) + len(excluded) == len(recs)
    by = {r.sample_id: r for r in recs}
    for sid, side in kept.items():
        ids = by[sid].notes.split("ids=")[1].split(";")[0].split("|")
        assert {half[i] for i in ids} == ({1} if side == SEALED else {0})
    assert {SEALED, DEV} == set(kept.values())
    assert all(e["reason"] == "source and target identities on different sides" for e in excluded) and excluded


def test_verify_detects_a_shared_identity():
    recs = make()
    kept, _, _ = assign(recs)
    bad = dict(kept)
    from collections import Counter
    by = {r.sample_id: r for r in recs}
    src = lambda sid: by[sid].group_id  # noqa: E731
    n = Counter(src(s) for s, v in kept.items() if v == DEV)
    victim = next(s for s, v in kept.items() if v == DEV and n[src(s)] >= 2)    # its source identity also stays in dev through another clip
    bad[victim] = SEALED
    assert not verify(recs, bad)["ok"]


def test_salt_is_the_documented_one():
    assert SALT == "FakeAVCeleb-E015-2026-10-26"
