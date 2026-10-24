"""Documented stubs: datasets that have not arrived. Layout NOT yet verified for every module function here.

Each stub raises ``LayoutNotVerified`` with the exact assumption. The name-parsing rules that the
documentation does give are implemented as pure functions (and unit-tested on synthetic strings), so the
builder can be finished quickly once the files can be inspected. Nothing here has been run on real files.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..manifest import Record
from . import LayoutNotVerified

VOX_ID_RE = re.compile(r"id\d{5}")


def extract_voxceleb_ids(text: str) -> list[str]:
    """VoxCeleb2 speaker ids (idNNNNN) found in a path or a metadata string, in order, without repeats."""
    seen: list[str] = []
    for m in VOX_ID_RE.findall(text):
        if m not in seen:
            seen.append(m)
    return seen


_DFD_ORIG_RE = re.compile(r"^(?P<actor>\d{2})__(?P<scene>[^_].*?)\.mp4$")
_DFD_MANIP_RE = re.compile(r"^(?P<target>\d{2})_(?P<source>\d{2})__(?P<seq>.+?)__(?P<exp>[A-Za-z0-9]+)\.mp4$")


def parse_dfd_name(filename: str) -> dict:
    """DFD names (DATA_LOG.md): original ``<actor>__<scene>.mp4``; manipulated
    ``<target actor>_<source actor>__<sequence>__<experiment id>.mp4``.

    Returns dict(label, target, source, scene_or_sequence, experiment). Raises ValueError. The numeric
    two-digit actor format is an assumption taken from the documented pattern, not seen on disk.
    """
    m = _DFD_MANIP_RE.match(filename)
    if m:
        return {"label": 1, "target": m["target"], "source": m["source"], "scene_or_sequence": m["seq"],
                "experiment": m["exp"]}
    m = _DFD_ORIG_RE.match(filename)
    if m:
        return {"label": 0, "target": m["actor"], "source": None, "scene_or_sequence": m["scene"],
                "experiment": None}
    raise ValueError(f"not a DFD file name: {filename}")


def build_dfd(data_root: Path) -> tuple[list[Record], dict]:
    raise LayoutNotVerified(
        "dfd: layout not yet verified. Assumption (DATA_LOG.md): data/dfd/original_sequences/actors/c23/videos/"
        "<actor>__<scene>.mp4 (363 files) and data/dfd/manipulated_sequences/DeepFakeDetection/c23/videos/"
        "<target>_<source>__<sequence>__<experiment id>.mp4 (3,068 files); family 'dfd'; group = the "
        "target actor (identity_known true, 28 actors), source actor kept in notes; test_x only."
    )


def build_in_the_wild(data_root: Path) -> tuple[list[Record], dict]:
    raise LayoutNotVerified(
        "in_the_wild: layout not yet verified. Assumption: data/in_the_wild/release_in_the_wild/*.wav plus "
        "meta.csv with columns file, speaker, label (bona-fide / spoof) as in the dataset card; group = "
        "speaker (58 speakers, identity_known true), family 'in_the_wild', test_x only. Not yet downloaded."
    )


def build_lavdf(data_root: Path) -> tuple[list[Record], dict]:
    raise LayoutNotVerified(
        "lavdf: layout not yet verified. Assumption: data/lavdf/ holds the extracted LAV-DF.tar with a "
        "metadata JSON listing per video: file, split (train/dev/test), speaker/identity, modify_video, "
        "modify_audio; group = VoxCeleb2 identity (idNNNNN, 153 identities, read from the metadata or path via "
        "extract_voxceleb_ids), identity_known true; official_split from the metadata (91 / 31 / 31 "
        "identities). Family 'lavdf', flagged as a POSSIBLE shared family with 'fakeavceleb' (both use "
        "VoxCeleb2 real videos); see dfd.data.leakage.POSSIBLE_SHARED_FAMILIES."
    )


def build_fakeavceleb(data_root: Path) -> tuple[list[Record], dict]:
    raise LayoutNotVerified(
        "fakeavceleb: layout not yet verified. Assumption: data/fakeavceleb/ holds category folders "
        "(RealVideo-RealAudio, RealVideo-FakeAudio, FakeVideo-RealAudio, FakeVideo-FakeAudio) with paths "
        "containing the VoxCeleb2 id (idNNNNN) of the source speaker, plus a metadata CSV (20,000 rows); "
        "group = that id via extract_voxceleb_ids, identity_known true; label = fake if either the video or "
        "the audio is fake (per-modality labels would need extra columns, not in the schema); family "
        "'fakeavceleb', possible shared family with 'lavdf'. Not yet downloaded."
    )


def build_wilddeepfake(data_root: Path) -> tuple[list[Record], dict]:
    raise LayoutNotVerified(
        "wilddeepfake: layout not yet verified. Assumption: data/wilddeepfake/ holds the extracted "
        "fake_test and real_test tar.gz shards, each a face-sequence folder of images (806 sequences in "
        "total); sample = one sequence (modality image, path = the sequence folder), group = the sequence "
        "(no identity metadata, identity_known false, group_basis source_video), family 'wilddeepfake', "
        "test_x only. Not yet downloaded."
    )
