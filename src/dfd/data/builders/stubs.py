"""Documented stubs: datasets that have not arrived. Layout NOT yet verified for every module function here.

Each stub raises ``LayoutNotVerified`` with the exact assumption. The name-parsing rules that the
documentation does give are implemented as pure functions (and unit-tested on synthetic strings), so the
builder can be finished quickly once the files can be inspected. Nothing here has been run on real files. (dfd and in_the_wild have real builders now: dfd.py, in_the_wild.py.)
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


from .dfd import parse_dfd_name  # noqa: F401  (kept here for older imports)



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
