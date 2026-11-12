"""Every number behind a verdict rule equals its stored result (D-087). The result files are git-ignored under data/; a missing file
skips its check (the skip names the file), it never passes silently."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from dfd.platform import verdict
from dfd.platform.verdict_evidence import EVIDENCE, resolve_path

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("eid", sorted(EVIDENCE))
def test_registry_value_equals_the_result_file(eid):
    e = EVIDENCE[eid]
    f = ROOT / e["file"]
    if not f.exists():
        pytest.skip(f"result file not present on this machine: {e['file']}")
    got = resolve_path(json.loads(f.read_text(encoding="utf-8")), e["path"])
    assert abs(float(got) - e["value"]) < 1e-4, (eid, got, e["value"])


@pytest.mark.parametrize("eid", sorted(EVIDENCE))
def test_percent_in_the_text_is_the_stored_value_rounded(eid):
    """The first percentage written in a sentence is the stored rate (or its complement for 'caught' or specificity sentences)."""
    e = EVIDENCE[eid]
    m = re.search(r"(\d+(?:\.\d+)?) percent", e["text"])
    assert m, eid
    shown = float(m.group(1))
    v = e["value"] * 100
    assert min(abs(shown - v), abs(shown - (100 - v))) <= 0.6, (eid, shown, v)


def test_every_profile_cites_known_evidence():
    for key, prof in verdict._PROFILES.items():
        for eid in prof.get("evidence", []):
            assert eid in EVIDENCE, (key, eid)
