"""Preprocessing hashes that label the same data, verified by comparison.

`preprocessing_config_hash` covers the view settings and the code version
together. That strictness is correct and has earned its place: a cache built by
different preprocessing code is a different cache even when the settings match,
and silently mixing the two feeds a model inputs it was not trained on.

It also means two runs that used identical settings and identical code under
different version strings get different hashes for byte-identical data. That
happened here. `runs/streams-20260905` cached LAV-DF under `streams-v1` and
`runs/program-20260906` cached everything else under `program-v1`, with the same
`ViewConfig`, which left the project's only fully audiovisual corpus unable to be
compared against any other.

Nothing in this module is asserted from the configs looking alike. Each entry was
established by loading the same clips from both caches and comparing every array
element by element, which `scripts/verify_cache_equivalence.py` does and records.
An entry without that evidence does not belong here.

The check is narrow on purpose. It answers "are these two labels the same data",
not "is this cache close enough", and it must never be widened into a tolerance.
"""

from __future__ import annotations

PROGRAM_V1 = "a6fe6c0d041538d9fda06a0898bec876f35ebb7f2e0090f025af29fe8f6df6c6"
STREAMS_V1 = "e20c01466223cb37df8f3502deaba13c883eca49c5e572ff9a042f80a4dcdc1a"

# Each group holds hashes proven to label identical views. The evidence file and
# the sample size are named so the claim can be rechecked rather than believed.
VERIFIED_EQUIVALENT: tuple[frozenset[str], ...] = (
    frozenset({PROGRAM_V1, STREAMS_V1}),
)

EVIDENCE = {
    frozenset({PROGRAM_V1, STREAMS_V1}): (
        "runs/cache-equivalence.json: 60 clips spread across the 1,099 shared "
        "between runs/streams-20260905/cache and runs/lavdf-av-20260914/cache, "
        "all five arrays identical, no missing arrays. The settings differ only "
        "in code_version, streams-v1 against program-v1."
    ),
}


def same_preprocessing(left: str, right: str) -> bool:
    """True when two hashes are equal or verified to label the same data."""
    if left == right:
        return True
    return any(
        left in group and right in group for group in VERIFIED_EQUIVALENT
    )


def evidence_for(left: str, right: str) -> str:
    """Why two different hashes are treated as one, for an error or a log."""
    for group in VERIFIED_EQUIVALENT:
        if left in group and right in group:
            return EVIDENCE.get(group, "verified, evidence not recorded")
    return ""
