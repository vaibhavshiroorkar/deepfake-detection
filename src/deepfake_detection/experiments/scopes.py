"""What a run's numbers are allowed to be used for.

`evidence_scope` decides whether a metric can be quoted as a result, and until
now it was a free-form string. Six different values had appeared across YAML
configs, a hardcoded constant in smoke.py, and two ad-hoc evaluation scripts,
with the only validation anywhere being a single literal comparison in the
dashboard. That is how a software-fixture AUC of 1.0 ends up one copy-paste away
from looking like a research result.

The vocabulary is ordered from "proves nothing" to "final external evidence".
"""

from __future__ import annotations

# A deterministic fixture with no real data behind it. Never a result.
SOFTWARE_FIXTURE_ONLY = "software_fixture_only"
# A dry run that proves the pipeline executes, typically one epoch. Not a result.
PROTOTYPE_ONLY = "prototype_only"
# The current frozen baseline the Evidence gate serves.
DEVELOPMENT_BASELINE = "development_baseline"
# A candidate trained to be compared against the baseline under one protocol.
DEVELOPMENT_COMPARISON = "development_comparison"
# In-domain validation metrics for a trained candidate.
DEVELOPMENT_VALIDATION = "development_validation"
# In-domain metrics on the held-out test partition, read after selection.
DEVELOPMENT_TEST = "development_test"
# Cross-dataset zero-shot metrics on Celeb-DF-v2.
GENERALIZATION_CELEBDF = "generalization_celebdf"
# The locked external benchmark. Read once, after models and thresholds freeze.
EXTERNAL_MNW = "external_mnw"

EVIDENCE_SCOPES = (
    SOFTWARE_FIXTURE_ONLY,
    PROTOTYPE_ONLY,
    DEVELOPMENT_BASELINE,
    DEVELOPMENT_COMPARISON,
    DEVELOPMENT_VALIDATION,
    DEVELOPMENT_TEST,
    GENERALIZATION_CELEBDF,
    EXTERNAL_MNW,
)

# Scopes whose metrics describe a dataset the model was never trained on.
GENERALIZATION_SCOPES = frozenset({GENERALIZATION_CELEBDF, EXTERNAL_MNW})

# Scopes that must never be quoted as evidence about real-world performance.
NON_EVIDENCE_SCOPES = frozenset({SOFTWARE_FIXTURE_ONLY, PROTOTYPE_ONLY})


def validate_evidence_scope(value: str) -> str:
    """Reject a scope outside the vocabulary, naming what was allowed."""
    if value not in EVIDENCE_SCOPES:
        raise ValueError(
            f"Unknown evidence_scope {value!r}. Expected one of: "
            f"{', '.join(EVIDENCE_SCOPES)}."
        )
    return value


def is_generalization(value: str) -> bool:
    """Whether this scope's metrics come from an unseen dataset."""
    return value in GENERALIZATION_SCOPES
