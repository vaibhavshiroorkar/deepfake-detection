"""Evaluation Lab backend (D-088): re-analysis of recorded score files on the dfd.eval safeguards.

It does not run models, train, write the ledger, write checkpoints or touch the approved heads or thresholds
of the product. Everything it writes lives under data/_evallab/.
"""

from .api import create_blueprint, register  # noqa: F401
from .lab import EvalLab  # noqa: F401
