"""Explainability: locked until Stage 10.

Dimmed and unclickable in the sidebar. There is nothing to explain until a
trained model exists, so the views are listed rather than stubbed as dead buttons.
"""
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from dashboard.lib import locked
from dashboard.lib.stream_spec import EXPLAINABILITY

locked.render(st, EXPLAINABILITY)
