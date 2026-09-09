"""Explainability: locked until Stage 10.

Dimmed and unclickable in the sidebar. There is nothing to explain until a
trained model exists, so the views are listed rather than stubbed as dead buttons.
"""

import streamlit as st

from deepfake_detection.dashboard.lib import locked
from deepfake_detection.dashboard.lib.stream_spec import EXPLAINABILITY

locked.render(st, EXPLAINABILITY)
