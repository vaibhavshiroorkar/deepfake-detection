"""Lip-sync stream: the trained model on one clip, stage by stage.

Runs the real checkpoint rather than the random projections this page used to
show. See the dashboard's lib/cross_modal_page.py.
"""

import streamlit as st

from deepfake_detection.dashboard.lib import cross_modal_page

cross_modal_page.render(st, "lipsync")
