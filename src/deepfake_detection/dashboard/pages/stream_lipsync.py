"""Lip-sync stream: audio against mouth motion. See the dashboard's lib/cross_modal.py."""

import streamlit as st

from deepfake_detection.dashboard.lib import cross_modal
from deepfake_detection.dashboard.lib.stream_spec import LIPSYNC_STREAM

cross_modal.render(st, LIPSYNC_STREAM, key="lipsync", video_source="mouths")
