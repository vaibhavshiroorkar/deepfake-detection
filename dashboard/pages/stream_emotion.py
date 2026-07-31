"""Emotion stream: vocal affect against facial expression. See dashboard/lib/cross_modal.py."""
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from dashboard.lib import cross_modal
from dashboard.lib.stream_spec import EMOTION_STREAM

cross_modal.render(st, EMOTION_STREAM, key="emotion", video_source="faces")
