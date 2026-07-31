"""Lip-sync stream: audio against mouth motion. See dashboard/lib/cross_modal.py."""
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from dashboard.lib import cross_modal
from dashboard.lib.stream_spec import LIPSYNC_STREAM

cross_modal.render(st, LIPSYNC_STREAM, key="lipsync", video_source="mouths")
