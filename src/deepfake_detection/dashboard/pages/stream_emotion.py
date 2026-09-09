"""Emotion stream: vocal affect against facial expression. See the dashboard's lib/cross_modal.py."""

import streamlit as st

from deepfake_detection.dashboard.lib import cross_modal
from deepfake_detection.dashboard.lib.stream_spec import EMOTION_STREAM

cross_modal.render(st, EMOTION_STREAM, key="emotion", video_source="faces")
