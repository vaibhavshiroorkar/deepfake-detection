"""Shared Signal dataclass used by every detector.

Each detector function returns a Signal: a name, a 0..1 suspicion score,
and a human-readable detail string. Detail strings follow the convention
of "technical numbers. plain-language takeaway." so the frontend can
split them into a bold summary and a smaller mono technical line.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Signal:
    name: str
    score: float  # 0..1, higher means more suspicious
    detail: str
