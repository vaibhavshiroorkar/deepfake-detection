"""
Text forensics. Detect machine-generated prose via statistical
signatures: burstiness (sentence-length variance), lexical diversity,
function-word rhythm, punctuation entropy, and repetition.

These are the signals that originally motivated tools like GLTR —
LLMs produce prose that is unusually uniform in surface statistics.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Signal:
    name: str
    score: float
    detail: str


_SENT_SPLIT = re.compile(r"(?<=[\.!?])\s+(?=[A-Z\"'\(])")
_WORD = re.compile(r"[A-Za-z']+")

# High-frequency function words — LLMs lean on these
FUNCTION_WORDS = {
    "the", "of", "and", "to", "a", "in", "is", "it", "that", "this",
    "for", "as", "with", "was", "on", "by", "an", "be", "are", "or",
    "which", "but", "however", "moreover", "furthermore", "therefore",
    "additionally", "consequently", "ultimately", "essentially",
}

# Words and phrases LLMs overuse
LLM_TICS = [
    "delve", "delves", "delving",
    "tapestry", "leverage", "leveraging",
    "in the realm of", "in the world of", "in today's",
    "it is important to note", "it is worth noting", "it's worth noting",
    "navigate the", "landscape of",
    "a testament to", "plays a crucial role", "plays a pivotal role",
    "unlock", "unleash",
    "seamless", "seamlessly",
    "robust", "dynamic",
    "ever-evolving", "ever-changing",
    "furthermore,", "moreover,", "additionally,",
]


def _sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def _burstiness_signal(sents: list[str]) -> Signal:
    """Burstiness = variance of sentence lengths. Human writing swings
    between short and long sentences; LLMs tend toward uniform length."""
    if len(sents) < 3:
        return Signal(
            "Burstiness",
            0.3,
            "Too few sentences for a reliable burstiness reading.",
        )
    lengths = np.array([len(_WORD.findall(s)) for s in sents], dtype=np.float32)
    if lengths.mean() == 0:
        return Signal("Burstiness", 0.5, "No words found.")
    cv = float(lengths.std() / lengths.mean())  # coefficient of variation
    # Human prose often lands cv ~0.5–1.0. LLM prose trends lower.
    if cv < 0.25:
        score = 0.85
    elif cv < 0.45:
        score = 0.6
    elif cv < 0.7:
        score = 0.3
    else:
        score = 0.15
    return Signal(
        "Burstiness",
        score,
        f"Coefficient of variation across sentence lengths: {cv:.2f}. "
        + (
            "Uniform sentence rhythm — a signature of LLM drafting."
            if score > 0.5
            else "Sentence length varies freely — closer to how people write."
        ),
    )


def _lexical_signal(words: list[str]) -> Signal:
    """Type–token ratio and function-word share. LLMs tend to produce a
    slightly lower TTR than humans at equal length, and lean more
    heavily on function words."""
    if len(words) < 40:
        return Signal("Lexical rhythm", 0.3, "Text too short to measure vocabulary diversity.")
    ttr = len(set(w.lower() for w in words)) / len(words)
    fw_share = sum(1 for w in words if w.lower() in FUNCTION_WORDS) / len(words)
    # Heuristic zones
    score = 0.0
    if ttr < 0.35:
        score += 0.45
    elif ttr < 0.45:
        score += 0.2
    if fw_share > 0.42:
        score += 0.4
    elif fw_share > 0.38:
        score += 0.15
    score = float(np.clip(score, 0.0, 1.0))
    return Signal(
        "Lexical rhythm",
        score,
        f"Type–token ratio {ttr:.2f}, function-word share {fw_share:.2f}. "
        + (
            "Vocabulary is unusually even and function-word heavy."
            if score > 0.45
            else "Word choice varies and drifts, as natural writing does."
        ),
    )


def _repetition_signal(sents: list[str], words: list[str]) -> Signal:
    """N-gram repetition. LLMs often repeat scaffolding phrases, and
    overuse a small set of tics."""
    if len(words) < 30:
        return Signal("Phrase repetition", 0.2, "Not enough text to check.")
    lower = " ".join(w.lower() for w in words)
    tic_hits = sum(lower.count(t) for t in LLM_TICS)
    # trigram repetition
    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    counts = Counter(trigrams)
    repeats = sum(c - 1 for c in counts.values() if c > 1)
    repeat_rate = repeats / max(1, len(trigrams))
    tic_rate = tic_hits / max(1, len(words) / 100)  # per 100 words

    score = 0.0
    score += min(1.0, tic_rate / 1.5) * 0.6
    score += min(1.0, repeat_rate * 20) * 0.4
    return Signal(
        "Phrase repetition",
        float(np.clip(score, 0.0, 1.0)),
        f"LLM-tic hits: {tic_hits} across {len(words)} words. Trigram repetition: {repeat_rate*100:.1f}%. "
        + (
            "Frequent use of phrases that are statistically over-represented in LLM output."
            if score > 0.45
            else "No notable bias toward common LLM scaffolding phrases."
        ),
    )


def _punctuation_signal(text: str) -> Signal:
    """Punctuation entropy. LLMs rely heavily on commas and semicolons
    to hedge. Missing em-dashes, exclamations, parentheticals, and
    contractions is its own signal."""
    if len(text) < 120:
        return Signal("Punctuation entropy", 0.2, "Text too short to check.")
    counts = Counter(c for c in text if c in ",;:—–-!?()[]\"'")
    total = sum(counts.values()) or 1
    commas = counts.get(",", 0) / total
    em_like = (counts.get("—", 0) + counts.get("–", 0)) / total
    exclam = counts.get("!", 0) / total
    parens = (counts.get("(", 0) + counts.get(")", 0)) / total
    contractions = len(re.findall(r"\w+'\w+", text)) / max(1, len(text.split()))

    score = 0.0
    if commas > 0.55:
        score += 0.35
    if em_like < 0.01 and exclam < 0.005 and parens < 0.01:
        score += 0.35
    if contractions < 0.01 and len(text) > 400:
        score += 0.3
    return Signal(
        "Punctuation entropy",
        float(np.clip(score, 0.0, 1.0)),
        f"Comma share {commas:.2f}, em-dash share {em_like:.2f}, contraction rate {contractions:.3f}. "
        + (
            "Punctuation is comma-heavy and contraction-light — a common LLM register."
            if score > 0.45
            else "Punctuation mix reads like natural writing."
        ),
    )


def _verdict(signals: list[Signal]) -> tuple[float, str]:
    weights = {
        "Burstiness": 1.0,
        "Lexical rhythm": 0.9,
        "Phrase repetition": 0.9,
        "Punctuation entropy": 0.7,
    }
    num = sum(s.score * weights.get(s.name, 0.5) for s in signals)
    denom = sum(weights.get(s.name, 0.5) for s in signals)
    score = num / denom if denom else 0.0
    if score < 0.3:
        label = "likely written by a human"
    elif score < 0.55:
        label = "inconclusive"
    elif score < 0.75:
        label = "likely AI-generated"
    else:
        label = "highly likely AI-generated"
    return float(np.clip(score, 0.0, 1.0)), label


def analyze_text(text: str) -> dict[str, Any]:
    text = text or ""
    sents = _sentences(text)
    words = _WORD.findall(text)
    signals = [
        _burstiness_signal(sents),
        _lexical_signal(words),
        _repetition_signal(sents, words),
        _punctuation_signal(text),
    ]
    score, label = _verdict(signals)
    return {
        "kind": "text",
        "length": {"characters": len(text), "words": len(words), "sentences": len(sents)},
        "suspicion": score,
        "verdict": label,
        "confidence": round(abs(score - 0.5) * 2, 3),
        "signals": [
            {"name": s.name, "score": round(s.score, 3), "detail": s.detail} for s in signals
        ],
    }
