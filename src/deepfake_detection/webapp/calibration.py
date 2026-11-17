"""Turn the three check scores of an upload into one of four outcomes.

Each check's score is placed as a percentile among real calibration clips, the
real clips of the four P-001 evaluation sets with every corpus weighted equally.
The combined score is the highest available percentile. The threshold is the
95th weighted percentile of the combined score over the same real clips, so at
most 5 in 100 of them are flagged; the 80th percentile starts "can't tell".
The design is in docs/superpowers/specs/2026-11-17-website-design.md.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

CHECKS = ("face", "voice", "sync")
FLAG_QUANTILE = 0.95
UNSURE_QUANTILE = 0.80
OWN_FLAG = 0.95          # a check is named as raising the flag at or above its own 95th percentile
_EPS = 1e-7


def check_score(probabilities: list[float]) -> float:
    """The mean logit over seeds. Logits keep the order of probabilities near 0 and 1 apart."""
    logits = [math.log(p / (1 - p)) for p in np.clip(probabilities, _EPS, 1 - _EPS)]
    return float(np.mean(logits))


def corpus_weights(corpora: list[str]) -> list[float]:
    """One weight per clip so that every corpus sums to the same share and all sum to 1."""
    counts = Counter(corpora)
    return [1 / (len(counts) * counts[c]) for c in corpora]


def weighted_quantile(values: list[float], weights: list[float], q: float) -> float:
    """The smallest value whose cumulative weight reaches q."""
    order = np.argsort(values, kind="stable")
    cum = np.cumsum(np.asarray(weights, float)[order])
    cum /= cum[-1]
    return float(np.asarray(values, float)[order][np.searchsorted(cum, q - 1e-12)])


@dataclass(frozen=True)
class Reference:
    """Sorted real-clip scores of one check, with their normalized weights."""

    values: tuple[float, ...]
    weights: tuple[float, ...]

    @classmethod
    def from_scores(cls, scores: list[float], weights: list[float]) -> Reference:
        order = np.argsort(scores, kind="stable")
        w = np.asarray(weights, float)[order]
        return cls(tuple(float(v) for v in np.asarray(scores, float)[order]), tuple(float(x) for x in w / w.sum()))

    def percentile(self, score: float) -> float:
        """Weighted share of reference clips below `score`, counting ties as half."""
        values = np.asarray(self.values)
        cum = np.concatenate([[0.0], np.cumsum(self.weights)])
        left = int(np.searchsorted(values, score, "left"))
        right = int(np.searchsorted(values, score, "right"))
        return float(min(1.0, cum[left] + 0.5 * (cum[right] - cum[left])))


@dataclass(frozen=True)
class Outcome:
    kind: str                      # signs, cant_tell, no_signs or cant_check
    combined: float | None
    percentiles: dict[str, float | None]
    raised: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Calibration:
    references: dict[str, Reference]
    threshold: float
    lower: float
    meta: dict = field(default_factory=dict, compare=False)

    def percentiles(self, scores: dict[str, float | None]) -> dict[str, float | None]:
        return {c: None if scores.get(c) is None else self.references[c].percentile(scores[c]) for c in CHECKS}

    def decide(self, scores: dict[str, float | None]) -> Outcome:
        """`scores`: each check's score, or None when its quality gate failed."""
        pct = self.percentiles(scores)
        available = [p for p in pct.values() if p is not None]
        if not available:
            return Outcome("cant_check", None, pct)
        combined = max(available)
        if combined > self.threshold:
            bar = min(OWN_FLAG, combined)
            return Outcome("signs", combined, pct, [c for c in CHECKS if pct[c] is not None and pct[c] >= bar])
        return Outcome("cant_tell" if combined > self.lower else "no_signs", combined, pct)

    def to_json(self) -> str:
        refs = {c: {"values": list(r.values), "weights": list(r.weights)} for c, r in self.references.items()}
        return json.dumps({"meta": self.meta, "threshold": self.threshold, "lower": self.lower,
                           "references": refs}, indent=1)

    @classmethod
    def from_json(cls, text: str) -> Calibration:
        data = json.loads(text)
        refs = {c: Reference(tuple(r["values"]), tuple(r["weights"])) for c, r in data["references"].items()}
        return cls(refs, float(data["threshold"]), float(data["lower"]), data.get("meta", {}))


def build(rows: list[dict]) -> tuple[Calibration, dict]:
    """Calibration from rows {corpus, label (0 real, 1 fake, clip label), scores {check: float | None}}.

    Returns the calibration and, per corpus and label, how many clips land in each outcome.
    """
    reals = [r for r in rows if r["label"] == 0]
    references = {}
    for check in CHECKS:
        have = [r for r in reals if r["scores"].get(check) is not None]
        if not have:
            raise ValueError(f"no real calibration clip has a {check} score")
        references[check] = Reference.from_scores([r["scores"][check] for r in have],
                                                  corpus_weights([r["corpus"] for r in have]))
    draft = Calibration(references, threshold=1.0, lower=1.0)
    scored = [(r, draft.decide(r["scores"]).combined) for r in reals]
    scored = [(r, c) for r, c in scored if c is not None]
    combined = [c for _, c in scored]
    weights = corpus_weights([r["corpus"] for r, _ in scored])
    final = Calibration(references, weighted_quantile(combined, weights, FLAG_QUANTILE),
                        weighted_quantile(combined, weights, UNSURE_QUANTILE))
    summary: dict = {}
    for r in rows:
        side = summary.setdefault(r["corpus"], {}).setdefault("fake" if r["label"] else "real", Counter())
        side[final.decide(r["scores"]).kind] += 1
    return final, {c: {s: dict(n) for s, n in sides.items()} for c, sides in summary.items()}
