"""Turn the three check scores of an upload into one of four outcomes.

Each check's score is placed as a percentile among real calibration clips, the
real clips of the four P-001 evaluation sets with every corpus weighted equally.
Since 11-20 the evidence sum of all checks decides (`Fusion`); before, and in older files, the combined
score was the highest available percentile. The threshold is the
95th weighted percentile of the combined score over the same real clips, so at
most 5 in 100 of them are flagged; the 80th percentile starts "can't tell".

A video longer than one clip is checked in parts, and its most unusual part
decides. With n parts a real video has n chances to cross the line, so both
lines move to the 1 - 0.05/n and 1 - 0.20/n quantiles of the real combined
scores (a Bonferroni correction): a real video is then flagged at most about as
often as a real clip, whatever the parts have in common.
The design is in docs/superpowers/specs/2026-11-17-website-design.md.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

# "gen" (generated video) is optional: a calibration without its reference ignores it.
CHECKS = ("face", "voice", "sync", "gen")
FLAG_QUANTILE = 0.95
UNSURE_QUANTILE = 0.80
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
        return cls(
            tuple(float(v) for v in np.asarray(scores, float)[order]),
            tuple(float(x) for x in w / w.sum()),
        )

    def quantile(self, q: float) -> float:
        return weighted_quantile(list(self.values), list(self.weights), q)

    def percentile(self, score: float) -> float:
        """Weighted share of reference clips below `score`, counting ties as half."""
        values = np.asarray(self.values)
        cum = np.concatenate([[0.0], np.cumsum(self.weights)])
        left = int(np.searchsorted(values, score, "left"))
        right = int(np.searchsorted(values, score, "right"))
        return float(min(1.0, cum[left] + 0.5 * (cum[right] - cum[left])))


def evidence_sum(percentiles: list[float]) -> float:
    """Fisher's statistic over the checks that ran, divided by its degrees of freedom: several mildly unusual
    checks add up. Nothing is learned from fakes; the lines come from real clips like every other line."""
    stat = -2 * sum(math.log(max(1 - p, 1e-6)) for p in percentiles)
    return stat / (2 * len(percentiles))


@dataclass(frozen=True)
class Fusion:
    """The overall score: the evidence sum, with its own lines set on the same real clips (11-20).

    Tested on corpora it never saw (scripts/fusion_test.py): 32% of fakes caught against 29% for the
    most-unusual-check rule, at about 5 in 100 real clips; a trained fusion dropped to 41% on FakeAVCeleb.
    """

    reference: Reference  # the real clips' evidence sums, one part each
    threshold: float
    lower: float
    parts_lines: dict[int, tuple[float, float]] | None = None

    def limits(self, parts: int) -> tuple[float, float]:
        if parts <= 1:
            return self.threshold, self.lower
        if self.parts_lines:
            counts = sorted(self.parts_lines)
            at_least = [n for n in counts if n >= parts]
            return self.parts_lines[at_least[0] if at_least else counts[-1]]
        return (
            self.reference.quantile(1 - (1 - FLAG_QUANTILE) / parts),
            self.reference.quantile(1 - (1 - UNSURE_QUANTILE) / parts),
        )


@dataclass(frozen=True)
class Outcome:
    kind: str  # signs, cant_tell, no_signs or cant_check
    combined: float | None
    percentiles: dict[str, float | None]
    raised: list[str] = field(default_factory=list)
    part: int | None = None  # the index of the part that decided
    parts: list[float | None] = field(
        default_factory=list
    )  # each part's combined percentile


@dataclass(frozen=True)
class Calibration:
    references: dict[str, Reference]
    threshold: float
    lower: float
    meta: dict = field(default_factory=dict, compare=False)
    combined: Reference | None = (
        None  # the real clips' combined percentiles, for videos in parts
    )
    # Lines measured on real videos checked in parts: {parts: (flag, lower)}. They replace the Bonferroni
    # bound, which assumes a video's parts are unrelated and so sets the line far too high.
    parts_lines: dict[int, tuple[float, float]] | None = None
    # With a fusion block the evidence sum decides; the per-check lines above still place each check's bar.
    fusion: Fusion | None = None

    def reading_limits(self, parts: int) -> tuple[float, float]:
        """The lines of the score that decides: the evidence sum's when there is one."""
        return self.fusion.limits(parts) if self.fusion else self.limits(parts)

    def percentiles(self, scores: dict[str, float | None]) -> dict[str, float | None]:
        return {
            c: None
            if scores.get(c) is None or c not in self.references
            else self.references[c].percentile(scores[c])
            for c in CHECKS
        }

    def limits(self, parts: int) -> tuple[float, float]:
        """The flag line and the "can't tell" line for a video checked in `parts` parts."""
        if parts <= 1:
            return self.threshold, self.lower
        if self.parts_lines:
            counts = sorted(self.parts_lines)
            at_least = [n for n in counts if n >= parts]
            return self.parts_lines[at_least[0] if at_least else counts[-1]]
        if self.combined is None:
            raise ValueError(
                "a video in parts needs the combined reference; rebuild the calibration file"
            )
        return (
            self.combined.quantile(1 - (1 - FLAG_QUANTILE) / parts),
            self.combined.quantile(1 - (1 - UNSURE_QUANTILE) / parts),
        )

    def decide(self, scores: dict[str, float | None]) -> Outcome:
        """`scores`: each check's score, or None when its quality gate failed."""
        return self.decide_parts([scores])

    def decide_parts(self, parts: list[dict[str, float | None]]) -> Outcome:
        """One outcome for a video from the check scores of each of its parts.

        Parts where no check could run are skipped and do not count toward the correction.
        """
        pcts = [self.percentiles(scores) for scores in parts]
        combined = [
            max((p for p in pct.values() if p is not None), default=None)
            for pct in pcts
        ]
        checked = [i for i, c in enumerate(combined) if c is not None]
        if not checked:
            return Outcome(
                "cant_check",
                None,
                pcts[0] if pcts else {c: None for c in CHECKS},
                parts=combined,
            )
        if self.fusion is not None:
            return self._decide_fused(pcts, checked)
        worst = max(checked, key=lambda i: combined[i])
        pct, top = pcts[worst], combined[worst]
        threshold, lower = self.limits(len(checked))
        if top > threshold:
            # Named only when it crosses the line in force, so every named model reads 0.6 or more on the page.
            raised = [c for c in CHECKS if pct[c] is not None and pct[c] > threshold]
            return Outcome("signs", top, pct, raised, worst, combined)
        return Outcome(
            "cant_tell" if top > lower else "no_signs", top, pct, [], worst, combined
        )

    def _decide_fused(self, pcts: list[dict], checked: list[int]) -> Outcome:
        fused = [
            evidence_sum([p for p in pct.values() if p is not None])
            if i in checked
            else None
            for i, pct in enumerate(pcts)
        ]
        worst = max(checked, key=lambda i: fused[i])
        pct, top = pcts[worst], fused[worst]
        threshold, lower = self.fusion.limits(len(checked))
        if top > threshold:
            # A check is named when it would cross its own line; an empty list means several checks together.
            try:
                own = self.limits(len(checked))[0]
            except ValueError:
                own = self.threshold
            raised = [c for c in CHECKS if pct[c] is not None and pct[c] > own]
            return Outcome("signs", top, pct, raised, worst, fused)
        return Outcome(
            "cant_tell" if top > lower else "no_signs", top, pct, [], worst, fused
        )

    def to_json(self) -> str:
        refs = {
            c: {"values": list(r.values), "weights": list(r.weights)}
            for c, r in self.references.items()
        }
        data = {
            "meta": self.meta,
            "threshold": self.threshold,
            "lower": self.lower,
            "references": refs,
        }
        if self.combined is not None:
            data["combined"] = {
                "values": list(self.combined.values),
                "weights": list(self.combined.weights),
            }
        if self.parts_lines:
            data["parts_lines"] = {
                str(n): list(v) for n, v in sorted(self.parts_lines.items())
            }
        if self.fusion is not None:
            f = self.fusion
            data["fusion"] = {
                "rule": "evidence sum",
                "threshold": f.threshold,
                "lower": f.lower,
                "reference": {
                    "values": list(f.reference.values),
                    "weights": list(f.reference.weights),
                },
            }
            if f.parts_lines:
                data["fusion"]["parts_lines"] = {
                    str(n): list(v) for n, v in sorted(f.parts_lines.items())
                }
        return json.dumps(data, indent=1)

    @classmethod
    def from_json(cls, text: str) -> Calibration:
        data = json.loads(text)
        refs = {
            c: Reference(tuple(r["values"]), tuple(r["weights"]))
            for c, r in data["references"].items()
        }
        combined = data.get("combined")
        fused = data.get("fusion")
        fusion = (
            None
            if fused is None
            else Fusion(
                Reference(
                    tuple(fused["reference"]["values"]),
                    tuple(fused["reference"]["weights"]),
                ),
                float(fused["threshold"]),
                float(fused["lower"]),
                {
                    int(n): (float(v[0]), float(v[1]))
                    for n, v in fused["parts_lines"].items()
                }
                if fused.get("parts_lines")
                else None,
            )
        )
        return cls(
            refs,
            float(data["threshold"]),
            float(data["lower"]),
            data.get("meta", {}),
            None
            if combined is None
            else Reference(tuple(combined["values"]), tuple(combined["weights"])),
            {int(n): (float(v[0]), float(v[1])) for n, v in data["parts_lines"].items()}
            if data.get("parts_lines")
            else None,
            fusion,
        )


def build(
    rows: list[dict], checks: tuple[str, ...] | None = None
) -> tuple[Calibration, dict]:
    """Calibration from rows {corpus, label (0 real, 1 fake, clip label), scores {check: float | None}}.

    `checks` names the checks the input can feed: a picture feeds the face check alone. By default it is
    every check some real clip has a score for.

    Returns the calibration and, per corpus and label, how many clips land in each outcome.
    """
    reals = [r for r in rows if r["label"] == 0]
    if checks is None:  # every check some real clip has a score for
        checks = tuple(
            c for c in CHECKS if any(r["scores"].get(c) is not None for r in reals)
        )
    references = {}
    for check in checks:
        have = [r for r in reals if r["scores"].get(check) is not None]
        if not have:
            raise ValueError(f"no real calibration clip has a {check} score")
        references[check] = Reference.from_scores(
            [r["scores"][check] for r in have],
            corpus_weights([r["corpus"] for r in have]),
        )
    draft = Calibration(references, threshold=1.0, lower=1.0)
    scored = [(r, draft.decide(r["scores"]).combined) for r in reals]
    scored = [(r, c) for r, c in scored if c is not None]
    combined = [c for _, c in scored]
    weights = corpus_weights([r["corpus"] for r, _ in scored])
    sums = [
        evidence_sum(
            [p for p in draft.percentiles(r["scores"]).values() if p is not None]
        )
        for r, _ in scored
    ]
    final = Calibration(
        references,
        weighted_quantile(combined, weights, FLAG_QUANTILE),
        weighted_quantile(combined, weights, UNSURE_QUANTILE),
        combined=Reference.from_scores(combined, weights),
        fusion=Fusion(
            Reference.from_scores(sums, weights),
            weighted_quantile(sums, weights, FLAG_QUANTILE),
            weighted_quantile(sums, weights, UNSURE_QUANTILE),
        ),
    )
    summary: dict = {}
    for r in rows:
        side = summary.setdefault(r["corpus"], {}).setdefault(
            "fake" if r["label"] else "real", Counter()
        )
        side[final.decide(r["scores"]).kind] += 1
    return final, {
        c: {s: dict(n) for s, n in sides.items()} for c, sides in summary.items()
    }


def raised_by(calibration: Calibration, rows: list[dict]) -> dict:
    """Per corpus, how many fake clips each check flagged, for the page's per-check catch rates.

    A fake flagged by two checks counts for both. Real clips are left out.
    """
    counts: dict = {}
    for r in rows:
        if not r["label"]:
            continue
        tally = counts.setdefault(
            r["corpus"], {"fakes": 0} | dict.fromkeys(calibration.references, 0)
        )
        tally["fakes"] += 1
        outcome = calibration.decide(r["scores"])
        if outcome.kind == "signs":
            for check in outcome.raised:
                tally[check] += 1
    return counts


# The page's one scale, set by Vaibhav: 0 is real, 0.5 is neutral, 1 is fake. "Can't tell" is the band
# around 0.5 and a score is flagged from 0.6.
SHOWN_LOWER, SHOWN_FLAG = 0.4, 0.6


def display_score(reading: float | None, flag: float, lower: float) -> float | None:
    """A reading (share of real clips scored lower) on the page's 0-to-1 scale, piecewise linear so the
    can't-tell line lands on 0.4 and the flag line on 0.6. Order is kept, so no answer changes."""
    if reading is None:
        return None
    if reading < lower:
        shown = SHOWN_LOWER * reading / lower
    elif reading < flag:
        shown = SHOWN_LOWER + (SHOWN_FLAG - SHOWN_LOWER) * (reading - lower) / (
            flag - lower
        )
    else:
        shown = (
            1.0
            if flag >= 1
            else SHOWN_FLAG + (1 - SHOWN_FLAG) * (reading - flag) / (1 - flag)
        )
    return round(min(max(shown, 0.0), 1.0), 3)
