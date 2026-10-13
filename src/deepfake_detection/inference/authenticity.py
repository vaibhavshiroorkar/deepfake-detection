"""Is this footage like anything the model was trained on?

Fitted on authentic clips only and queried with a Mahalanobis distance. No fake
example is involved, which is the point: the question is "is this unlike the
footage I know", not "is this like a fake I have seen", and only the first
generalises to a camera nobody anticipated.

It exists because of a measured failure. At the fixed 0.50 threshold the visual
branch calls 90.97 percent of genuine Celeb-DF clips fake, 141 of 155 authentic
videos. Raising the threshold does not help, because forcing zero false alarms
there needs a 0.9998 cut-off at which nothing is detected. The motion gate next
door catches 5.7 percent of those accusations, because the clips are unfamiliar
in appearance rather than in motion.

This catches 74.4 percent of them at an 8.7 percent abstention rate on familiar
genuine footage, and 67.3 percent averaged over twenty reference splits, taking
false accusations from about 91 percent to between a quarter and a third.
It does not reach zero at any setting, and the gain is paid for in abstentions,
so it is only useful where the system may answer "outside what I was trained on"
instead of being forced to choose.

The reference set defines "familiar". Fitted on FakeAVCeleb it recognises
FakeAVCeleb-like footage, which is a corpus proxy rather than a deployment. A
deployment should refit on genuine clips from the devices it will actually meet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class FamiliarityCheck:
    """How far one clip sits from the authentic footage the model knows."""

    distance: float
    threshold: float
    familiar: bool
    reference: str

    @property
    def blocker(self) -> str | None:
        return None if self.familiar else "unfamiliar_capture_conditions"

    @property
    def reason(self) -> str:
        if self.familiar:
            return ""
        return (
            f"This clip sits {self.distance:.1f} from the authentic footage this "
            f"model was fitted on, past the {self.threshold:.1f} that {self.reference} "
            "clips stay within. Its capture conditions are unlike anything in "
            "training, and on such footage the detector accuses genuine video "
            "about nine times in ten, so no verdict is issued."
        )


@dataclass(frozen=True, slots=True)
class AuthenticityDensity:
    """A Gaussian over frozen features of authentic clips, plus its threshold."""

    mean: np.ndarray
    components: np.ndarray
    centre: np.ndarray
    inverse: np.ndarray
    threshold: float
    encoder: str
    reference: str
    flag_rate: float

    def distance(self, features: np.ndarray) -> float:
        reduced = (
            np.asarray(features, dtype=np.float64) - self.mean
        ) @ self.components.T
        delta = reduced - self.centre
        return float(np.sqrt(delta @ self.inverse @ delta))

    def check(self, features: np.ndarray) -> FamiliarityCheck:
        value = self.distance(features)
        return FamiliarityCheck(
            distance=value,
            threshold=self.threshold,
            familiar=value <= self.threshold,
            reference=self.reference,
        )


def save(path: Path, model: AuthenticityDensity) -> None:
    np.savez_compressed(
        path,
        mean=model.mean,
        components=model.components,
        centre=model.centre,
        inverse=model.inverse,
        threshold=np.asarray([model.threshold]),
        encoder=np.asarray([model.encoder]),
        reference=np.asarray([model.reference]),
        flag_rate=np.asarray([model.flag_rate]),
    )


def load(path: Path) -> AuthenticityDensity:
    payload = np.load(path, allow_pickle=False)
    return AuthenticityDensity(
        mean=payload["mean"],
        components=payload["components"],
        centre=payload["centre"],
        inverse=payload["inverse"],
        threshold=float(payload["threshold"][0]),
        encoder=str(payload["encoder"][0]),
        reference=str(payload["reference"][0]),
        flag_rate=float(payload["flag_rate"][0]),
    )


class FamiliarityGate:
    """The density model plus the encoder it needs, ready to score a file.

    Holds the encoder because loading a ViT per clip would dominate the latency
    of the thing it is guarding.
    """

    def __init__(self, model: AuthenticityDensity, device: str = "cuda") -> None:
        from deepfake_detection.inference import frozen_features

        self.model = model
        self.device = device
        self.encoder, self.size = frozen_features.load_encoder(model.encoder, device)

    def check_clip(self, path: Path) -> FamiliarityCheck | None:
        """None when the clip cannot be read, so the caller falls through."""
        from deepfake_detection.inference import frozen_features

        features = frozen_features.embed_clip(
            path, self.encoder, self.size, self.device
        )
        if features is None:
            return None
        return self.model.check(features)


def load_gate(path: Path, device: str = "cuda") -> FamiliarityGate | None:
    """The served gate, or None when no model has been fitted.

    Absence is a normal state: a checkout with no fitted density model should
    serve verdicts as before rather than fail, and say so rather than silently
    dropping the protection.
    """
    if not Path(path).is_file():
        return None
    return FamiliarityGate(load(Path(path)), device=device)
