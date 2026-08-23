from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


@dataclass(frozen=True, slots=True)
class FusionSample:
    branch_logits: Mapping[str, float]
    face_coverage: float
    audio_clipped: bool
    av_duration_delta_sec: float


@dataclass(frozen=True, slots=True)
class FusionArtifact:
    model: LateFusion
    split_hash: str
    preprocessing_hash: str

    @property
    def branch_names(self) -> tuple[str, ...]:
        return self.model.branch_names

    def validate_provenance(
        self,
        *,
        split_hash: str,
        preprocessing_hash: str,
    ) -> None:
        if split_hash != self.split_hash:
            raise ValueError("Fusion artifact uses a different split hash")
        if preprocessing_hash != self.preprocessing_hash:
            raise ValueError("Fusion artifact uses a different preprocessing hash")

    def predict_proba(self, samples: Sequence[FusionSample]) -> np.ndarray:
        return self.model.predict_proba(samples)


class LateFusion:
    def __init__(
        self,
        *,
        branch_names: tuple[str, ...],
        regularization: float = 1.0,
        classifier_kind: str = "logistic",
    ) -> None:
        if not branch_names:
            raise ValueError("At least one branch is required")
        if regularization <= 0:
            raise ValueError("Regularization must be positive")
        if classifier_kind not in {"logistic", "mlp"}:
            raise ValueError("Fusion classifier must be logistic or mlp")
        self.branch_names = branch_names
        self.regularization = regularization
        self.classifier_kind = classifier_kind
        self.calibrators: dict[str, LogisticRegression] = {}
        self.classifier: LogisticRegression | MLPClassifier | None = None

    def _validate(self, samples: Sequence[FusionSample]) -> None:
        for sample in samples:
            missing = [
                name for name in self.branch_names if name not in sample.branch_logits
            ]
            if missing:
                raise ValueError(f"Missing branch logits: {', '.join(missing)}")

    def _quality_features(self, samples: Sequence[FusionSample]) -> np.ndarray:
        return np.asarray(
            [
                (
                    sample.face_coverage,
                    float(sample.audio_clipped),
                    sample.av_duration_delta_sec,
                )
                for sample in samples
            ],
            dtype=np.float64,
        )

    def _calibrated_features(
        self,
        samples: Sequence[FusionSample],
        labels: Sequence[int] | None = None,
    ) -> np.ndarray:
        columns: list[np.ndarray] = []
        for name in self.branch_names:
            values = np.asarray(
                [[sample.branch_logits[name]] for sample in samples],
                dtype=np.float64,
            )
            if labels is not None:
                calibrator = LogisticRegression(
                    C=self.regularization, solver="liblinear"
                )
                calibrator.fit(values, labels)
                self.calibrators[name] = calibrator
            calibrator = self.calibrators.get(name)
            if calibrator is None:
                raise RuntimeError("Late fusion has not been fitted")
            probabilities = calibrator.predict_proba(values)[:, 1]
            probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
            columns.append(np.log(probabilities / (1 - probabilities)))
        return np.column_stack(columns)

    def fit(
        self,
        samples: Sequence[FusionSample],
        labels: Sequence[int],
    ) -> LateFusion:
        if len(samples) != len(labels) or not samples:
            raise ValueError("Samples and labels must have equal nonzero length")
        self._validate(samples)
        if set(labels) != {0, 1}:
            raise ValueError("Fusion training requires both binary classes")
        calibrated = self._calibrated_features(samples, labels)
        features = np.column_stack((calibrated, self._quality_features(samples)))
        if self.classifier_kind == "logistic":
            self.classifier = LogisticRegression(
                C=self.regularization,
                solver="liblinear",
            )
        else:
            self.classifier = MLPClassifier(
                hidden_layer_sizes=(8,),
                activation="relu",
                solver="lbfgs",
                alpha=1e-4 / self.regularization,
                max_iter=1_000,
                random_state=17,
            )
        self.classifier.fit(features, labels)
        return self

    def predict_proba(self, samples: Sequence[FusionSample]) -> np.ndarray:
        if self.classifier is None:
            raise RuntimeError("Late fusion has not been fitted")
        self._validate(samples)
        calibrated = self._calibrated_features(samples)
        features = np.column_stack((calibrated, self._quality_features(samples)))
        return self.classifier.predict_proba(features)[:, 1]

    def predict_branch_proba(
        self,
        samples: Sequence[FusionSample],
        *,
        branch: str,
    ) -> np.ndarray:
        if branch not in self.branch_names:
            raise ValueError(f"Fusion artifact has no {branch} branch")
        if any(branch not in sample.branch_logits for sample in samples):
            raise ValueError(f"Missing branch logits: {branch}")
        calibrator = self.calibrators.get(branch)
        if calibrator is None:
            raise RuntimeError("Late fusion has not been fitted")
        values = np.asarray(
            [[sample.branch_logits[branch]] for sample in samples],
            dtype=np.float64,
        )
        return calibrator.predict_proba(values)[:, 1]
