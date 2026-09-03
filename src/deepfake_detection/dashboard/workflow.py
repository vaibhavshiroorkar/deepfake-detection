from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class StepState(StrEnum):
    WAITING = "waiting"
    READY = "ready"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class WorkflowState:
    video: StepState
    preprocessing: StepState
    visual_model: StepState
    prediction: StepState


def workflow_state(
    *, has_upload: bool, has_prepared: bool, has_prediction: bool
) -> WorkflowState:
    return WorkflowState(
        video=StepState.COMPLETE if has_upload else StepState.READY,
        preprocessing=(
            StepState.COMPLETE
            if has_prepared
            else StepState.READY
            if has_upload
            else StepState.WAITING
        ),
        visual_model=StepState.READY,
        prediction=(
            StepState.COMPLETE
            if has_prediction
            else StepState.READY
            if has_prepared
            else StepState.WAITING
        ),
    )
