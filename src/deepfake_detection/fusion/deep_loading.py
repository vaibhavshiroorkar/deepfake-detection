"""Load a trained `StreamFusion` and score a clip from whatever streams ran.

`cli._deep_fusion` saves the head and the shape it was built with, and until now
nothing read that back: the deep path could train and never predict. This is the
other end of the trip.

The shape matters more than the weights. A `StreamFusion` is built from the
stream names and widths it was fitted on, so a head trained over five streams
cannot be handed four and asked to cope. Streams the caller does not supply are
passed as zeros with presence 0, which is the same thing the trainer did for a
clip that could not be read, and which `forward` masks after the projection so
the contribution is exactly zero including the bias.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from deepfake_detection.fusion.deep import StreamFusion


@dataclass(frozen=True, slots=True)
class LoadedFusion:
    """A fitted head plus the provenance it was trained under."""

    model: StreamFusion
    stream_dims: dict[str, int]
    split_hash: str
    preprocessing_hash: str
    # One cut-off per media kind, empty for a head fitted without them. A kind
    # with no entry has no calibrated threshold, and the caller is expected to
    # say so rather than reaching for 0.5.
    thresholds: dict[str, float] = field(default_factory=dict)

    def threshold_for(self, kind: str) -> float | None:
        return self.thresholds.get(kind)

    @property
    def streams(self) -> tuple[str, ...]:
        return tuple(sorted(self.stream_dims))


def load_fusion(path: Path, device: str = "cpu") -> LoadedFusion:
    """Rebuild the head from its own payload, never from caller-supplied shape.

    `weights_only=True` refuses to unpickle arbitrary objects, so the payload
    has to be plain data. That is why the trainer writes `stream_dims` as a
    dict rather than saving the module.
    """
    payload = torch.load(path, map_location=device, weights_only=True)
    for key in ("model", "stream_dims"):
        if key not in payload:
            raise ValueError(f"{path} is not a deep fusion checkpoint: no {key!r}")

    dims = {str(name): int(width) for name, width in payload["stream_dims"].items()}
    model = StreamFusion(
        stream_dims=dims,
        common_dim=int(payload.get("common_dim", 256)),
        hidden_sizes=tuple(payload.get("hidden_sizes", (128,))),
        # Inference only: dropout and stream dropout are training-time
        # regularisers and `eval()` disables them anyway, but building with them
        # at zero keeps the module's own repr honest about what it is doing.
        dropout=0.0,
        stream_dropout=0.0,
    )
    model.load_state_dict(payload["model"])
    return LoadedFusion(
        model=model.to(device).eval(),
        stream_dims=dims,
        split_hash=str(payload.get("split_hash", "")),
        preprocessing_hash=str(payload.get("preprocessing_hash", "")),
        thresholds={
            str(kind): float(cut)
            for kind, cut in (payload.get("thresholds") or {}).items()
        },
    )


def fuse(
    loaded: LoadedFusion,
    embeddings: Mapping[str, tuple[float, ...]],
    device: str = "cpu",
) -> float:
    """One clip's fake probability from the streams that produced an embedding.

    `embeddings` carries only the streams that ran. Anything the head knows
    about and the caller omits is supplied as zeros with presence 0.

    Raises when nothing ran: a head handed an all-absent input returns whatever
    its biases encode, which is a number with no evidence behind it. The caller
    abstains instead.
    """
    unknown = sorted(set(embeddings) - set(loaded.stream_dims))
    if unknown:
        raise ValueError(
            f"Fusion head was not trained on: {', '.join(unknown)}. "
            f"It knows {', '.join(loaded.streams)}."
        )
    present = {name: values for name, values in embeddings.items() if values}
    if not present:
        raise ValueError("No stream produced an embedding, so there is nothing to fuse")

    values: dict[str, Tensor] = {}
    presence: dict[str, Tensor] = {}
    for name, width in loaded.stream_dims.items():
        found = present.get(name)
        if found is not None and len(found) != width:
            raise ValueError(
                f"Stream {name!r} gave {len(found)} dimensions, head expects {width}"
            )
        values[name] = torch.tensor(
            [list(found)] if found else [[0.0] * width], dtype=torch.float32
        ).to(device)
        presence[name] = torch.tensor(
            [1.0 if found else 0.0], dtype=torch.float32
        ).to(device)

    with torch.inference_mode():
        output = loaded.model(values, presence)
    return float(torch.sigmoid(output.logit)[0])
