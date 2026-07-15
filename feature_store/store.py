"""
Shared feature store: the hand-off point between "streams" (Stages 2-5) and
"fusion" (Stage 6).

Every stream (Xception, EfficientNet, DINOv2, lip-sync, emotion) computes one
embedding vector per clip and writes a row here. Fusion later reads all rows
for a clip_id, one per stream, concatenates the embeddings, and feeds that
into the fusion MLP. Keeping this interface identical across streams is what
lets Stage 7's ablation swap streams in and out without touching fusion code.

Schema (one row per clip_id x stream_name):
    clip_id            str   -- matches clip_id in data/{train,val,test}.csv
    stream_name        str   -- e.g. "xception", "lipsync"
    embedding           list[float] -- the stream's output vector (already
                                        projected to common_dim, see
                                        PROJECT_OVERVIEW.md Section 3)
    split              str   -- "train" / "val" / "test", copied from the
                                 manifest so fusion doesn't need a join
    label              int   -- 1 = fake, 0 = real (clip-level), copied from
                                 the manifest for the same reason
    manipulation_type  str   -- one of the 4 FakeAVCeleb categories

Stored as Parquet because it's columnar (reading only "embedding" for one
stream is fast), typed (won't silently stringify a float list the way CSV
would), and pandas/pyarrow round-trip it exactly.
"""
from pathlib import Path
import numpy as np
import pandas as pd

STORE_DIR = Path(__file__).resolve().parent
SCHEMA_COLUMNS = ["clip_id", "stream_name", "embedding", "split", "label", "manipulation_type"]


def _store_path(stream_name: str) -> Path:
    """One Parquet file per stream keeps writes from different streams from racing each other."""
    return STORE_DIR / f"{stream_name}.parquet"


def write_embeddings(
    clip_ids: list[str],
    stream_name: str,
    embeddings: np.ndarray,
    splits: list[str],
    labels: list[int],
    manipulation_types: list[str],
) -> None:
    """
    Append/overwrite rows for a batch of clips for one stream.

    embeddings: shape [num_clips, embedding_dim] -- one row per clip_id, same
    order as clip_ids/splits/labels/manipulation_types.
    """
    try:
        if not (len(clip_ids) == len(embeddings) == len(splits) == len(labels) == len(manipulation_types)):
            raise ValueError(
                f"Mismatched lengths: clip_ids={len(clip_ids)}, embeddings={len(embeddings)}, "
                f"splits={len(splits)}, labels={len(labels)}, manipulation_types={len(manipulation_types)}"
            )

        new_rows = pd.DataFrame(
            {
                "clip_id": clip_ids,
                "stream_name": stream_name,
                # store each embedding as a plain python list so pyarrow writes a list<float> column
                "embedding": [np.asarray(e, dtype=np.float32).tolist() for e in embeddings],
                "split": splits,
                "label": labels,
                "manipulation_type": manipulation_types,
            }
        )

        path = _store_path(stream_name)
        if path.exists():
            existing = pd.read_parquet(path)
            # Replace any rows for the same clip_ids (re-running a stream overwrites, not duplicates).
            existing = existing[~existing["clip_id"].isin(new_rows["clip_id"])]
            combined = pd.concat([existing, new_rows], ignore_index=True)
        else:
            combined = new_rows

        combined.to_parquet(path, index=False)
    except Exception as e:
        raise RuntimeError(f"Failed to write embeddings for stream '{stream_name}': {e}") from e


def read_embeddings(stream_name: str, split: str | None = None) -> pd.DataFrame:
    """Read all rows for one stream, optionally filtered to one split."""
    path = _store_path(stream_name)
    try:
        if not path.exists():
            raise FileNotFoundError(f"No feature store file for stream '{stream_name}' at {path}")
        df = pd.read_parquet(path)
        if split is not None:
            df = df[df["split"] == split]
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read embeddings for stream '{stream_name}': {e}") from e


def read_fused(stream_names: list[str], split: str) -> pd.DataFrame:
    """
    Read multiple streams for one split and join them into one wide table,
    one row per clip_id, columns embedding_<stream_name> for each stream.
    This is what Stage 6 fusion and Stage 7 ablation will call directly.
    """
    try:
        merged = None
        for stream_name in stream_names:
            df = read_embeddings(stream_name, split=split)[["clip_id", "embedding", "label", "manipulation_type"]]
            df = df.rename(columns={"embedding": f"embedding_{stream_name}"})
            if merged is None:
                merged = df
            else:
                # Inner join on clip_id+label+manipulation_type: a clip only
                # belongs in fusion once every requested stream has embedded it.
                merged = merged.merge(
                    df, on=["clip_id", "label", "manipulation_type"], how="inner"
                )
        if merged is None:
            raise ValueError("stream_names is empty")
        return merged
    except Exception as e:
        raise RuntimeError(f"Failed to build fused table for streams {stream_names}: {e}") from e


if __name__ == "__main__":
    # Round-trip smoke test: write a dummy embedding, read it back, verify equality.
    dummy_clip_ids = ["clip_0001", "clip_0002"]
    dummy_embeddings = np.random.randn(2, 256).astype(np.float32)
    write_embeddings(
        clip_ids=dummy_clip_ids,
        stream_name="_smoke_test",
        embeddings=dummy_embeddings,
        splits=["train", "val"],
        labels=[0, 1],
        manipulation_types=["RealVideo-RealAudio", "FakeVideo-FakeAudio"],
    )
    roundtrip = read_embeddings("_smoke_test")
    recovered = np.stack(roundtrip.sort_values("clip_id")["embedding"].to_numpy())
    assert np.allclose(recovered, dummy_embeddings, atol=1e-5), "Round-trip mismatch!"
    print("Feature store round-trip test passed.")
    print(roundtrip)
    _store_path("_smoke_test").unlink()  # clean up the smoke-test file
    print("Cleaned up smoke-test parquet file.")
