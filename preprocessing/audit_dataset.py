"""
Stage 1 - dataset audit (authoritative, driven by meta_data.csv).

FakeAVCeleb ships data/raw/FakeAVCeleb_v1.2/meta_data.csv, which is the source
of truth for labels and identities. We use it instead of parsing folder paths
because it also gives us the deepfake `method` and the `source`/`target1`/
`target2` identities, which matter for a leakage-safe split (see below).

Output: data/full_manifest.csv, one row per existing video, columns:
    clip_id, video_path, label, manipulation_type, method,
    source, target1, target2, race, gender

Label convention (clip-level, per PROJECT_OVERVIEW.md): 1 = fake, 0 = real.
Only category A (RealVideo-RealAudio) is real; B/C/D are all fake overall.

Identity for splitting = `source` (the 500 VoxCeleb base identities). Why not
a connected-components split over source<->target swap pairs: the swap graph in
FakeAVCeleb is fully connected (one component of 578 identities), so components
are useless for splitting. `source` controls the DOMINANT leakage mode -- the
same underlying real footage (background, lighting, framing) appearing in a
train clip and its fake derivative in test -- because every fake is derived
from a source's real video. Residual risk (a target face swapped into
different sources across splits) is unavoidable given the dense swap graph and
is far milder, since the label is manipulation, not identity. Documented in
docs/stage-1-plan.md.
"""
import sys
from pathlib import Path
import pandas as pd

try:
    import cv2
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Run: uv sync --extra cpu (or --extra cu130 for GPU), see README.md")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "raw" / "FakeAVCeleb_v1.2"
META_CSV = DATA_ROOT / "meta_data.csv"
OUT_MANIFEST = REPO_ROOT / "data" / "full_manifest.csv"

# meta_data.csv has a trailing comma in its header, producing an unnamed final
# column; and its 9th header "path" actually holds the filename. We rename
# positionally to reflect the real content.
META_COLUMNS = ["source", "target1", "target2", "method", "category",
                "manipulation_type", "race", "gender", "filename", "dirpath"]

REAL_TYPE = "RealVideo-RealAudio"


def resolve_video_path(row) -> Path:
    """
    meta dirpath looks like 'FakeAVCeleb/<type>/<race>/<gender>/<id>'. The real
    extracted tree drops the leading 'FakeAVCeleb/' and lives under
    data/raw/FakeAVCeleb_v1.2/. Append the filename to get the actual file.
    """
    dirpath = row["dirpath"]
    # strip the leading 'FakeAVCeleb/' component if present
    parts = Path(dirpath).parts
    if parts and parts[0] == "FakeAVCeleb":
        parts = parts[1:]
    return DATA_ROOT.joinpath(*parts, row["filename"])


def check_readable(video_path: Path) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return False
    ok, _ = cap.read()
    cap.release()
    return ok


def audit(sample_check_n: int = 200):
    if not META_CSV.exists():
        raise FileNotFoundError(f"Expected {META_CSV} -- is the dataset extracted?")

    meta = pd.read_csv(META_CSV)
    meta.columns = META_COLUMNS  # positional rename (see note above)

    rows = []
    missing = 0
    for _, r in meta.iterrows():
        vp = resolve_video_path(r)
        if not vp.exists():
            missing += 1
            continue
        label = 0 if r["manipulation_type"] == REAL_TYPE else 1
        clip_id = f"{r['category']}__{r['source']}__{Path(r['filename']).stem}"
        rows.append({
            "clip_id": clip_id,
            "video_path": str(vp.relative_to(REPO_ROOT / "data")),  # relative to data/
            "label": label,
            "manipulation_type": r["manipulation_type"],
            "method": r["method"],
            "source": r["source"],
            "target1": r["target1"],
            "target2": r["target2"],
            "race": r["race"],
            "gender": r["gender"],
        })

    df = pd.DataFrame(rows)
    # meta_data.csv lists ~22 physical files twice with conflicting `method`
    # labels (e.g. 'wav2lip' vs 'faceswap-wav2lip' for the same .mp4). Dedupe
    # on the actual file path, keeping the first, so each file appears once.
    before = len(df)
    df = df.drop_duplicates(subset="video_path", keep="first").reset_index(drop=True)
    if before != len(df):
        print(f"Deduped {before - len(df)} rows that pointed at the same physical file.")

    # clip_id must be unique for the feature store to key on it; assert it.
    dupes = df["clip_id"].duplicated().sum()
    if dupes:
        raise RuntimeError(f"{dupes} duplicate clip_ids -- clip_id scheme is not unique!")

    print(f"Meta rows: {len(meta)}, resolved to existing files: {len(df)}, missing: {missing}")
    print(f"Label balance: real={(df['label']==0).sum()}, fake={(df['label']==1).sum()}")
    print(f"manipulation_type counts:\n{df['manipulation_type'].value_counts().to_string()}")
    print(f"Unique source identities: {df['source'].nunique()}")

    # Corruption spot-check on a random sample (opening all ~21k is too slow).
    sample = df.sample(n=min(sample_check_n, len(df)), random_state=42)
    bad = [row["video_path"] for _, row in sample.iterrows()
           if not check_readable(REPO_ROOT / "data" / row["video_path"])]
    print(f"\nCorruption spot-check: {len(sample)} sampled, {len(bad)} unreadable")
    for b in bad:
        print(f"  UNREADABLE: {b}")

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_MANIFEST, index=False)
    print(f"\nWrote full manifest ({len(df)} rows) to {OUT_MANIFEST}")


if __name__ == "__main__":
    audit()
