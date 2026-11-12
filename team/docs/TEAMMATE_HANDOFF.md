# Teammate hand-over: what is in git, what is on Drive, what you rebuild (2026-10-31, D-099)

This page is for a team member who clones the repository on a machine that already holds the datasets. Read `CLAUDE.md`, then `docs/STATUS.md`, then `docs/SETUP.md`.

## 1. In git (clone the repository)

Everything under `src/`, `scripts/`, `tests/`, `docs/`, `deliverables_v2/`, `submission/`, `Exhibition/`, `ledger/`, `paper/`, `report/`, `slides/`, `assets/`, plus `pyproject.toml` and `uv.lock`. Install with `uv sync` (Python 3.13), see `docs/SETUP.md` section 1.

## 2. On the shared Google Drive folder (unzip into the repository root)

Each zip keeps repository-relative paths, so extracting it in the repository root puts every file in place. Each zip carries a `MANIFEST_*.csv` with the SHA-256 and size of every file; check them after extraction.

| Bundle | Contents | Needed for |
|---|---|---|
| `dfd_01_weights_splits.zip` | `checkpoints/I-5-freeze-candidate` (frozen v1 detector), `checkpoints/genvideo-E020`, `checkpoints/modern-E027`, `checkpoints/modern-E029`, `checkpoints/I-0-mvp`, `data/_splits`, `data/_manifests`, `data/_prep_bench` (face detector ONNX files, benchmark CSV and JSON; dataset-derived images left out) | Running the platform; using the same source-identity splits (CLAUDE.md rule 3) |
| `dfd_02_experiment_records.zip` | `data/_results`, `data/_final` (sealed test scores, features and evaluation JSON), `data/_final_dry`, `data/_evallab`, `data/_logs` | The Evaluation Lab and re-deriving every number in the documents; sealed outputs must not be regenerated |

The loader checks the checkpoint hashes in each `metadata.json` (D-028, D-044), so a damaged file is reported, not silently used.

## 3. Not uploaded: get or rebuild these yourself

| Item | Why not uploaded | How to get it |
|---|---|---|
| Raw datasets (`data/ffpp`, `data/celebdf_v2`, `data/lavdf`, `data/fakeavceleb`, `data/df26`, others) and their archives | You already have them; their terms forbid redistribution | Place them at the paths listed in `docs/DATA_LOG.md` |
| `data/models/syncnet/syncnet_v2.model` | Weights licence not stated by the source (DATA_LOG G4) | Download from the official URL in DATA_LOG G4; SHA-256 `961e8696…5442`. Copy `SOURCE.json` from the original machine or recreate it as in DATA_LOG |
| Hugging Face encoders (EfficientNet-B4, DINOv2-small and others) and `data/_logs/encoder_weights.json` paths | Public, pinned by revision | `scripts/sweep/fetch_encoders.py` downloads and hash-checks them and rewrites `encoder_weights.json` with your local paths |
| `data/_demo_inputs/demo_set/` | Copies of dataset media | `scripts/platform/make_demo_set.py` rebuilds it from your datasets |
| `data/_crops*`, `data/_features*`, `data/_canonical*`, `data/_cache` | Large caches derived from dataset videos | Rebuilt by the scripts named in each `docs/EXPERIMENTS.md` entry |
| `checkpoints/I-1`, `I-1b`, `I-6` (7.6 GB) | Earlier iterations, not used by the platform | Ask the original machine's owner if an experiment needs them |
| `Document Template/` | Names and roll numbers of an earlier project (D-037) | Shared privately inside the team only; needed only to rebuild the deck |

## 4. Check that it works

`PYTHONPATH=src .venv/Scripts/python.exe deliverables_v2/website/serve_v2.py 8012 --live`, then open `http://localhost:8012`. The test suite and its expected count are in `docs/TESTING_RECORD.md`.
