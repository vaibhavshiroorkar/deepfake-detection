# SETUP

How to set up this repository and what is needed for each supported capability. Written 2026-10-26 after the merge-readiness audit of commit 0704b3a. Everything here was read from the repository or the verified development environment; items that were not verified are marked "not verified". Nothing here grants a licence.

## 1. Environment (verified on the development machine)

| Item | Value | Source |
|---|---|---|
| OS | Windows 10 Pro 10.0.19045 | development machine |
| Python | 3.13.15 in `.venv` (project requires `>=3.13`) | pyproject.toml, `.venv` |
| Package manager | uv 0.12.12 | development machine |
| PyTorch | 2.14.1+cu130, from the `pytorch-cu130` index declared in pyproject.toml | pyproject.toml |
| GPU | NVIDIA, CUDA 13.0 wheels (RTX 5070 class stated by the user; see docs/EXPERIMENTS.md for the verified environment) | CLAUDE.md section 6 |
| ffmpeg / ffprobe | on PATH, 9.0.1 (needed by media tests and the canonical re-encode) | development machine |

CPU: the PyTorch wheels are the CUDA build. The unit tests were run with the GPU hidden (`CUDA_VISIBLE_DEVICES=""`) and pass; modules choose `cuda` when available and fall back to `cpu`. Speed on CPU and CPU-only wheels for a machine without NVIDIA drivers were not verified. No other OS or Python combination is claimed.

## 2. Install

```
uv sync
```

This installs the declared dependencies and the `dev` group (pytest, scikit-learn, matplotlib, playwright, requests, cryptography). Add `--extra c2pa` for real C2PA verification (section 4). Run code with `PYTHONPATH=src` (pytest already sets `pythonpath = ["src"]`).

## 3. Dependency declarations (updated 2026-10-26, second remediation pass)

`pyproject.toml` and `uv.lock` were regenerated with `uv lock` (network available). Before this pass the lock at commit 0704b3a was already out of date even for packages that pyproject.toml declared (flask, matplotlib, playwright were missing from it), so `uv sync --locked` failed at that commit. Classification, from the imports in `src/`, `scripts/` and `tests/`:

| Package | Declared as | Why |
|---|---|---|
| `transformers` >=5.20 | core dependency | DINOv2-small in the generated-video module (`genvideo/features.py`) and `dfd.sweep.encoders`, which the real inference adapter (`dfd/infer/real_adapter.py`) imports |
| `timm` >=1.0.30 | core dependency | `dfd.sweep.encoders`, used by the real inference adapter |
| `requests` >=2.34 | `dev` group | live E2E scripts and `tests/platform_live_face` only |
| `cryptography` >=50.0 | `dev` group | ephemeral test PKI of `tests/platform_c2pa` and `scripts/platform/live_c2pa_e2e.py` only |
| `c2pa-python==0.38.0` | optional extra `c2pa` | optional capability (section 4). Pinned to 0.38.0: the 0.38.1 release that uv picked first has no Windows wheel (only Linux and macOS wheels), 0.38.0 has `win_amd64` and was the version tested |

The lock diff is additive (1,223 lines added, none removed; existing pins such as torch 2.14.1+cu130 and numpy 2.5.4 are unchanged). `uv lock --check` passes. `UV_PROJECT_ENVIRONMENT=<empty dir> uv sync --locked --dry-run` (with and without `--extra c2pa`) resolves. A real install into an empty environment was not done (it would download several GB, including PyTorch CUDA wheels), so "setup works from a clean environment" is not verified beyond the dry runs.

Documentation checks use a different interpreter: they need `python-pptx` and `lxml`, present in the system Python 3.14.0 on the development machine and not in `.venv`, and not declared anywhere in the project.

## 4. Optional: C2PA verification

Install with `uv sync --extra c2pa` (declared as the optional extra `c2pa`, section 3; 0.38.0 was used, PyPI, MIT OR Apache-2.0, win_amd64 wheel, bundled C2PA SDK 0.91.0; DATA_LOG G3).

| Situation | What the platform reports |
|---|---|
| Package absent | Marker byte scan only. Finding `inconclusive` (markers present) or `no_credentials_found` (no markers); the module's tested scope says "no manifest was validated" and its limitations say no credential was verified. It never reports `credentials_valid` or `credentials_invalid`. |
| Package present, no trust anchor configured | Signature and hashes are checked. A correctly signed file gives `inconclusive` with observation `C2PA_VALID_SIGNER_UNTRUSTED`: integrity checked, signer not established. |
| Package present, anchor configured via the file named in `DFD_C2PA_TRUST_ANCHORS` (PEM) | Signer chains to that anchor: `credentials_valid` (relative to the operator's configuration; it does not make content authentic). No trust list is bundled. |
| Tampered after signing | `credentials_invalid` |
| No manifest | `no_credentials_found` |
| Unsupported file type or unreadable manifest | `inconclusive` with a warning naming the reason |

Tests: `tests/platform_c2pa/` (11 tests) is skipped as a whole when `c2pa` is not importable (`pytest.importorskip`); the signed video and audio tests also need ffmpeg. `tests/platform_provenance/test_c2pa_unavailable.py` simulates the missing package and runs either way.

## 5. Artifacts that are git-ignored and must exist locally

`data/` and `checkpoints/` are ignored (CLAUDE.md section 5). Hashes are recorded, not the files. Obtaining each file is only possible where the licence allows it and, for gated data, with your own access; this document gives no download links beyond those already recorded in docs/DATA_LOG.md.

| Artifact | Used by | Expected identity | How obtained (documented source) | If missing |
|---|---|---|---|---|
| `checkpoints/I-5-freeze-candidate/` (frozen v1 detector) | image, face-video and speech modules | `metadata.json` lists per-file SHA-256 (for example `head_T-I.npz` 42bbb349…, `weights_T-V.pt` e0c2154d…); the loader checks them (D-028, D-044) | produced by this project's training; not downloadable | those modules report unsupported or failed |
| `checkpoints/genvideo-E020/head.npz` | generated_video | file SHA-256 `b67d01ac0001ec69f53ce659c88ab8b4415bc3fb810ce30041651de3bfa7fafb`; weights equal the frozen `head_C2.npz` (`c0f4b6d3…fafb`); meta protocol `canonical-v3`, feature `cls_mean` (384) | rebuilt by `scripts/genvideo/install_head.py` from `head_C2.npz`; `head_C2.npz` comes from `scripts/genvideo/e020.py dev` on DeepAction | module status `unsupported` ("no trained generated-video detector is installed") |
| `data/_logs/encoder_weights.json` and the DINOv2-small directory it names (IMG-07, revision ed25f3a3…) | generated_video features | `model.safetensors` SHA-256 ae1e99fc… (DATA_LOG); the runner refuses an unverified record | `scripts/sweep/fetch_encoders.py` (network; not run in the audit) | module fails with "DINOv2-small weights not hash-verified" |
| `data/models/syncnet/syncnet_v2.model` + `SOURCE.json` | lip_sync | SHA-256 `961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`, 54,573,114 bytes; the module re-hashes and compares with SOURCE.json | official URL recorded in `SOURCE.json` and DATA_LOG G4 | module status `unsupported` ("SyncNet weights file is not installed") |
| `data/_prep_bench/models/` (YuNet `yunet_2023mar.onnx`) | lip_sync face detection | SHA-256 in DATA_LOG | OpenCV zoo URL in DATA_LOG | lip_sync fails at the face step |
| `data/deepaction_v1`, `data/df26`, `data/fakeavceleb`, other corpora | experiments and their scripts only | see DATA_LOG | gated or licensed; see section 7 | experiments cannot be reproduced; the platform does not need them |
| `data/_results/**`, `data/_canonical*`, `data/_features*` | experiment outputs | recorded in docs/EXPERIMENTS.md and the ledger | produced by the scripts named in the entries; sealed outputs must not be regenerated | the numbers in the documents cannot be re-derived from files |

## 6. Compatibility checks done at load time

| Check | Where |
|---|---|
| Generated-video head: feature name known and input size matches (21 or 384) | `GeneratedVideoModule.applicability` |
| Generated-video head: preprocessing protocol equals `canonical-v3` (`src/dfd/platform/genvideo/canonical.py`) | same; otherwise `unsupported` |
| Generated-video head: unreadable or corrupt npz gives `unsupported`, not a crash | same |
| SyncNet weights: file exists (`unsupported` if not) and SHA-256 equals `SOURCE.json` | `LipSyncModule.applicability`, `weights_sha256` |
| DINOv2 weights: hash-verified record | `DinoRunner._load` |
| Frozen v1 checkpoint: file hashes and metadata | `dfd.infer` loader (not modified in this work) |

## 7. Unresolved licence and terms questions (not resolved by this document)

- **SyncNet v2 weights** (Oxford VGG): no licence stated by the source. The code is MIT; the weights are used for an academic prototype at the user's instruction, research-only labelling, no permission inferred (DATA_LOG G4).
- **FakeAVCeleb v1.2**: released under a request form whose terms are not recorded in DATA_LOG; derived data may not be published (D-020). Do not redistribute clips or derived frames.
- **DF26**: evaluation only; its licence prohibits training, tuning, selection, calibration or threshold setting (DATA_LOG G2).
- **DeepAction**: generated folders CC BY 4.0, real folder under the Pexels licence (DATA_LOG G1).
- **C2PA trust list**: none is bundled and none was downloaded.

## 8. How a future validated generated-video head would be approved

The gate hashes the same bytes that it loads (one read), so the approval covers the artifact that is used. Today `APPROVED_HEADS` in `src/dfd/platform/modules/generated_video.py` is empty, so every head, including the installed E-020 C2 head, gives `inconclusive` whatever its score and whatever its metadata says (D-074). To allow definitive findings for a new head, all of the following are needed:

1. A preregistered validation experiment (hypothesis, success criterion and failure criterion written in docs/EXPERIMENTS.md before the run, committed as its own commit before the run), on data that was not used to train, select or set the threshold of the head, with at least 3 seeds where training is involved, intervals reported, and cross-dataset evidence (the E-020 head failed this on DF26: AUC 0.4957).
2. A result that meets the pre-stated criterion, logged with the experiment id.
3. A head file whose meta has `validated: true` (boolean) and whose `evaluation`, `tested_scope` and `limitations` text is consistent with that result (the gate additionally rejects, as a secondary fail-closed check and not as the source of trust, texts containing "not validated", "unvalidated", "did not generalise", "chance level" or "experimental").
4. A reviewed code change adding the SHA-256 of that exact head file to `APPROVED_HEADS`, with the experiment id as the value. Any later edit of the file, including its metadata, changes the hash and removes the approval.
5. Tests for the new head in `tests/platform_genvideo`, and an update of the module documentation and the dashboard text.

DF26 results must not be used to train or select a replacement head.

## 9. Running the checks (use separate commands so each exit code is visible)

Each command below was run on the development machine with the stated result recorded in docs/STATUS.md.

```
# tests (GPU hidden; PYTHONDONTWRITEBYTECODE avoids writing .pyc files)
CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/platform_genvideo -q
CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/platform_lipsync -q
CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/platform_c2pa tests/platform_provenance -q
CUDA_VISIBLE_DEVICES= PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests -q          # full suite
```

The group counts are subsets of the full suite; do not add them. The run header (visible without `-q`) prints the tested `git HEAD` and whether tracked files differ from it, so a saved log identifies its source state. Opt-in tests: `RUN_PARITY=1` and `RUN_LIVE_FACE=1` (need the GPU and local data). Live integration scripts: `scripts/platform/live_full_e2e.py`, `live_c2pa_e2e.py`, `live_face_e2e.py` start the app in-process on an ephemeral port; they check structure, not accuracy.

Documentation checks:

```
PYTHONIOENCODING=utf-8 python scripts/check_docs.py --numbers
```

Use the system Python that has `python-pptx` and `lxml` (the script says so in its docstring). Exit code 0 means no failed row; 1 means at least one failed row. With `.venv` Python the two optional checkers (`check_report`, `check_slides`) crash with `ModuleNotFoundError` and are counted as failures (16 pass, 2 fail on 2026-10-26): that is an interpreter choice, not a source defect. The rows for the paper, report and slides need the built PDF, DOCX and PPTX files that are produced by their build scripts.
