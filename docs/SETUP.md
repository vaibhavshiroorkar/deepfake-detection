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
| ffmpeg / ffprobe | on PATH, 9.0.1 (external tool, see section 1a) | development machine |

CPU: the PyTorch wheels are the CUDA build. To hide the GPU on this Windows machine set `CUDA_VISIBLE_DEVICES=-1`. An EMPTY value (`CUDA_VISIBLE_DEVICES=`) does NOT hide it here: measured on 2026-10-26 in the clean environment, an empty value left `torch.cuda.is_available()` True (with `device_count()` 0, an inconsistent state) while `-1` gave False. Test commands that say "GPU hidden" therefore use `-1`; earlier records in docs/STATUS.md that show an empty value ran with the GPU visible (the unit tests do not need or use it; a run with `-1` gave the same pass counts, 495 passed and 2 skipped on the safe subset, 2026-10-26). Modules choose `cuda` when available and fall back to `cpu`. Speed on CPU and CPU-only wheels for a machine without NVIDIA drivers were not verified. No other OS or Python combination is claimed.

### 1a. External prerequisites that `uv sync` does NOT install

Keep these five groups apart: (1) Python packages installed by `uv sync` (core and `dev` group); (2) the optional Python extra `c2pa` (section 4); (3) external executables on `PATH`, below; (4) downloaded browser binaries, below; (5) research artifacts (section 5), which are never installed automatically.

| Tool | Needed by | If absent | Check |
|---|---|---|---|
| `ffmpeg` and `ffprobe` | media fixtures in `tests/conftest.py` and `tests/prep`, signed video/audio tests of `tests/platform_c2pa`, the synthetic clip of the generated-video tests, `tests/platform_provenance`; and at run time the canonical re-encode of the generated-video module, audio extraction and media probing | the tests named above skip with "ffmpeg not on PATH" (not verified for every test that calls ffmpeg); the platform cannot analyse video or audio | `ffmpeg -version` |
| `pdftotext` (Poppler) | `scripts/check_docs.py` (the paper `paper/main.pdf` and the report preview `report/*preview*.pdf`; the slide rows read the PPTX, not a PDF) and `report/layout_check.py` (which also needs `pdftoppm`, and Microsoft Word once for the template PDF) | the paper rows of the documentation check cannot read the PDFs | `pdftotext -v` |
| `latexmk` with a TeX distribution | only to BUILD the paper (`paper/`); not needed for the checks if the built PDF exists | the paper cannot be rebuilt | `latexmk -v` |
| Playwright browser binary (Chromium) | the browser tests of `tests/web`, `tests/platform_dashboard` and `tests/platform_live_face`, and `scripts/platform/live_full_e2e.py --browser` `scripts/platform/live_face_e2e.py --browser` and `scripts/platform/ui_demo_run.py` (falls back to an installed Edge or Chrome, as the tests do) (`live_c2pa_e2e.py` has no browser mode) | the browser tests skip with "no usable browser for playwright" or "Chromium not available" (the fixtures also try an installed Microsoft Edge or Google Chrome) | `.venv/Scripts/python.exe -m playwright --version` shows the Python package only; the browser is separate |

The `playwright` Python package (in the `dev` group, version 1.64.1) does not contain a browser; it needs its own matching browser build (Chromium build 1248 for 1.64.1). Download it with `uv run playwright install chromium` (or `.venv/Scripts/python.exe -m playwright install chromium`); by default it goes under the user profile (`%LOCALAPPDATA%\ms-playwright`), or under the folder named by `PLAYWRIGHT_BROWSERS_PATH`. How the tests find a browser on the development machine (checked 2026-10-26): `%LOCALAPPDATA%\ms-playwright` holds only an older build (`chromium-1234`), which 1.64.1 cannot use; `tests/web/test_web_browser.py` sets `PLAYWRIGHT_BROWSERS_PATH=D:\cache\playwright` for the whole pytest process when that folder exists, and that folder holds `chromium-1248` (downloaded in earlier work); the dashboard browser fixtures additionally fall back to an installed Microsoft Edge (`channel="msedge"`) or Google Chrome (`channel="chrome"`), both present on this machine. So the 2026-10-26 runs used a browser that `uv sync` does not install. On a new machine run `playwright install chromium` (not verified from a fresh profile here) or rely on an installed Edge or Chrome for the dashboard tests; `tests/web/test_web_browser.py` itself tries only the Playwright Chromium and skips with "Chromium not available" without it. Check: `.venv/Scripts/python.exe -m playwright --version` shows the Python package; `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -c "from playwright.sync_api import sync_playwright as s; p=s().start(); p.chromium.launch(channel='msedge').close(); print('edge ok')"` checks Edge. A tool being present on the development machine does not mean a new machine has it.

## 2. Install

```
uv sync
```

This installs the declared dependencies and the `dev` group (pytest, scikit-learn, matplotlib, playwright, requests, cryptography, lxml, python-docx, python-pptx). Add `--extra c2pa` for real C2PA verification (section 4). Use `uv sync --extra c2pa` every time once C2PA is wanted: a later plain `uv sync` removes `c2pa-python` again, because uv removes every package that the requested groups and extras do not name (observed 2026-10-26). Run code with `PYTHONPATH=src` (pytest already sets `pythonpath = ["src"]`).

## 3. Dependency declarations (updated 2026-10-26, second remediation pass)

`pyproject.toml` and `uv.lock` were regenerated with `uv lock` (network available). Before this pass the lock at commit 0704b3a was already out of date even for packages that pyproject.toml declared (flask, matplotlib, playwright were missing from it), so `uv sync --locked` failed at that commit. Classification, from the imports in `src/`, `scripts/` and `tests/`:

| Package | Declared as | Why |
|---|---|---|
| `transformers` >=5.20 | core dependency | DINOv2-small in the generated-video module (`genvideo/features.py`) and `dfd.sweep.encoders`, which the real inference adapter (`dfd/infer/real_adapter.py`) imports |
| `timm` >=1.0.30 | core dependency | `dfd.sweep.encoders`, used by the real inference adapter |
| `requests` >=2.34 | `dev` group | live E2E scripts and `tests/platform_live_face` only |
| `cryptography` >=50.0 | `dev` group | ephemeral test PKI of `tests/platform_c2pa` and `scripts/platform/live_c2pa_e2e.py` only |
| `lxml` >=5.0 | `dev` group (added 2026-10-26, stage 1) | `scripts/check_docs.py` reads the report DOCX XML with it (`docx_parts`), so the report rows crash without it |
| `python-pptx` >=1.0 | `dev` group (added 2026-10-26, stage 1) | `scripts/check_docs.py` opens the slide deck with it and `slides/check_slides.py` and `slides/build_slides.py` import it; it also pulls in `xlsxwriter` |
| `python-docx` >=1.1 | `dev` group (added 2026-10-26) | `report/build_report.py` imports `docx`; before this it was run with an ad hoc `uv run --no-project --with python-docx --with lxml` |
| `c2pa-python==0.38.0` | optional extra `c2pa` | optional capability (section 4). Pinned to 0.38.0: the 0.38.1 release that uv picked first has no Windows wheel (only Linux and macOS wheels), 0.38.0 has `win_amd64` and was the version tested |

The lock diff of the second pass was additive (1,223 lines added, none removed; existing pins such as torch 2.14.1+cu130 and numpy 2.5.4 unchanged). Stage 1 (2026-10-26, after the clean-environment verification) added `lxml` and `python-pptx` to the `dev` group: the lock gained 120 lines and exactly three packages (lxml 6.1.3, python-pptx 1.0.2, xlsxwriter 3.3.0), no line was removed and no other package changed. `uv lock --check` passes.

Clean-install evidence (2026-10-26, not a fresh clone), two separate environments outside the repository, each built with `UV_PROJECT_ENVIRONMENT=<new dir> uv sync --locked --extra c2pa --python 3.13`: (a) from the lock of commit 5af239a with an empty package cache: 75 packages, about 11 minutes, exit 0; every declared package imported; the safe test suite gave 495 passed, 2 skipped, 13 data-dependent items deselected; the documentation check gave 16 pass, 2 fail (lxml and python-pptx missing). (b) from the stage 1 lock (lxml and python-pptx added): 78 packages, exit 0; only the documentation check (71 pass, 0 fail) and the test groups platform_modules, platform_genvideo, platform_lipsync, platform_c2pa and platform_provenance (111 passed, 1 skipped) were run there, not the whole safe suite. The lock has since gained python-docx (one package); environment (b) predates that. It was run on a machine that also had the git-ignored artifacts of section 5, so it does not show that a checkout without them works; see section 5 for what is then skipped or unsupported. Python packages installed by uv do not include the executables and browser of section 1a.

The documentation check needs `lxml` and `python-pptx`; since stage 1 they are in the `dev` group, so `uv sync` provides them. Before stage 1 they were only in the system Python.

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

Run from the repository root. `CUDA_VISIBLE_DEVICES=-1` hides the GPU on this Windows machine (section 1; an empty value does not). `PYTHONDONTWRITEBYTECODE=1` avoids writing `.pyc` files.

```
# tests (GPU hidden)
CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/platform_genvideo -q
CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/platform_lipsync -q
CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/platform_c2pa tests/platform_provenance -q
CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests -q          # full suite
```

The group counts are subsets of the full suite; do not add them. The run header (visible without `-q`) prints the tested `git HEAD` and whether tracked files differ from it, so a saved log identifies its source state. Opt-in tests: `RUN_PARITY=1` and `RUN_LIVE_FACE=1` (need the GPU, so do not hide it, and local data). Live integration scripts: `scripts/platform/live_full_e2e.py`, `live_c2pa_e2e.py`, `live_face_e2e.py` start the app in-process on an ephemeral port; they check structure, not accuracy.

The full suite also contains 13 tests (in 10 functions of `tests/data`) that read the real dataset inventories under `data/`, including one that hashes real files; on the 2026-10-26 clean-environment run they were excluded on purpose (deselected), the rest gave 495 passed and 2 skipped. Without those datasets most of them skip by design (they test `REAL_ROOT` paths and skip when absent); that was not tested here.

Test dependency on ignored artifacts: `tests/platform_modules/test_modules.py` no longer reads `checkpoints/I-5-freeze-candidate/metadata.json`; it uses synthetic metadata (the wrapper tests only check that metadata values pass through). The two tests of the installed generated-video head skip with "no installed head in this checkout" if `checkpoints/genvideo-E020/head.npz` is missing, and the SyncNet weight-load test skips if the weights are missing. Among the tests, the real frozen checkpoint is used only by the opt-in `test_parity.py` and `test_live_face.py` runs; the live E2E scripts and the dashboard in live mode also load it. `test_real_checkpoint_metadata_has_the_fields_the_wrappers_read` (default suite) reads only the real `metadata.json`, checks the keys and types the wrappers consume, and skips when it is absent. Side effect of the opt-in `test_parity.py`: it writes `data/_results/platform/module_parity.json` on every run (overwriting the previous one). A genuinely fresh clone (without any ignored file) has still NOT been run end to end.

Documentation checks:

```
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/check_docs.py --numbers
```

Exit code 0 means no failed row; 1 means at least one failed row. It needs `lxml` and `python-pptx` (dev group), `pdftotext` on `PATH` and `Pillow` (installed through the lock), and it reads the built paper PDF, the report DOCX and its preview PDF, and the slide PPTX, which are git-ignored build outputs of the build scripts and must exist locally. A failed import of a checker is reported as a failed row ("checker crashed"), not hidden. Result in a clean environment built from the stage 1 lock (2026-10-26): 71 pass, 0 fail, exit 0 (this was on the machine that has those built files; it says nothing about a checkout that lacks them). With the earlier lock, before `lxml` and `python-pptx` were declared, the same command gave 16 pass, 2 fail in a clean environment.
