# Security

The website takes video uploads from visitors, decodes them with FFmpeg and runs downloaded model files, all on Vaibhav's home PC. This file lists what is worth protecting, each weakness, its countermeasure and its status. Status labels: **verified** (checked, with how), **assumed** (believed, with what would check it), **unknown**. Review it whenever an input path, the hosting or a dependency changes.

## What is worth protecting

- The home PC: its files, its accounts and the research data under `data/` (licensed, not redistributable).
- Visitors' videos: they may show real people, so they must not be kept.
- Secrets: `RESEND_API_KEY`, `CONTACT_TO` and `DDF_ADMIN_TOKEN`.

## Weaknesses and countermeasures

| Weakness | Countermeasure | Status |
|---|---|---|
| Who can reach the site | The server listens on 127.0.0.1 only. `demo_host.ps1 -Action start` shares it with Vaibhav's own Tailscale devices (Serve). `-Public` shares it with the internet through Tailscale Funnel, so the static frontend on GitHub Pages (https://vaibhavshiroorkar.github.io/deepfake-detection/) can call it; anyone with the address can then upload. `-Action stop` turns either off | Listening address **verified** in `scripts/demo_host.ps1`. Public reach **verified** 11-18: an upload through Funnel's public relay (176.58.90.63) returned a result. Funnel was on from 11-18 evening |
| Other websites using the API from a visitor's browser | CORS lets only the Pages origin read the API's answers (`DDF_CORS_ORIGINS` overrides it). This does not stop direct requests, which the rows below limit | **Verified** by `tests/test_export_site.py` and through Funnel: the Pages origin gets `Access-Control-Allow-Origin`, another origin gets none |
| A hostile video exploits FFmpeg's decoders | Only `.mp4`, `.mov` and `.webm`; at most 90 MB, counted while streaming; `ffprobe` must find exactly one video and at most one audio stream, with a 20 s time limit; one job runs at a time; the server runs as an ordinary user, not an administrator | Upload rules **verified** by `tests/test_site.py`. Every FFmpeg and FFprobe call an upload reaches now has a time limit (300 s), so a slow-to-decode file cannot hold the one worker forever: **verified** by `tests/test_upload_path_timeouts.py`, 11-18. No memory limit yet. A decoder bug itself is **unknown** risk. Keep FFmpeg updated |
| Uploads kept on disk | The upload and every derived file are deleted before the result can be read, and on any error | **Verified** by `tests/test_web_detector.py` and a live run on 11-17 (no files left) |
| Flooding the detector | At most three uploads wait behind the running one; then "busy"; at most 10 uploads an hour per address (500 for the demo, where every request comes through one local proxy) | **Verified** by tests. Behind Funnel every visitor shares that one address, so one person can use up the 500 an hour for everyone: a denial of service, not a compromise. Per-visitor limits need the proxy's forwarded address: **open** |
| Internal details in error messages | Failures are logged on the PC; the visitor sees a fixed message | **Verified** by `tests/test_web_detector.py` |
| A changed model file runs code when loaded | Branch and sync checkpoints load with `weights_only=True`. The AV-HuBERT loader must unpickle with `weights_only=False`, and its unpickler is not a sandbox, so the website refuses that file unless its SHA-256 equals the released checkpoint's | **Verified**: hash check and its test (`verified_avhubert`), 11-17. Other model files (MediaPipe, MTCNN) are loaded by their libraries' own formats: **assumed** safe |
| A third-party detector (v2 website models, branch `web-models-v2`) is changed or runs code | The live models are named in `results/web-models.json` with a pinned SHA-256, and the site refuses a file with another hash before reading it (`candidates.verified`). AntiDeepfake voice weights are safetensors (no code). The CLIP face model is TorchScript: `torch.jit.load` runs TorchScript code, not pickled Python, but it is code from the model's author (MIT licence, Hugging Face `yermandy/deepfake-detection`). Effort (`.pth`) loads with `weights_only=True` | **Verified**: hash refusal tested (`test_a_live_model_file_with_another_hash_is_refused`); the AntiDeepfake XLS-R 1B file's hash equals the one Hugging Face lists, 11-19. **Verified** 11-19: the TorchScript graphs of the CLIP face model hold only `aten` (tensor) and `prim` (control) operations, 2,622 nodes, no file, system or network calls |
| Secrets in the repository | `.env` is git-ignored; secrets come from the host's environment | **Verified** with `git check-ignore`, 11-17 |
| Contact form spam | A hidden field only bots fill in; 5 messages an hour per address | **Verified** by tests |
| Admin page | Needs `DDF_ADMIN_TOKEN`; not used for the exhibition | **Assumed** adequate for a private demo; not reviewed for public hosting |
| Research data shown publicly | Demo clips stay on our own machines (`data/` is git-ignored); only result-card screenshots, with no video frames, are committed | **Verified**: `git check-ignore data/demo/upload/x.mp4` |

## Before any public hosting

The site is public through Funnel when `demo_host.ps1 -Public` is running (decided by Vaibhav 11-18). Done: time limits on every FFmpeg call on the upload path, and CORS limited to the Pages origin. Still open, in the pending list: memory limits on FFmpeg, per-visitor rate limits behind the proxy, an allowlist unpickler for AV-HuBERT (its file is checked by hash, and an upload cannot replace it), an admin-page review, and running uploads in a container without access to `data/`. Until those are done, run it public only while someone is watching, and stop it otherwise.

## Reporting a problem

Write to Vaibhav through the site's Contact page, or open an issue in the repository without exploit details and ask for a private channel.
