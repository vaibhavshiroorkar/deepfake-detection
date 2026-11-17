# Security

The website takes video uploads from visitors, decodes them with FFmpeg and runs downloaded model files, all on Vaibhav's home PC. This file lists what is worth protecting, each weakness, its countermeasure and its status. Status labels: **verified** (checked, with how), **assumed** (believed, with what would check it), **unknown**. Review it whenever an input path, the hosting or a dependency changes.

## What is worth protecting

- The home PC: its files, its accounts and the research data under `data/` (licensed, not redistributable).
- Visitors' videos: they may show real people, so they must not be kept.
- Secrets: `RESEND_API_KEY`, `CONTACT_TO` and `DDF_ADMIN_TOKEN`.

## Weaknesses and countermeasures

| Weakness | Countermeasure | Status |
|---|---|---|
| Who can reach the site | For the exhibition the server listens on 127.0.0.1 only and is shared through Tailscale Serve, so only Vaibhav's own logged-in devices reach it; nothing is opened to the public internet | Listening address **verified** in `scripts/demo_host.ps1`; Tailscale's access control **assumed** (its documented behaviour; check by opening the address from a device outside the account) |
| A hostile video exploits FFmpeg's decoders | Only `.mp4`, `.mov` and `.webm`; at most 90 MB, counted while streaming; `ffprobe` must find exactly one video and at most one audio stream, with a 20 s time limit; one job runs at a time; the server runs as an ordinary user, not an administrator | Upload rules **verified** by `tests/test_site.py`. The FFmpeg calls that decode and re-encode for scoring have **no time limit**, so a slow-to-decode file could hold the one worker; a decoder bug itself is **unknown** risk. Keep FFmpeg updated |
| Uploads kept on disk | The upload and every derived file are deleted before the result can be read, and on any error | **Verified** by `tests/test_web_detector.py` and a live run on 11-17 (no files left) |
| Flooding the detector | At most three uploads wait behind the running one; then "busy"; at most 10 uploads an hour per address (500 for the private demo, where every request comes through one local proxy) | **Verified** by tests |
| Internal details in error messages | Failures are logged on the PC; the visitor sees a fixed message | **Verified** by `tests/test_web_detector.py` |
| A changed model file runs code when loaded | Branch and sync checkpoints load with `weights_only=True`. The AV-HuBERT loader must unpickle with `weights_only=False`, and its unpickler is not a sandbox, so the website refuses that file unless its SHA-256 equals the released checkpoint's | **Verified**: hash check and its test (`verified_avhubert`), 11-17. Other model files (MediaPipe, MTCNN) are loaded by their libraries' own formats: **assumed** safe |
| Secrets in the repository | `.env` is git-ignored; secrets come from the host's environment | **Verified** with `git check-ignore`, 11-17 |
| Contact form spam | A hidden field only bots fill in; 5 messages an hour per address | **Verified** by tests |
| Admin page | Needs `DDF_ADMIN_TOKEN`; not used for the exhibition | **Assumed** adequate for a private demo; not reviewed for public hosting |
| Research data shown publicly | Demo clips stay on our own machines (`data/` is git-ignored); only result-card screenshots, with no video frames, are committed | **Verified**: `git check-ignore data/demo/upload/x.mp4` |

## Before any public hosting

Add time and memory limits to every FFmpeg call, replace the AV-HuBERT unpickler with an allowlist, review the admin page, and run the uploads in a container without access to `data/`. These are in the pending list for after the exhibition.

## Reporting a problem

Write to Vaibhav through the site's Contact page, or open an issue in the repository without exploit details and ask for a private channel.
