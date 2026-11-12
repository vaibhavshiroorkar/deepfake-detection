# Media Analysis, front end v2 (deliverables_v2/website)

A redesigned, isolated front end for the platform dashboard. It reuses the existing backend unchanged: `serve_v2.py` builds the app with
`dfd.platform.dashboard.serve.create_app` (same /api routes, same live mode, same 512 MB limit and job queue, same Evaluation Lab backend) and
replaces only the static-page routes. Nothing under `src/`, `scripts/`, `tests/`, `docs/`, `data/`, `ledger/`, `paper/`, `report/`, `slides/`
or `assets/` was edited. (Correction 2026-10-29: this folder was committed in 69de794 and merged into main as db38f98; the screenshots under test_evidence/ stay local and are not in git.)

## Start

```
PYTHONPATH=src .venv/Scripts/python.exe deliverables_v2/website/serve_v2.py 8012 --live
```

Open http://127.0.0.1:8012/ (binds 127.0.0.1 only). Without `--live` the page offers the MOCK demo scenarios only (same as the original).
Ports: 8001 frozen v1 app, 8002 original dashboard, 8012 this site. The old asset paths (/dashboard.js, /dashboard.css, /evallab.js,
/evallab.css) answer 404 on this server; the new files are served from /assets/ (allowlist of three names).

## Files

| File | What it is |
|---|---|
| `serve_v2.py` | Wrapper: same app, new static routes (`/`, `/assets/app.css`, `/assets/app.js`, `/assets/evallab.js`). |
| `static/index.html` | Page: header with three views (Analyse a file, Evaluation Lab, About the checks), upload stage, progress stage, results, About view. |
| `static/app.css` | All styling and the colour tokens. The cream light theme always applies, also when the operating system is in dark mode (user instruction 2026-10-28; the automatic `prefers-color-scheme` switch was removed). The dark tokens remain only as an unused explicit opt-in (`data-theme="dark"`). Also restyles the Evaluation Lab. |
| `static/app.js` | Vanilla JS, no build step, no CDN. Renders reports, the "Checks by outcome" chart, score bars, upload and progress. |
| `static/evallab.js` | Byte-identical copy of `src/dfd/platform/dashboard/evallab.js` (verified with `cmp`): every API call and safeguard message is the original. |
| `contrast_check.py` | Computes WCAG contrast of the CSS tokens (light and dark); exit code 1 on a failure. |
| `ui_v2_check.py` | Playwright (Edge) check of the new page against the running server. |
| `test_evidence/` | `ui_v2_run.json` (full run record), `screens/` (PNG screenshots), `contrast_table.md`, `server_8012.log`. |

Branding: the original dashboard has no logo image, only a CSS mark and the name "Media Analysis" with its subtitle; both are kept (the mark
is recoloured to the new brown). No brand was invented.

## What changed for the user

- **Content type.** The original page never asked what the user is analysing (it reads the file extension). The new page adds a client-side
  choice "What are you analysing?" (Detect from the file / Image / Video / Audio). It filters the file picker's accepted types, lists the
  checks that can run for that kind (from `/api/vocabulary`), and, if a chosen file does not match the choice, says so and keeps the
  Analyse button disabled until the choice or the file is changed. The request is unchanged: the browser sends one multipart field,
  `file` (verified in the run). Routing is the server's, as before.
- **Stages.** Upload (type choice, drop zone, supported types and the 512 MB limit read from `/api/mode`, what will run), then progress
  (current check from the job's `progress`, "N of M checks done", a list of checks as they are seen running), then results. Errors
  (unsupported type, damaged file, queue full) show the server's error assessment label and code, with a button back to the upload stage.
- **Results.** File overview, "Checks by outcome" chart next to the server-written summary, then cards grouped by outcome. A card shows the
  module name, the `assessment.verdict_label`, an uncertainty chip, one line of meaning, and for detector modules a score bar. Folded
  below: "How this was reached", "Limitations", "Technical details" (raw score, score kind, whether it is a probability, threshold, model,
  version, weights SHA-256, licence, evaluation, tested scope, warnings). Checks that did not apply are folded into one list.
- **Lip-sync.** The offset curve with the band (-6 to +2 frames, taken from the report's own `band`) shaded, on the card.
- **Disclaimer.** "What this report cannot tell you": the report's fixed statement plus the server's `not_established` and probability
  statements. "Analyse another file" at the bottom of the report. MOCK scenarios keep the hatched MOCK banner and a MOCK badge on every card.

## The donut chart, and why it is honest

Inspected: `schema.py` has no probability field; `verdict.py` writes `assessment.calibration.is_probability = False` for every module
(also checked on every report of the browser run: the set of values was exactly `{False}`). So no module provides a valid fake-versus-real
probability. The page never draws one, never uses `score` or `1 - score`, and never combines scores.

The donut is "Checks by outcome". For each outcome group of THIS report (merits attention, nothing found within tested scope, credential
verified, inconclusive, file information, experimental no finding made, did not apply, not available, did not complete) it shows how many
checks fall in it. The counts are the lengths of `report.summary.groups[].modules`, exactly the grouping the original page uses (fallback
for old reports: the group of each module result). No score and no report key is involved, so no key was added and no overall number
exists. The caption reads: "This chart counts the checks in this report by outcome. It is not a probability, a score or an overall
verdict." Accessibility: the SVG has `role="img"` and an `aria-label` listing the counts; the legend is a real list of label plus count
(each entry links to its group); slices differ in pattern (stripes, dots, lines) as well as colour; the centre shows the total.
The browser check asserts that the legend counts equal the module counts per group from the API JSON and that the centre equals the
number of modules.

## Score bars (decision to review)

For detector modules whose `assessment.raw.score_kind` is `model_output`, the card face shows a 0 to 1 bar with the decision-line marker,
labelled "Model output, not a probability" with the note "A bar position is not a chance of being fake". The original page deliberately
kept the raw score off the card face (Technical details only); showing it on the face follows this task's brief. Tagged
`decided autonomously - needs user review`. Experimental modules keep the original's rule: the number is not on the card face; a muted,
dashed "Experimental ranking score, no finding" bar sits inside Technical details only. Modules whose score kind is `none` (lip-sync) get
no bar (a first version showed one for lip-sync; the check found it and it was fixed). The bar fill is the brown primary for every module,
never red or green. A mock module's bar says "MOCK number, not a measurement".

Other safeguards kept: no report-level verdict and no overall score; "nothing found" and "no credentials found" are neutral beige (the
original tinted the provenance tier teal; this version reserves teal for verified credentials only); the face-video flag reads exactly
"Possible manipulation"; experimental cards read "Experimental signal observed, no reliable verdict available", have a dashed outline and
are grouped under "Experimental, no finding made", never under "Merits attention"; `APPROVED_HEADS` and all backend code are untouched;
the Evaluation Lab is the original script (no run was started in the check).

## Design tokens

Warm and calm: cream page, ivory cards, beige panels, warm-brown primary, restrained terracotta accent, neutral borders. System font stack
(the serif for headings is the platform's existing stack), no web fonts, no images, almost no gradients (only the MOCK banner stripe and
the experimental bar stripe). Spacing scale 4/8/12/16/24/32/48/72 px; radii 12 (cards), 16 (panels), pill for buttons and chips.

| Token | Light | Dark | Use |
|---|---|---|---|
| `--bg` | #F6F0E4 | #1B1612 | page (cream) |
| `--card` | #FFFCF6 | #241E18 | cards and panels (ivory) |
| `--card-2` | #F3EBDD | #2D261E | beige panels, drop zone |
| `--border` | #E4D9C6 | #40362B | decorative separators (never carry meaning alone) |
| `--control-border` | #8A7861 | #8F7F6B | control edges, drop zone, bar track |
| `--text`, `--text-2`, `--muted` | #2A2018, #4A3D31, #665747 | #F2EADD, #D9CDBB, #B4A593 | text levels |
| `--primary`, `--primary-ink` | #5C3B27, #FFFCF6 | #E6C7A6, #2A1C10 | warm brown, buttons, bar fill |
| `--accent`, `--accent-text` | #B07650, #85472B | #D39A74, #E8B391 | non-text marks, kicker text |
| outcome label pairs | attn #9A2D1C on #FAE8E1; warn #774608 on #F6EACF; none #4A4034 on #ECE4D5; inc #45497D on #ECECF4; sys #76400F on #F5E7D8; cred #1D5A57 on #DFEDEA; mock #8A1A52 on #F7E6EE; exp #3A4A7B on #E5E9F4 | lighter variants | outcome labels |
| chart slices | #B03A26, #6E5F50, #2C7873, #5B63A8, #9A8156, #7A4E9B, #8B8374, #A15E26 | lighter variants | donut |

Contrast (WCAG 2.x), computed by `contrast_check.py` (also saved as `test_evidence/contrast_table.md`; 0 failures; text needs 4.5,
non-text marks need 3):

| Pair (foreground on background) | Use | Needs | Light | Dark |
|---|---|---|---|---|
| text #2A2018 / #F2EADD on bg #F6F0E4 / #1B1612 | body text on page | 4.5 | 14.04 | 15.03 |
| text #2A2018 / #F2EADD on card #FFFCF6 / #241E18 | body text on card | 4.5 | 15.56 | 13.81 |
| text #2A2018 / #F2EADD on card-2 #F3EBDD / #2D261E | text on beige panel | 4.5 | 13.46 | 12.50 |
| text-2 #4A3D31 / #D9CDBB on bg | secondary text on page | 4.5 | 9.24 | 11.45 |
| text-2 on card | secondary text on card | 4.5 | 10.24 | 10.52 |
| text-2 on card-2 | secondary text on beige | 4.5 | 8.86 | 9.52 |
| muted #665747 / #B4A593 on bg | muted text on page | 4.5 | 6.13 | 7.47 |
| muted on card | muted text on card | 4.5 | 6.79 | 6.86 |
| muted on card-2 | muted text on beige | 4.5 | 5.88 | 6.21 |
| muted on none-bg #ECE4D5 / #322A21 | muted text on neutral chip | 4.5 | 5.51 | 5.87 |
| primary #5C3B27 / #E6C7A6 on card | brown link or summary on card | 4.5 | 9.72 | 10.27 |
| primary on bg | brown link on page | 4.5 | 8.77 | 11.18 |
| primary on primary-soft #EFE3D2 / #3A2E22 | brown on soft brown | 4.5 | 7.86 | 8.21 |
| primary-ink #FFFCF6 / #2A1C10 on primary | button text on brown | 4.5 | 9.72 | 10.29 |
| text on primary-soft | text on soft brown | 4.5 | 12.58 | 11.04 |
| accent-text #85472B / #E8B391 on bg | kicker on page | 4.5 | 6.31 | 9.65 |
| accent-text on card | kicker on card | 4.5 | 6.99 | 8.86 |
| accent-text on card-2 | accent on beige | 4.5 | 6.05 | 8.02 |
| text-2 / text / muted on accent-soft #F5E6DA / #36291F | limits panel | 4.5 | 8.59 / 13.06 / 5.70 | 8.97 / 11.77 / 5.85 |
| attn-fg on attn-bg | merits attention label | 4.5 | 6.40 | 8.05 |
| warn-fg on warn-bg | weak evidence or warning label | 4.5 | 6.60 | 8.28 |
| none-fg on none-bg | nothing-found label (neutral) | 4.5 | 8.02 | 9.17 |
| inc-fg on inc-bg | inconclusive label | 4.5 | 7.14 | 7.80 |
| sys-fg on sys-bg | did not complete label | 4.5 | 6.89 | 7.88 |
| cred-fg on cred-bg | credential verified label | 4.5 | 6.58 | 7.76 |
| mock-fg on mock-bg / on card | MOCK badge and banner | 4.5 | 7.45 / 8.73 | 8.42 / 8.75 |
| exp-fg on exp-bg / on card | experimental label | 4.5 | 7.07 / 8.39 | 7.75 / 9.20 |
| warn-fg on card; attn-fg on card | uncertainty marker, error text | 4.5 | 7.70; 7.41 | 9.84; 8.62 |
| text on none-bg; text on warn-bg; text-2 on none-bg | text on neutral and warning panels | 4.5 | 12.61; 13.34; 8.30 | 11.81; 11.63; 9.00 |
| control-border #8A7861 / #8F7F6B on card / bg / card-2 | control edges, drop zone | 3 | 4.15 / 3.75 / 3.59 | 4.25 / 4.63 / 3.85 |
| primary on card; text on card | score bar fill; threshold marker | 3 | 9.72; 15.56 | 10.27; 13.81 |
| chart slices on card (attention, nothing found, credential, inconclusive, file information, experimental, did not apply, not available, did not complete) | donut | 3 | 5.89, 6.00, 5.07, 5.40, 3.63, 6.05, 3.66, 3.66, 4.95 | 5.32, 6.00, 6.41, 5.89, 7.19, 5.94, 5.28, 5.28, 6.45 |
| accent on card | brand dot (non-text) | 3 | 3.69 | 6.79 |

The full machine-generated table (every pair with its hex values) is `test_evidence/contrast_table.md`.

## What was tested (2026-10-28, this machine, Edge 156 via Playwright)

| Command | Result |
|---|---|
| `PYTHONPATH=src .venv/Scripts/python.exe deliverables_v2/website/serve_v2.py 8012 --live` (background) | started; GET / 200; /api/mode live true; /dashboard.js, /dashboard.css, /evallab.css 404; /assets/app.js 200; /assets/serve.py 404 |
| `PYTHONPATH=src .venv/Scripts/python.exe deliverables_v2/website/ui_v2_check.py --base http://127.0.0.1:8012 --out deliverables_v2/website/test_evidence` | `problems: 0`, 28 s, no console or page errors on any page; record in `test_evidence/ui_v2_run.json` |
| `.venv/Scripts/python.exe deliverables_v2/website/contrast_check.py` | 0 failures |
| `CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe -m pytest tests/platform_dashboard tests/platform_evallab tests/platform_verdict -q -p no:cacheprovider` | 194 passed in 28.83 s. These test the original dashboard and the backend; they show the backend is unaffected, not that the new page passes them. |
| static house-rule scan of the new static files (no em or en dash, no http URL other than the SVG namespace, no innerHTML or eval, none of the original tests' forbidden words) | clean |

`ui_v2_check.py` covered, at 1366x900 and 390x844 (no horizontal overflow found at either size on any page): landing and accepted types
with the 512 MB limit; the type selector (picker filter, mismatch message and blocked button, re-enabled after a matching choice); real
uploads of the demo inputs: image (5), face-swap video (1), talking-head video with audio (3), speech audio (4), synthetic no-face tone
video (6); error paths: `.txt` (UNSUPPORTED_MEDIA_TYPE) and a corrupt video (UNREADABLE_FILE), each with the way back to the upload stage.
For every report: the upload request carries exactly one field `file`; `schema.validate_report` is empty and no `FORBIDDEN_REPORT_KEYS`
key is present; `is_probability` is false on every module; the donut legend counts equal the per-group module counts from the API; no
percent sign next to a number in the report text; every sentence mentioning probability, percent, chance or confidence contains a
negation (those sentences are listed in the JSON for review); no report-level verdict label; the card label equals
`assessment.verdict_label`; the face-video flag reads "Possible manipulation" exactly when the finding is `indicates_manipulation`;
experimental cards read the fixed label, are not styled or grouped as a flag, show no bar on the face and no finding wording; detector bars
exist and say "not a probability"; no bar for a score kind that is neither model output nor ranking score; the lip-sync curve and band are
drawn (band [-6, 2]); "How this was reached" and "Technical details" expand and collapse; navigation to the About page and back keeps the
report. Also: the Evaluation Lab (read only: catalogue of 15 score files, a development block chosen, settings form shown; no run was
started), the direct `#lab` link, the About page, two MOCK scenarios (banner, MOCK badge on every card, chart counts) with "Analyse another
file" returning to the upload stage, one dark colour-scheme screenshot, and the progress stage with a stubbed `/api/jobs` (routes
intercepted inside the browser, no analysis) because real analyses finish in under four seconds and rarely show a check mid-way.

Screenshots I opened and inspected: landing 1366 and 390, face-swap report 1366 and 390, expanded card, talking-head report and lip-sync
card, Evaluation Lab, MOCK scenario at 390, error page, progress (real and stubbed, 390), About. Problems seen there and fixed: single-card
groups left half the width empty (now two to a row); "No content credentials found" had a teal tint (now neutral); lip-sync showed a bar
labelled "Model output" although its score kind is none (removed); the chart legend was cramped at 390 px (stacked); "RUNNING" wrapped
(fixed); an "a image" grammar slip (fixed).

## Not tested, known issues

- Not tested: Firefox and Safari; a real screen reader (the chart has `role="img"` text and a legend list, but no assistive technology was
  run); a keyboard-only walk-through (focus rings are styled, not exercised); print output; the Evaluation Lab's run, compare and export
  steps (they write run records on the backend, outside this folder, so they were not started); dark mode beyond one screenshot and the
  contrast table; uploads near 512 MB; the queue-full and timeout paths; the live "Queued" state.
- The step list during a run is limited by the API: it lists only checks seen while polling (1 s); the "N of M checks done" counter is exact.
- Mock reports have no decision line, so their bars have no marker (stated on the card).
- The scenario list lives on the upload stage; to switch scenario, use "Analyse another file" first.
- The "best 0" label on the lip-sync curve sits close to the plotted point.
- Full-page screenshots of the Evaluation Lab show the sticky header overlapping content mid-page (a capture artefact, not seen in use).
- Nothing was committed. `git status` shows `docs/DECISIONS.md` as modified after staging; this workstream did not touch it.
