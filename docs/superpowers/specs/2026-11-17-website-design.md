# Website design: Deepfake Check

Status: draft for Vaibhav's approval, 2026-11-17. The look is in [`docs/DESIGN.md`](../../DESIGN.md). Open work is tracked in [`docs/PENDING.md`](../../PENDING.md), section 5.

## Intent

The site is for people who want to know whether a video is real or fake. They arrive with a clip, upload it, and leave with an honest answer: what the checks saw, how sure the answer is, and what it cannot tell them.

The research is the reason that answer can be trusted, so it supports the detector instead of leading. The site's message is the paper's: detectors were partly grading the file, not the face, and an honest detector says when it does not know.

## Goals

- A visitor can upload a clip and get one of four outcomes, with the three check scores and a plain explanation, within about two minutes on the free host.
- Every number on the site comes from `results/numbers.json`, so the site never disagrees with the paper.
- The detector runs exactly the P-001 pipeline: the same neutralizer, preprocessing hash, models and thresholds rule.
- It works on a phone, from the keyboard, and with motion turned off.
- Everything on the current web app carries over, updated to P-001 (see "What carries over").

## Non-goals (for the first version)

- Accounts, saved history, or an API for other services.
- Checking a link (YouTube and so on). Fetching URLs on the server invites abuse, and the clip rules matter.
- Several faces, long videos, or live video.
- A light theme. The look is dark by design; a light theme can come later.
- Other languages.

## Anti-goals: it fails if

- It says "this is fake" or "this is real" as a certainty. The outcomes are "signs of manipulation", "no signs found", "can't tell" and "can't check".
- "No signs found" reads as proof of authenticity.
- It keeps uploaded videos after the answer is sent.
- It shows a number that `numbers.json` cannot trace, or an accuracy figure without saying which data it was measured on.

## Approaches considered

How the detector answers decided the most, so it was settled first.

| Approach | Trade-off | Decision |
|---|---|---|
| **1. Three outcomes, with a "can't check" for unusable clips.** Each check's score is placed against real clips; one combined threshold flags about 5 in 100 real clips. | A clear answer that also says when it does not know. Needs a calibration set and careful wording. | **Chosen by Vaibhav, 2026-11-17** |
| 2. Scores only, no verdict | Most honest, but leaves the visitor to read raw numbers | Rejected: wrong for the audience |
| 3. One confident verdict from the fusion (original B's mockup) | Simplest, but overclaims. A12 found that a fixed fusion of the three paths does not transfer between corpora | Rejected |

Already decided on 2026-11-16 (project log): evolve the existing FastAPI web app rather than start a new stack, and host on a free Hugging Face Docker Space (CPU Basic: 2 vCPU, 16 GB RAM, sleeps after 48 h idle), with this PC and a temporary link as the fallback.

## Pages

Routes are server-rendered from Jinja2 templates with one shared layout (nav, footer, fonts), replacing today's single page with tabs. Numbers are filled in on the server from `numbers.json`. Each page works without JavaScript except the detector and the charts. *(Builder's call, made in this draft: templates instead of one static page, so the nav and footer live in one place.)*

| Route | Page | Content |
|---|---|---|
| `/` | Home | Original B's hero ("Detect the fake. Not the file format."), with a compact drop zone that hands the file to `/check`. Then v2's "Three checks, side by side". Then the bento of findings, from `numbers.json`. Then "Read the limits". |
| `/check` | Detector | The full tool: drop zone, sample clips, progress, the outcome card, the trace, "How sure?", and the limits beside the result. |
| `/how-it-works` | How it works | Cleaning the file (neutralization), the three checks, the example trace with v2's real-against-dubbed toggle, how the outcome is decided, and the dumbbell chart of file-cue ceilings. |
| `/research` | Research | P-001 in plain words: the hypotheses and what happened to each, the main tables, the limits, prior work, and links to the paper and report PDFs and the GitHub repository (public, verified 2026-11-17). Carries over today's Results, Against prior work, and Failures and limits tabs. |
| `/about` | About us | The team, the guide, the college, and why the project exists. |
| `/contact` | Contact | A short form (name, email, message) sent to Vaibhav through Resend. |
| `/admin` | Admin (hidden) | Not linked. Token-guarded settings, as today. |

## The detector

### Clip rules

One person speaking to the camera; MP4, MOV or WebM; 3 to 60 seconds; at most 90 MB. *(Builder's call: 60 s keeps decode time bounded; only the first 3 s from the first sound are scored, as in P-001.)* The rules sit under the drop zone and are checked on the server, not only in the browser.

### Pipeline per upload

1. **Receive and validate:** size, container and stream checks with `ffprobe`; reject anything that is not one video and at most one audio stream.
2. **Neutralize:** the P-001 neutralizer v4, unchanged (one codec, 3.00 s from the first sound, fixed loudness, dither).
3. **Preprocess:** the P-001 preprocessing, refusing to score if the preprocessing hash differs from `5f2954fb...` (the hash all P-001 models were trained on).
4. **Quality gates:** the preprocessing quality report decides what can be checked. No stable face track: no face or lip-sync check. No audio: no voice or lip-sync check. If neither face nor voice can be checked, the outcome is "can't check", with the reason.
5. **Score:** the face check (P-001 visual path), the voice check (P-001 audio path) and the lip-sync check (AV-HuBERT Base with the P-001 sync heads). Each check averages its three seeds. *(Builder's call: averaging the seeds removes the seed lottery the research warns about.)*
6. **Decide the outcome** (below).
7. **Delete** the upload and every derived file before the response is sent, or on any error.

The job runs in the background with progress stages the page shows: received, cleaning, finding the face, scoring, done. One job runs at a time; up to three wait; beyond that the page says the detector is busy and to try again in a minute.

### Deciding the outcome

**Calibration set:** real clips only (clip label real: neither stream fake), from the four P-001 evaluation sets: the FakeAVCeleb identity-strict test, DFDC, LAV-DF test (window label) and HiFi-AVDF, each corpus weighted equally. Their check scores already exist in `results/` (`p001-branches`, `p001-sync-targets`, `p001-sync`), so calibration needs no new model runs.

1. For each check, the clip's score is turned into a **percentile among the calibration real clips** ("higher than 97% of real clips").
2. The clip's **combined score** is the highest of its available check percentiles.
3. The **threshold** is the 95th percentile of the combined score over the calibration real clips, so about 5 in 100 real clips from those corpora are flagged.
4. Outcomes:
   - combined score at or above the threshold: **signs of manipulation**, naming the check or checks above their own 95th percentile;
   - between the 80th percentile and the threshold: **can't tell**;
   - below the 80th percentile: **no signs found**;
   - quality gates failed: **can't check**, with the reason.

The calibration is computed once by a script, stored as a small versioned JSON file with its source files' hashes, and loaded by the server. *(Builder's call: the 80th percentile for "can't tell" is a starting point; the plan measures how many clips land in each outcome on the calibration and target sets before fixing it.)*

**Wording rules:** the "How sure?" text states the false-alarm rate on the calibration corpora and, from the same results, how many fakes the threshold catches there, which is low on corpora the models never trained on. It says that the 5 in 100 is measured on those corpora and is an estimate for anything else. These calibration clips are test data in the paper; the site uses them only to place scores, and must not present them as a new evaluation.

### The outcome card

As in the [design system](../../DESIGN.md): the outcome, which check raised it, three check rows (score, percentile bar, colour by check), the trace, and "How sure?". A check that could not run shows as "not checked" with the reason, not as a score.

### The trace

Speech loudness (amber) and mouth opening (blue) over the 2 s sync window of the uploaded clip, with the lip-sync score beside it. Loudness comes from the neutral audio. Mouth opening is the risk: preprocessing keeps only the mouth corners, not the lip outline. The plan's first spike tries a mouth-opening measure from the cached mouth crops and checks on real FakeAVCeleb clips that it tracks the loudness. If it does not, the card shows the loudness trace with the per-window lip-sync misfit instead, and How it works keeps the illustrative real-against-dubbed example.

## What carries over from the current web app

| Today | On the new site |
|---|---|
| Check tab, drop zone, preview | `/check`, restyled |
| Sample clips (today: FakeAVCeleb test clips) | `/check`, re-encoded to H.264 so browsers play them. **Licence rule:** a clip is served publicly only if its corpus's terms allow redistribution, checked per corpus before launch. FakeAVCeleb is obtained by request form, so its clips are assumed not to qualify. If no corpus qualifies, the samples are clips the team records of themselves, with written consent for a fake version |
| Model picker (ensemble, single model, multimodal) | Replaced by the three P-001 checks; the September models leave the detector |
| "Withhold on unfamiliar footage" toggle | Becomes the "can't tell" and "can't check" outcomes |
| How it works tab | `/how-it-works`, rewritten for P-001 |
| Results, Against prior work, Failures and limits tabs | `/research`, updated to P-001 |
| Report tab and `/report.pdf` | Links to the compiled P-001 paper and report PDFs |
| Admin settings | Kept: enable or disable the detector, upload limit, show research pages. Dropped: default model, default threshold and threshold lock, because the threshold now comes from calibration |
| Upload body-size limit (ASGI middleware) | Kept, set to the new 90 MB limit |

Vaibhav decides later whether anything else is dropped (PENDING, section 5).

## Constraints

- **Free tiers only.** Hugging Face CPU Basic (verify on Vaibhav's account that free Docker Spaces still run) and Resend's free plan (verify its current daily and monthly limits; when they run out, the form says to email directly).
- **Model files stay out of git.** AV-HuBERT Base (`models/avhubert/base_vox_iter5.pt`, 1.24 GB), wav2vec 2.0 Base, and the nine P-001 checkpoints are fetched at image build time from a private Hugging Face model repository. Their licences must be confirmed for public serving (AV-HuBERT's licence is **unknown** here).
- **Speed on two CPUs is unknown.** The first milestone measures one full check on CPU. If it takes much over two minutes, the options are fewer seeds per check, or keeping the lip-sync check and dropping the others' seeds to one, decided with numbers.
- **Security.** The site decodes untrusted media with ffmpeg and loads model files, so `docs/SECURITY.md` (PENDING, 5b) is written before launch: upload limits and stream checks, ffmpeg run with time and memory limits, rate limits per address, the admin token, deletion of uploads, no secrets in the image, and the contact form protected against spam (a hidden field and a rate limit).
- **Secrets** (`RESEND_API_KEY`, the contact address, `DDF_ADMIN_TOKEN`) come from the host's settings, never from the repository.

## Left to the builder

- Template and file layout inside `src/deepfake_detection/webapp/`.
- The progress mechanism (polling an endpoint is enough).
- Chart drawing (small inline SVG from JSON, as v2 does).
- The exact copy, within the wording rules above.
- Which findings fill the Home bento. Each cell names its `numbers.json` key in the template, and original B's stale cells ("11 amendments") are replaced.
- Test fixtures for the outcome logic and the quality gates.

## Milestones (for the plan)

1. **Spike:** one upload through the full P-001 pipeline on CPU, timed; the mouth-opening measure tested.
2. Calibration script and file, with tests of the outcome logic.
3. Detector backend: job queue, gates, outcomes, deletion, limits.
4. Templates and the design system; Home and `/check`.
5. How it works, Research, About, Contact (Resend).
6. Admin carry-over; `docs/SECURITY.md`.
7. Docker image and the Hugging Face Space; check the free tier and the speed there.
8. Retire the old in-browser demo (`hosting/browser/`, `docs/hosting.md`, the README link).

The build starts after the submission on 20 November.
