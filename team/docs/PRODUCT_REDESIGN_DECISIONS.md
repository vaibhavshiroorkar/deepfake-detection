# Product redesign: decisions to approve

Companion to `docs/PRODUCT_REDESIGN_AND_MODEL_ROADMAP.md` (2026-10-26). Each decision has a recommended default; approving the defaults lets Phase 1, Phase 2 and the first two experiments start at once. Nothing here changes a detector decision, threshold, checkpoint, frozen result or `APPROVED_HEADS`, except where a decision says so explicitly.

| ID | Decision | Recommended default | Consequence if approved | Consequence if rejected |
|---|---|---|---|---|
| D1 | Visual direction and palette of roadmap section 4 (slate base, teal accent `#0E6E6E` / `#4FC1B9`, no green for "nothing found") | approve | CSS rewrite against the token table; contrast pairs tested | keep current styling; Phase 1 limited to layout fixes |
| D2 | Keep plain JavaScript and Flask, split into three files plus `/api/vocabulary` | approve | no build step; existing browser tests keep working with migrated hooks | a framework migration (not justified by any finding) |
| D3 | Report schema 2.1: optional, server-generated `explanation` per module and `summary` per report, validated; dashboard accepts 2 and 2.1 | approve | explanations testable against fields; report, paper and slides describe the addition | explanations written in the browser, harder to test, drift risk |
| D4 | Expose per-frame face scores and face crops from the frozen adapter (`src/dfd/infer/real_adapter.py`), additive, with a parity test proving clip scores unchanged | defer until after 30 October | frame-level evidence in the UI; touches protected code | no frame-level face evidence; explanation says "mean over N frames" only |
| D5 | Return the lip-sync offset curve (31 values) and show it in technical details | approve | an honest chart from computed data; `lip_sync.py` and its tests change, decision band unchanged | lip-sync explanation text only |
| D6 | Return the 8 canonical frames the generated-video model saw (small thumbnails) and show them in technical details | approve | shows exactly what the model used, including the crop; report JSON grows by about 40 KB | no visual for the generated-video module |
| D7 | Human-readable module names in the UI; internal experiment IDs, versions and hashes only in collapsed technical details | approve | evaluator-friendly screens; IDs stay in the JSON | current technical cards |
| D8 | Images with a face: also run the generated-image model (whole image) and show it as a separate, EXPERIMENTAL output | defer until X5 measures T-I on modern generators | closes the routing gap UX-7 honestly; a new output whose scope is untested on portraits | generated portraits keep being checked only by the face-swap model; documented limitation |
| D9 | Speech on non-speech audio: add a warning (heuristic, from the lip-sync speech check) without changing the speech finding | approve warning-only | users see the caveat on the speech row; finding and score unchanged; one new warning code and test | caveat appears only as fixed explanation text (D-075 as is) |
| D10 | (REJECTED by the user, 2026-10-27: out of scope, not to be built, tested or planned) Build the optional visible-AI-marking module (template matching first; OCR only if an OCR engine install is approved) in a new report section `declared_markings` | approve as a prototype after 30 October; labelled set built from team-generated media | visible labels reported with the "not proof either way" caveat, experimental until its evaluation passes | Gemini-style marks stay unreported; UI states that visible marks are not checked |
| D11 | Optional LLM explanation layer | reject for now | none | n/a |
| D12 | Amend D-071 (4): allow detector training on DF26's open-generator subset under its leave-one-generator-out protocol (closed subset stays evaluation-only, one preregistered look per experiment) | approve after X2 is logged | enables X3, X4, X6, the most promising improvement route | generated-video research limited to DeepAction-only diagnostics |
| D13 | Download a preregistered, test-only subset of Community Forensics Eval (CC BY-NC-SA 4.0, by shard) for X5 | approve a small subset (record terms and hashes in DATA_LOG first) | first measurement of the image detector on modern generators | image detector scope stays "2022-era generators only" |
| D14 | Live mode opens on the upload panel; mock scenarios move to a "Demo scenarios" menu with full MOCK treatment and MOCK-named exports | approve | no invented data on the first screen | mock report remains the first thing visitors see |
| D15 | Scope for 30 October: Phases 1 and 2 (P0 and P1 UI items, deterministic explanations), X1 and X2, manual matrix, document updates; Phase 3, X3 to X6 after the evaluation | approve | realistic, fully tested deliverable; research claims unchanged unless X2 adds a logged diagnostic | compressed research with a high risk of unlogged or unfinished experiments |

Approving D1, D2, D3, D5, D6, D7, D9, D14 and D15 is enough to start implementation tomorrow. D12 and D13 gate the model research. D4, D8 and D10 are best decided after the evaluation.

## Status on 2026-10-27

The user's instruction of 2026-10-27 asked for the presentation and explanation work (dashboard redesign, deterministic explanations,
evidence visualisations from data already computed, defect fixes) and for the X2 protocol to be written before it runs, without changes to
detectors, thresholds, checkpoints, dataset roles or the experimental status. Read against that instruction:

| ID | State |
|---|---|
| D1, D3, D5, D6, D7, D14 | implemented (D-076, autonomous within the instruction, needs user review) |
| D2 | implemented except the three-file split of the script (one file kept) |
| D9 | implemented at the explanation layer only: the speech result and score are unchanged; the explanation cites the lip-sync speech screen |
| D11 | not built (the recommended default) |
| D15 | followed: Phases 1 and 2, X1 and X2; nothing beyond |
| D10 | rejected by the user |
| D4 | open, recommended default unchanged (after 30 October; touches protected code) |
| D8 | open, waits for D13 and X5 |
| D12 | open; X2 is now logged (E-026), which was its stated precondition |
| D13 | open |
