# LIVE STATUS: modern-generator, literature and product work (from 2026-10-27 00:45 IST)

Short running record so that parallel work does not lose track. Each owner updates only its own rows. Final results go to the
living documents (EXPERIMENTS, DATA_LOG, STATUS); this file is a coordination aid, not a result record.

## Ownership (D-081)

| Track | Owner | May create or edit | Must not touch |
|---|---|---|---|
| W1-V video: DF26 open-generator training, E-027 | agent `video` | scripts/modern_video/, data/_results/modern/e027/, checkpoints/modern-E027/, docs/EXPERIMENTS.md (E-027 block only), docs/MODERN_GENERATOR_RESEARCH.md (section "Video" only), rows W1-V below | everything in "Protected" |
| W1-I image: Community Forensics, E-028 and E-029 | agent `image` | scripts/modern_image/, data/cf_eval/, data/cf_small/, data/_results/modern/e028/ and e029/, checkpoints/modern-E029/, docs/EXPERIMENTS.md (E-028, E-029 blocks only), docs/DATA_LOG.md (I6, I7 rows and new rows), docs/MODERN_GENERATOR_RESEARCH.md (section "Image" only), rows W1-I below | everything in "Protected" |
| W2 literature and benchmarking | agent `literature` | docs/LITERATURE_SURVEY.md, docs/PERFORMANCE_BENCHMARK.md, docs/RESEARCH.md (append rows), docs/NUMBERS_LIT.json only if its build script requires it, rows W2 below | everything in "Protected"; no code |
| Product, integration, deliverables | main agent | src/dfd/platform/**, tests/**, scripts/platform/**, docs/DECISIONS.md, docs/STATUS.md, report/, paper/, slides/, all other docs | as below |

Protected for everyone (read only): src/dfd/infer/**, src/dfd/web/**, checkpoints/I-*/**, checkpoints/genvideo-E020/**, ledger/**,
data/_final/**, every existing folder under data/_results/ (new subfolders only), sealed splits (FakeAVCeleb sealed half and every split the
eval guard refuses without --final), APPROVED_HEADS, every detector threshold, the lip-sync band (-6, +2). No git commit, push, merge, reset,
clean or checkout. Visible-logo or watermark detection is excluded (D-077).

## Rows

| Row | Track | State | Note |
|---|---|---|---|
| W1-V | video | E-027 COMPLETED 2026-10-27 14:51 | prereg hash re-checked equal before the run; chosen P8-IMG-13 (CLIP ViT-L/14, 8-frame mean, LR), open LOFO 0.7453 [0.7174, 0.7710], H-X3 SUPPORTED; H-T no temporal gain; confound re-check CONFOUND PERSISTS (0.8023 of 440 other-source reals flagged); closed look (once) 0.7548 [0.7189, 0.7885] vs 200 reals, H-X3c SUPPORTED; RESULT in docs/EXPERIMENTS.md, section Video in docs/MODERN_GENERATOR_RESEARCH.md; recommendation: candidate for experimental head under roadmap 8.5 (needs user review); nothing installed |
| W1-I | image | E-028 and E-029 COMPLETED 2026-10-27 (E-029 frozen 21:08:04, one Eval look 21:08:12) | Both verified independently by the main agent (docs/EXPERIMENTS.md). E-028: frozen T-I macro AUC 0.7039, INADEQUATE. E-029: H-29 SUPPORTED (0.8839, +0.1800 [+0.1643, +0.1959]) but 0.5213 of real Eval images flagged at the validation threshold, NOT ELIGIBLE (D-086). Nothing running |
| W2 | literature | DONE 2026-10-27 15:20 | docs/PERFORMANCE_BENCHMARK.md (benchmark table, per-module analysis, gap analysis, viva findings, open issues, verified bibliography) and docs/LITERATURE_SURVEY.md (8 tasks, paper-to-module map, 46 papers cited) written; RESEARCH.md rows R-083 to R-086 appended (all PARTIAL) with a re-check note; rows `[pending E-027]` (E-027 RESULT now in EXPERIMENTS.md, not copied) and `[pending E-028/E-029]` left for the main agent; open issue: T-A EER on the sealed sets is not logged, so audio rows stay partially comparable |
| P | product | 2026-10-27 22:00 DONE, waiting for the user's review | D-083 verified and its text updated from E-028; E-027, E-028, E-030 verified; E-029 integrated into benchmark, literature survey, paper, report, slides, viva, testing record; final tests in docs/test_evidence/2026-10-27_final. Nothing committed |
