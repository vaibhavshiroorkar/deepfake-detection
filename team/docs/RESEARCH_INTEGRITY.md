# RESEARCH INTEGRITY: what the repository can and cannot show

Written 2026-10-26 after the merge-readiness audit of commit 0704b3a. Purpose: state exactly which claims about the order of work are supported by evidence a third party can check, which rest on the project's own text, and which are unverified. Nothing here alleges that any step happened in a different order from the one claimed; the point is that git alone cannot show it.

## 1. Preregistration ordering of E-019 to E-025

Observed facts (checked with git on 2026-10-26):

- The entries E-019 (v2 and v3 blocks), E-020, E-021, E-022, E-023, E-024 and E-025, and the decisions D-071, D-072 and D-073, first appear in `docs/EXPERIMENTS.md` and `docs/DECISIONS.md` in one commit, 0704b3a (author date 2026-10-26 05:29:41 +05:30). The parent commit cf84b99 contains no line matching E-019 to E-025 and no D-071 to D-073.
- The same commit adds the scripts that ran the experiments and the ledger line for E-020. Result files are git-ignored (`data/_results/`), so they are not in any commit.
- Git therefore cannot show that a hypothesis, success criterion or amendment was written before the corresponding run.

In-document claims (the project's own text; not independently checkable from git):

- Each entry states that it was registered before the run (for example E-020 "before any DeepAction or DF26 feature was extracted", E-025 "before any E-025 clip was run", with the amendment "appended before any E-025 clip was scored").
- E-020 states that the three heads were hashed before DF26 was evaluated. E-025's script writes the band file before it touches validation or Wav2Lip clips.

Consistent but weak evidence (file timestamps, which can be changed and are not a trusted record):

- `data/_results/genvideo/e020_dev_results.json` (03:58) holds the three head hashes and was written before `e020_df26_results.json` (04:00); both head hashes in the DF26 file equal the values in the development file and equal the SHA-256 of the files on disk.
- `data/_results/platform/e025_band.json` (05:19) precedes `e025_validation.json` (05:25); the script enforces that order.

Unverified: the time at which the preregistration text, the E-025 amendment and the decision rules were written relative to the first run of each experiment.

This limitation is a gap in the record. It is not evidence that the order was wrong, and it is not a finding of misconduct. History is not rewritten or back-dated to repair it (CLAUDE.md rule 5).

## 2. Rule for all future experiments

Before a run starts:

1. Commit the preregistration (hypothesis, success and failure criteria, data and exclusions, analysis, decision rule) as its own commit, with the experiment id in the message, for example `docs(prereg): E-026 ...`. Do not put the run's code changes or results in that commit.
2. Optionally tag it `prereg-E-026` and, when a remote is available, push it, so that the push time is recorded by a third party.
3. Record in the run's output file and in the EXPERIMENTS.md result block: the source commit (`git rev-parse HEAD`) and whether tracked files were modified, the SHA-256 of the protocol/preregistration text, the SHA-256 of the configuration and of the script, the run id, the command line, the output location and the SHA-256 of each output file.
4. Append results in a later commit. Amendments are new, dated entries committed before the affected run; they never replace the original text.
5. The test suite header prints the git HEAD and dirty state (tests/conftest.py); keep a log of any run whose result is cited, with the header.

## 3. Ledger

- `ledger/final_ledger.jsonl` is append-only. Between cf84b99 and 0704b3a exactly one line was added (E-020, dataset df26, split label `test_x`) and no earlier line changed (git diff: +1, -0).
- Field note for that line: `score_file_sha256` (65a607c471641edbbfb00389448d0e419827a58fb4143a849c2880c3ab2f6183) is the SHA-256 of `data/_results/genvideo/features_df26.npz`, the DF26 feature table, not of a file of scores; `run_ids` are the SHA-256 of the three head files, not run identifiers. The scores are in `e020_df26_results.json` (SHA-256 b2d88f6e43cf1749ab97af642dc25b081575a5443045f6487484633ece4712eb on 2026-10-26). The historical line is not edited. Future entries should name the field they hash (a separate key for the feature file) and use real run ids.

## 4. Protected paths and frozen artifacts

What is established:

- Tracked files: `git diff cf84b99 0704b3a` shows no change under `paper/`, `report/`, `slides/`, `src/dfd/infer`, `src/dfd/web` or the 33 tracked files of `data/_final`; the only change to `ledger/` and the only non-additive change to `docs/NUMBERS.json` (a regenerated header line) are described above.
- Frozen E-020 heads: the SHA-256 of `head_C1/C2/C3.npz` on disk equals the hashes recorded in `e020_dev_results.json` and in the ledger line.
- Sealed E-015: `data/_results/e015/predictions_test_x.csv` has SHA-256 ac58b9f8…, equal to the ledger entry.
- `checkpoints/I-5-freeze-candidate/`: the files hash to the values in that checkpoint's own `metadata.json`.

What is NOT established: `checkpoints/I-*`, `data/_results/freeze`, `data/_final` outputs that are untracked, and sealed result files other than the E-015 predictions are git-ignored, and no independent, trusted hash manifest recorded before the sprint exists for all of them. For these only file modification times (nothing newer than the branch point) and, for some, self-recorded hashes were checked. A modification time does not prove a file is unchanged. A trusted baseline would need a hash manifest committed (or otherwise recorded by a third party) before the artifacts are used; creating one now would only certify the present state and is not done.

## 5. Statement classes in this repository

| Class | Examples |
|---|---|
| Observed fact (recomputed from stored outputs or git) | E-025 3/230, 27/230, 0/160; E-020 AUCs; the single-commit history above |
| In-document claim (not independently checkable) | "registered before the first run" statements; autonomous-decision tags |
| Interpretation | "artefact of the narrow rule"; "probably a source confound" (E-020 C2 on DeepAction) |
| Unverified | ordering of text vs run; integrity of untracked frozen artifacts beyond the hashes listed |

## 6. Known scientific limitations (not integrity gaps)

- E-025: the LAV-DF half of both pools was split by file, not by speaker; speaker overlap cannot be excluded (see the post-audit clarification in docs/EXPERIMENTS.md). Both sources derive from VoxCeleb2.
- E-020: the installed head did not generalise to DF26 (AUC 0.4957 [0.4755, 0.5168]); DeepAction results are probably confounded by source.
- E-019: the v3 header audit passed because the header fields are constant by construction; pixel statistics still reach about 0.77 to 0.79 AUC. Not shortcut-free.
- E-023: within FakeAVCeleb, codec and method are fully confounded.
