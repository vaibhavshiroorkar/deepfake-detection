# PROPOSED DELIVERABLE INSERTIONS (DRAFT FOR REVIEW, NOT APPLIED)

Written 2026-10-31. This is a draft list of possible additions to the paper, report and slides based on E-033, E-034 and the verified literature comparison. **Nothing here has been applied.** submission/final/ and deliverables_v2/ are unchanged. Applying any item needs the team's decision, then a rebuild with the deliverables' own build scripts, the number checks, and a regenerated submission manifest (D-093).

## What is eligible for inclusion

These rest on logged experiments and verified sources.

| Item | Evidence | Eligible as | Wording guard |
|---|---|---|---|
| E-033 and E-034 headline AUROCs with intervals and counts | docs/EXPERIMENTS.md E-033, E-034 (results and independent checks) | exploratory post-freeze evaluation | "a replication on previously unread samples of a seen distribution"; never "independent test" or "confirmatory" |
| Diagnostic groups (T-V produced-only 0.486; T-A in-window 0.733; T-I PNG-only 0.569) | same | exploratory, descriptive | association language only ("falls when", "is consistent with"), not cause |
| Sample-size table (nested groups) | docs/POST_FREEZE_BENCHMARK_COMPARISON.md section 2 | teaching point about uncertainty | say the groups are nested views of one population |
| Tier B placements: In-the-Wild EER against Muller et al. and Pascu et al.; ASVspoof 2019 LA EER against the LFCC-GMM baseline and AASIST; DFD AUROC against SBI on the same videos | docs/LITERATURE_BENCHMARK_EVIDENCE.md rows 9, 33 to 37; docs/RESEARCH.md | related-work comparison | "on the same dataset under a different protocol"; never "better than" |
| Tier C context: LAV-DF detectors trained on LAV-DF; AVFF on FakeAVCeleb | rows 38, 39 | context only | state the training-data difference in the same sentence |

## Not eligible, or needs care

- The AP comparison with Community Forensics Table 2 (mAP 0.592). The same scores give mean AP 0.34 or 0.77 depending on the class ratio of the cells, so it is not interpretable. Use AUROC, or omit it.
- Any global ranking statement, or any statement that a head is approved.
- Derived F1 values for the freeze sets (computed post hoc from stored seed-mean rates). If used, label them post hoc.

## Suggested placement (minimal)

1. **Paper:** one short paragraph and one table in the results or limitations section. The table holds the E-033 and E-034 G4 rows plus the diagnostic groups, cited as exploratory. Add one sentence of related-work comparison with Muller et al. and Pascu et al. (Tier B).
2. **Report:** a subsection "Post-freeze evaluation on unread data", built from docs/POST_FREEZE_EVALUATION_PROTOCOL.md sections 1 to 4 and the sample-size table, with the comparison table restricted to rows 1, 2, 7, 8, 9 and the tier legend.
3. **Slides:** one backup slide with the sample-size table and the three diagnostic results, and one backup slide with the tiered comparison (speech rows only, to stay readable).

## Decision needed from the team

Include (all, some, none). If "some", name the items. Until then the submitted set stays as built on 2026-10-30.
