# Deliverables v2: shared specification (2026-10-28)

New, shorter versions of the four deliverables. The originals (paper/, report/, slides/, src/dfd/platform/dashboard/) are reference
material and are NOT modified. Nothing here is committed, pushed or published. Project rules in CLAUDE.md apply (no em-dashes, no
unmeasured result, no invented citation, dates real).

## Ownership (one workstream per folder; no workstream edits another's folder or any file outside deliverables_v2/)

| Folder | Workstream |
|---|---|
| deliverables_v2/CONTENT_MAP.md | evidence and content map (read by all others, written by one) |
| deliverables_v2/research_paper/ | new paper (LaTeX) |
| deliverables_v2/project_report/ | new report (Word builder) |
| deliverables_v2/presentation/ | new deck (python-pptx builder) |
| deliverables_v2/website/ | redesigned dashboard (new static front end + thin server wrapper) |

Protected, never edited: src/ (all), ledger/, data/, checkpoints/, scripts/, tests/, docs/ (except by the main agent), paper/, report/, slides/,
assets/, Document Template/. Generated outputs are written only inside the workstream's own folder.

## Identity

- Title (all four): "Multimodal Deepfake Detection for Images, Video and Audio". No subtitle. The old subtitle
  "A Hypothesis-Driven Study Under a Group-Disjoint Protocol" must not appear in any v2 output (user instruction 2026-10-28).
- Author order (user instruction 2026-10-28): Shivansh Gupta, Vaibhav Shiroorkar, then the third student. Spelling of the third name:
  see the AUTHORS line below (set by the main agent; use this exact value everywhere, including document metadata).
- AUTHORS: Shivansh Gupta (16014023052), Vaibhav Shiroorkar (16014024805), Adithya Jithesh (16014024803)
  (spelling "Adithya" confirmed by the user on 2026-10-28; the subtitle is dropped from all four v2 deliverables, user choice 2026-10-28)
- All other identity fields (university, school, department, guide, HoD, footer, year) exactly as docs/IDENTITY.md.
  No "first author" labels, no new affiliations, e-mails or contribution statements.

## Colours

- Paper (unchanged identity): paper/college-style.tex, collegemaroon #B7202F (border, section headings, title), collegebrown #312016
  (subsection italics). The v2 paper reuses college-style.tex and the IEEEtran class as they are.
- Report: every element of the original report that uses a brown (for example the footer rule colour 622423 in report/build_report.py,
  and any other brown found in the builder or the template) becomes #B7202F in v2. Body text stays black.
- Deck: keep the original deck's template, fonts and colours (slides/slide_common.py).

## Science (non-negotiable)

- Every number comes from deliverables_v2/CONTENT_MAP.md, which cites its source (docs/EXPERIMENTS.md E-xxx, docs/NUMBERS.json,
  docs/PERFORMANCE_BENCHMARK.md, result files). No number may be invented, rounded into a new claim, or moved to a different dataset or
  split. Exploratory vs sealed vs post hoc must stay labelled.
- Score semantics: detector scores are temperature-scaled on each detector's validation split (a probability only for content like that
  split); the speech score has an exploratory calibration; experimental heads give uncalibrated ranking scores; the platform's
  `calibration.is_probability` is always false. Never call a score "the probability that the file is fake".
- Face-video flag label: "Possible manipulation" (high uncertainty). Never "Likely fake" or "Likely manipulated" as a current finding
  (the frozen v1 page's "Likely fake" may be mentioned only as history).
- No report-level verdict, no overall or fused score. Experimental heads never give a finding. APPROVED_HEADS is empty.
- Gemini: no Gemini data, model or experiment exists. Only docs/GEMINI_FEASIBILITY.md (assessment and proposal) and the nearest measured
  generator, Google Imagen 3 inside E-028/E-029. It is an unresolved research direction, not a capability.
- Comparisons with published numbers are contextual unless docs/PERFORMANCE_BENCHMARK.md classes the row as directly comparable.
- Citations only from paper/refs.bib entries with READ (or PARTIAL, for the parts read) status in docs/RESEARCH.md.
- No statements about which AI tools helped write anything.
