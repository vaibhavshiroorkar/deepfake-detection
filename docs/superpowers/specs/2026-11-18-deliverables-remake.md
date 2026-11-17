# Deliverables remake: paper, report and PPT (design)

Decided with Vaibhav on 11-18. Exhibition on 11-20; the live demo comes first, then the PPT, then the report.

## The one idea

**Most of what deepfake benchmarks reward does not survive an upload.** Platforms re-encode every video, which wipes out the file clues (codec, length, silence) that benchmark labels leak through. Our cleaning step is a simulated upload. We measure what still works after it: a lip-sync check that learned from real videos only works at home and weakens on new data, and no part of the system recovers the near-perfect benchmark scores.

Everything in the three deliverables serves this idea. The file audit is the motivation and the measuring stick, not the headline.

## What the research says to do (sources in the plan)

- Paper: one clear idea; the contribution list written first and each claim pointing forward to its evidence; open with a concrete example; write for the reader, not the story of discovery; related work near the end (Peyton Jones, "How to write a great research paper").
- Slides: a sentence headline as the slide's point, one visual as its evidence (assertion-evidence, Alley); no printed text that repeats the speaker; cut what does not serve the point (Mayer: redundancy, coherence).
- Figures: one visual style across the paper; colours in method diagrams match the result charts; light text inside figures; flat style; built at column width (scientific-figure guides).
- Prose that reads as written by a person: concrete examples, varied sentence length, no stock phrases, a clear point of view, precision kept.

## Decisions

- **Paper title:** *What Survives the Upload? Evaluating Audio-Visual Deepfake Detectors After Removing File Shortcuts.* Report and PPT keep the registered project title (Vaibhav confirms it with the guide).
- **Paper order:** Introduction (opens with the codec example; contributions with forward references), Method by stage (file audit, cleaning, face and voice checks, lip-sync check, comparing with real clips), Experimental setup (data, splits, seeds, measures, the four preregistered tests H1 to H4), Results (what files give away; what cleaning leaves; the lip-sync check at home; on new data; false alarms; combining checks; further checks), Related work, Discussion and limitations, Conclusion, Appendix (the dated changes to the plan).
- **Labels:** keep H1 to H4, defined once in the setup. No amendment numbers (A4, A9, A15 ...) in running text; they appear only in the appendix table, beside a plain description. Exploratory checks are named by what they do ("moving the audio", "a fixed bitrate").
- **Report:** the same story and order, at black-book length: Introduction, Literature review, Project history and lessons, Method, Experimental setup, Results, Discussion, limitations and future scope, Conclusion, Appendix (reproducibility; the dated changes). Same rules on labels.
- **PPT:** the college's 13 slides stay; the Introduction asks the upload question; the flowchart slide shows the new stage figure.
- **Figures:** one style for all seven: the same font, the colours of the three checks (face blue, voice orange, lip sync aqua) wherever a check appears, plain-English axis labels ("file-clue score" not "shortcut ceiling (AUC, clip label)").
- **Unchanged:** every number (all still filled from `results/numbers.json`), every claim, every limitation. This is a rewrite of order and words, not of results.

## Not in scope

The poster (not needed at the exhibition); the website; new experiments.
