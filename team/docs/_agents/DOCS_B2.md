# DOCS_B2: fixes of the verifier defects for the report and the slides (2026-10-25, fork B2)

Source of the defects: docs/_agents/VERIFY3.md. No git commands were run. Files changed: report/ch3.py, ch4.py, ch4_results.py, ch4_iter.py, report/label_shots.py (new), report/_labelled/ (new, labelled copies of four screenshots), slides/build_slides.py, and the built report/DeepFake_Detection_Technical_Report.docx and preview PDF, slides/DeepFake_Detection_Presentation.pptx and preview PDF.

| # | Defect | Fix | Check |
|---|---|---|---|
| 1 | Report 3.2 said "89 percent fake" | now "90.3 percent fake, 9.7 percent real" (report/ch3.py). No other "89 percent" or "10.8" in report or slide sources (the 89 in ch4.py is a test count) | pdftotext of the preview: 0 hits for "89 percent", the new sentence present |
| 2 | pptx metadata of the reference deck | build_slides.py sets created and modified to the build time (UTC), revision 1, and replaces docProps/app.xml by a minimal one (slide count 37, TotalTime 0, no reference titles) | unzipped: core.xml revision 1, dates 2026-10-25; app.xml has no "Slides 9", "TotalTime 510" or reference titles. The docx metadata (Word-written) was already clean: created 2026-10-25, title and authors set |
| 3 | Chapter 4 box clipped | text shortened to four lines (the box is fixed at the template height of 71.4 pt) | rendered page 35 of the preview: the box shows all four lines |
| 4 | Figs 9 and 10 without (a)/(b) | report/label_shots.py writes labelled copies into report/_labelled/ (originals in assets/screens untouched); ch4.py and ch4_results.py embed them | rendered page 66: labels (a) and (b) inside the four images |
| 5a | T-I sentence missing in report and slides | added to report 4.5.2 (ch4_results.py) and as a bullet on slide 9 | rendered |
| 5b | "sign-flipped shortcut" missing | added in report 4.5.4 and on slide 9 | rendered |
| 5c | Different "s per 1,000" on slides 8, 26, 27 | slide 8 caption says median of 3 repeats of the cost record; the notes of the detail slides say a single extraction run, values differ slightly | text only, numbers unchanged |
| 5d | Body text touching the footer rule | not changed: the layout test compares against the Word template and passes, and the touching is template behaviour (not checked in Word beyond the rendered pages) | open, low |
| 5e | Tables 12 to 15 pt on slides | not changed beyond the automatic fit (tables are sized to the largest size that fits); the 16 pt rule of the checker is for body text only | open, low |
| 5f | Stale page counts | no page or slide count appears in the visible text of the report or slides; the counts in STATUS and D-053 are for the parent to update (report 83 PDF pages now, deck 37 slides) | grep |
| 6 | New figures of fork A2 | embedded at the widths of assets/figures/READY_A2.txt (report 15.5 cm for pipeline, ladder, rppg, 15.2 cm staircase, 14.4 cm LAV-DF cells; slides 11 and 15 now fit by height) | see below |

Slides 11 (ladder results) and 15 (LAV-DF cells) place the figures by available height, not by 8 in width, because the new figures are taller; the ladder chart on slide 11 is therefore about 10 pt at slide size (below the 12 pt aimed at). Slide 9 holds one more bullet and the table text is unchanged.

Checks after the final build: `PYTHONIOENCODING=utf-8 python scripts/check_docs.py --numbers` 71 pass, 0 fail; `python report/layout_check.py` 26 of 26 rows within 2 mm; `python slides/check_slides.py` 614 pass, 0 fail. The slide preview PDF was re-exported through PowerPoint (the earlier one was stale). I looked at slides 9, 11 and 15 and at report pages 35, 52 and 66 after the rebuild; I did not re-render every page and slide.

Open: the report has 83 PDF pages now (was 80) because the figures are taller; the verifier did not look at the new layout page by page.
