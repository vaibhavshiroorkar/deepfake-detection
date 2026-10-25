# DOCS_A2: figure legibility and paper fixes after VERIFY3 (fork A2, 2026-10-25)

Scope: verifier defects 2 (figure text under 9 pt) and 3 (paper Fig. 5 still the stand-in), plus the T-I sentence check (item 7). No git commands, no GPU.

## What changed

- scripts/make_figures.py: the schematic figures fig01, fig02, fig03, fig04, fig05, fig06 and the table fig08_shortlist are redrawn on a 6.1 in canvas (the report text width) with all text at 9 pt or more (box text 9.5 pt, notes 9 pt, the small "schematic" tag 9 pt). Layouts were shortened where needed (for example "Temp. scaling", the Task column of the shortlist became group rows, fig03 wraps long cells). The two result figures fig08_ladder_results and fig09_lavdf_cells are redrawn at 6.1 in with 9.5 pt text. fig03 now lists "heads trained on LAV-DF, fusion: visual head at chance within produced videos" as before (content unchanged). New function fig07_web_real (key fig07r) builds assets/figures/fig07_web_real.{png,pdf} from assets/screens/real_image.png, real_video.png and real_audio.png, uncropped, labels (a), (b), (c) in a band above each screenshot. The old fig07_web (stand-in) is kept unchanged for the report.
- Cause of the old small text: the figures were drawn 7.0 in wide with 6.3 to 9 pt text and then placed at \columnwidth (3.5 in, scale 0.5) or at 0.78 to 0.9 of the text width. Now every figure is placed at \textwidth (7.0 in) in the paper, i.e. scale 1.15, so text is 10.3 pt or more; in the report at 6.1 in the scale is 1.0.
- paper/main.tex: all figure* includes use width=\textwidth; Fig. 5 is fig07_web_real (real checkpoint, no MOCK banner); the text on the web page (software section) and the caption say the screenshots are of the real working-version checkpoint on training items and that the text inside is not meant to be read at that size; the list-of-figures caption is short-titled "Local web page with the real checkpoint". Paper figure PDFs copied to paper/figures. The paper is now 24 pages (23 before; the larger figures).
- T-I sentence (item 7): already present in the paper (results section: both CLIP models reach 1.0000 on validation, the leader 0.9935 is separated from every other candidate by the paired bootstrap on two generator units only). Nothing added.
- assets/figures/READY_A2.txt lists each figure with the embed width for fork B2.

## Measured after the change

Native widths (inches) of the PNGs: 6.12, 6.12, 6.12, 6.07, 6.13, 6.12 (fig01 to fig06), 6.12 (fig08_shortlist), 6.34 (fig08_ladder_results), 5.25 (fig09), 6.12 (fig07_web_real). Smallest text in every figure is 9 pt at native size; in the paper at 7.0 in it is 10.3 pt or more. Pages 4, 6, 11 to 13 and 19 of the paper PDF were rendered and viewed (Fig. 1, Fig. 2, Fig. 5, Fig. 6 and the tables near them).

Checks: latexmk clean (0 overfull, 0 undefined). `check_docs.py`: paper rows all PASS including number provenance (the parent had rebuilt NUMBERS.json), 71 pass, 0 fail in the whole run at the time.

## Limits and open items

- The screenshots in Fig. 5 are unreadable at page size (the page text is 12 to 16 px at 1,100 px width, about 2 in per panel). The labels are readable; the caption says so. Making the page text readable would need cropping, which D-053 item 5 forbids, or a full page figure.
- fig03_ladder is 5.2 in tall at 6.1 in wide (a table); in the report it needs a page-fraction of more than half a page.
- The old fig07_web.png (stand-in) is not used by the paper any more; the report Fig. 9 may still use it (fork B2).
- The paper's table text sizes (the verifier measured Table VII at 7.0 pt) were not touched and not re-measured.
