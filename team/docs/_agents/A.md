# Stream A report: paper, report and slides skeletons

Date: 2026-10-24. Scope: first versions of the IEEE paper (LaTeX), the technical report (.docx) and the presentation (.pptx), built from the supplied templates. No results exist yet, so every results place says "to be filled from E-xxx". Nothing was committed or pushed. Only these paths were written: `paper/`, `report/`, `slides/`, `assets/branding/`, `docs/_agents/A.md`. Scratch work is in `C:\Users\admin\AppData\Local\Temp\claude\d--Final-Year-Project\4375ec7f-df8d-4495-a96a-01f47710e634\scratchpad\streamA`. The originals in `Document Template/` were only read (copies were opened, converted or rendered).

## 1. Tools found and versions

| Tool | Result |
|---|---|
| pdflatex | MiKTeX-pdfTeX 4.27 (MiKTeX 26.5) |
| latexmk | 4.88 (9 March 2026). MiKTeX's `latexmk.exe` stub fails with "could not find the script engine perl". It works when Git for Windows' perl (5.38.2, `C:\Program Files\Git\usr\bin`) is on PATH for the shell. No system change was made. |
| MiKTeX auto install | `initexmf --set-config-value "[MPM]AutoInstall=1"` returned 0. The utility has no read-back option, so the value was not read back. All packages needed (IEEEtran, cite, booktabs, xcolor, hyperref, amsmath) were already present. |
| Microsoft Word | Installed (16.0.19127.20860), driven through COM from PowerShell. Used to convert .doc to .docx, update the table of contents and export PDF for checking. |
| Microsoft PowerPoint | Installed, driven through COM. Used to render the slides to PNG and PDF. |
| LibreOffice, pandoc, ImageMagick | Not installed. |
| python-docx 1.2.0, python-pptx 1.0.2 | Run with `uv run --no-project --with ...`. The project `.venv` and `uv.lock` were not touched. |
| pdftotext, pdftoppm | MiKTeX copies, used for text checks and page images. |

## 2. What the three templates are

### 2.1 `67F_B Tech Semester VII Project Report template.doc` (Word 97-2003, 5 pages)

It is a set of guidelines plus a skeleton, not a cover/certificate template. There is no logo and no cover page in it. Rules it states (quoted as a list):

- A4, Times New Roman, 1.5 line spacing, 12 pt body. Margins: Left 1.5", Right 1", Bottom 1", Top 1.5". Page number at bottom centre.
- Chapter subtopics 14 pt Times New Roman bold, 1.5 spacing, left aligned and justified. Body justified.
- Each chapter starts with a boxed description of 4 to 5 lines ("This chapter presents ...") under the chapter title. Each chapter ends with a summary of the chapter and an outline of the next.
- No page number on the first three pages, abstract and contents. Lower-case roman numerals for List of Figures through Nomenclature. Then arabic for the thesis (the template says to use three separate files; one file with three sections was used here).
- Expected references: 10 to 15. Literature survey: about 2 to 5 pages. Total report length: at least 30 to 60 pages. References quoted as [1] etc. Bibliography in standard format, ordered as cited.
- Chapters it names: 1 Introduction (overview, motivation, scope); 2 Literature survey (objective defined at the end); 3 Project design (mathematical or logical background); 4 Implementation and experimentation (results and analysis); a final chapter "Conclusions and further work" (8 to 10 lines, then points, then future scope). Then Bibliography, Publications (if any), Appendix A/B, Acknowledgement. "Number of chapters may change from thesis to thesis."
- First page of the bound thesis embossed in gold (a binding matter, nothing to do in the file).
- Look: page border (single line, 24 pt from page edge), footer in Times New Roman bold 12 pt with a double rule above it in colour `622423`, footer text "Department of Electronics and Telecommunication Engineering Semester VII 2020-24 Batch Page | N" (the second page onward shows "2014-18 Batch" and no "Engineering").
- Chapter heading: centred bold 16 pt, "Chapter N" then title. The description box is a bordered text box with italic 12 pt text.

Conversion to .docx: done by Microsoft Word itself (`SaveAs2`, docx format) on a copy, saved as `report/template_base_converted_from_doc.docx`. Word's own PDF export of that conversion was compared by eye with the original rendering: same page count (5), border, footer rule, chapter box and text. Lost or changed: nothing that was observed; the text-box shapes are VML text boxes in the converted file and were not reused (see 4.2).

### 2.2 `DeepFake_Detection_System_Presentation (2).pptx` (9 slides, a Google Slides export)

- Slide size 12192000 x 6858000 EMU (16:9). One master, 10 layouts (TITLE, OBJECT_WITH_CAPTION_TEXT, TITLE_ONLY, PICTURE_WITH_CAPTION_TEXT, OBJECT, TWO_OBJECTS, SECTION_HEADER, TWO_OBJECTS_WITH_CAPTION_TEXT, BLANK, VERTICAL_TEXT). The theme itself is the default Office colour set; the college look comes from shapes on the master.
- Master decorations: a dark red block bottom left (0, 6080760, 838200 x 790491), a long dark red bar along the bottom (838199, 6356350, 9331411 x 514901) and a bright red block bottom right (10169610, 6356350, 2022390 x 514900). Logos on the master: Somaiya Vidyavihar University logo top left (193862, 141576, 2316256 x 708705) and Somaiya Trust logo top right (11124596, 226123, 728472 x 539609).
- Palette: `A42530` (dark red, bars, table header, title accents), `D9222A` (bright red, bottom right block), `3F3F3F` (grey footer and slide number text), `F5F5F5` (table banding), `000000`, `FFFFFF`.
- Fonts: Times New Roman for all titles and body text (titles bold 32 pt, centred; body 12 to 14 pt). Embedded Fira Sans and Marcellus are present in the file but not visibly used. Footer placeholder text "LY Project" (centred) and slide number bottom right.
- Slide 1 (title): title (two lines, 24 pt bold, centred top), department line under it, two campus photographs (Aryabhat and Bhaskaracharya buildings), "Group Members" block bottom right, "Guide" bottom left.
- Slides 2 to 9: title plus a text box at (737846, 971403), one table (slide 4), three diagram pictures and a Gantt picture (not carried over).
- Old content (text, tables, diagrams, timeline) was NOT reused.

### 2.3 `Deepfake_detection_system_report_ (1).pdf` (41 pages, A4, made with LaTeX, 19 Sep 2026)

Used only for the cover, certificate and declaration structure and for identity details. Structure: cover (university logo, university, school, department, title and subtitle in dark red, "Submitted in Partial Fulfillment of Requirements For the Degree Of", degree, branch, "By" with names and roll numbers, "Guide" with name and designation, university address, year), Certificate (guide, head of department, examiner, director signature lines, date, place), Declaration (standard text, three signature blocks), Abstract, Acknowledgements, Contents, List of Figures, List of Tables, Abbreviations, chapters (Introduction, Literature Review, Methodology, Results and Discussion, Limitations and Future Work, Conclusions), Bibliography, Appendices. Footer: "Department of Electronics and Computer Engineering  Semester VIII  2026-27 Batch  Page n" with a red rule. Its body text was not reused. The standard declaration text is college boilerplate and was reused word for word.

## 3. Identity details extracted

| Field | Value used | Source |
|---|---|---|
| University | Somaiya Vidyavihar University | pdf, pptx logo |
| School | K. J. Somaiya School of Engineering (logo adds "formerly K J Somaiya College of Engineering") | pdf, pptx |
| Address | Vidyavihar, Mumbai - 400 077 | pdf |
| Degree | Bachelor of Technology (Electronics and Computer Engineering) | pdf |
| Students and roll numbers | Vaibhav Shiroorkar 16014024805; Shivansh Gupta 16014023052; Adithya Jithesh 16014024803 | pdf and pptx agree on names and numbers |
| Guide | Dr. Tejashree Phatak, Assistant Professor | pdf, pptx |
| Head of Department | Dr. Jagannath Nirmal | pdf |
| Director | not named anywhere (signature line left blank) | pdf |
| Academic year | 2026-2027 (cover), "2026-27 Batch" (footer) | pdf |
| Footer text of the slides | "LY Project" | pptx |
| Place | Mumbai-77 | pdf |

## 4. Files produced

### 4.1 Branding assets (`assets/branding/`, extracted unchanged from the pptx media)

| File | What it is | Where it appears |
|---|---|---|
| `somaiya_vidyavihar_logo.jpg` | University and school logo | pptx master (top left), report cover |
| `somaiya_trust_logo.png` | Somaiya Trust logo | pptx master (top right) |
| `campus_aryabhat_building.png` | Campus photograph | pptx title slide, left |
| `campus_bhaskaracharya_building.png` | Campus photograph | pptx title slide, right |

### 4.2 Paper (`paper/`)

| Path | Contents | Contribution |
|---|---|---|
| `paper/main.tex` | IEEEtran, `[conference]` (two column) paper. Three proposed titles in a comment (proposal 1 used as working title). Abstract skeleton, Introduction, Related Work (five subsections, paraphrased, from READ papers only), Method skeleton, Experimental Protocol (with iteration table), Results (empty, "to be filled from E-xxx"), Discussion, Limitations, Conclusion. Placeholders are red `[TBD: ...]` and `[to be filled from E-xxx]`. | The paper deliverable, first version. |
| `paper/refs.bib` | 18 entries, all from READ papers (R-010 only partly read, flagged in the note). Venue is given only where it was printed in the file read; otherwise arXiv preprint with a "venue not verified" note. | Verified bibliography. |
| `paper/main.pdf`, `paper/main.bbl` | Build outputs (4 pages). Can be left out of git. | Check of the compile. |

Compile: `latexmk -pdf main.tex` succeeded (exit 0), 4 pages, BibTeX with IEEEtran.bst, no undefined references or citations, no overfull boxes after one fix (a table with too wide placeholders). Remaining warnings: five harmless underfull hbox lines in the abstract and in justified paragraphs. No absolute paths, no non-standard packages, so it should compile on Overleaf as is (not tried on Overleaf). The one result-like number in the paper is the 0.9450 and 0.7718 mean AUC of Xception from DeepfakeBench (R-017), written as that paper's number under its protocol, with a `% source` marker.

Not cited because they are TO READ: R-019, R-020.

### 4.3 Report (`report/`)

| Path | Contents |
|---|---|
| `report/DeepFake_Detection_Technical_Report_draft.docx` | 26 pages. Cover (logo, identity, working title in `A42530`), Certificate, Declaration, Abstract, Acknowledgement, Contents (real Word TOC field, updated), List of Figures, List of Tables, Nomenclature, Chapters 1 to 5, Bibliography (18 numbered references in order of first citation), Publications, Appendix A and B. Chapter 2 holds the literature survey; Chapter 3 the problem statement, comparison rules and the definitions of ECE and temperature scaling (from R-016); Chapter 4 has placeholders for I-0 to I-7, final comparison, rPPG test. Every chapter has the boxed description and a closing summary and outline. |
| `report/template_base_converted_from_doc.docx` | The template .doc converted by Word. Base for the build. |
| `report/build_report.py` | Rebuilds the report from the base (`uv run --no-project --with python-docx python report/build_report.py`), then Word is used to update the TOC. Identity and wording that need confirming are marked `CONFIRM` in the script. |
| `report/DeepFake_Detection_Technical_Report_draft_preview.pdf` | Word's PDF export, for checking only. |

What was kept from the template: A4, the page border, Times New Roman, the footer style (bold, double rule in `622423`), the chapter heading layout, the boxed chapter description, the section scheme (no number, roman, arabic from 1). What was changed on purpose: footer text (department, year), margins set to the written rule (see question 3), and the VML text box replaced by a one-cell bordered table (same look, easier to edit). Chapter 2 is about 2 pages in draft; the template asks for 2 to 5 pages, the full text will grow it.

### 4.4 Slides (`slides/`)

| Path | Contents |
|---|---|
| `slides/DeepFake_Detection_Presentation_draft.pptx` | 19 slides built from the supplied deck's master, layouts, theme, logos, bars and title-slide photos. All old content removed (checked: old text and the old diagram images are no longer in the package). Order: 1 title; 2 problem; 3 scope and claims; 4 plan; 5 protocol; 6 literature table; 7 MVP (I-0); 8 to 14 Iterations 1 to 7 (tried, why, what happened); 15 rPPG feasibility; 16 final comparison table; 17 limitations; 18 conclusion; 19 references. Speaker notes on every slide. Result slides (7 to 16 and 18) carry a "RESULT: TO BE FILLED" tag. |
| `slides/build_slides.py` | Rebuild script (`uv run --no-project --with python-pptx python slides/build_slides.py`). Source pptx is opened read only. |
| `slides/DeepFake_Detection_Presentation_draft_preview.pdf` | PowerPoint's PDF export, for checking only. |

## 5. How it was verified

- Paper: latexmk compile, grep of the log for warnings, page images viewed (title block, abstract, table, references), grep for en and em dashes (none) in source, bib and PDF text.
- Report: Word re-opened the python-docx output, updated fields, exported PDF; pages 1 to 8, 10, 12, 14, 17, 19 and 22 were viewed as images (cover, certificate, declaration, TOC, nomenclature, footer text and numbers, chapter boxes). PDF text has no en or em dashes.
- Slides: PowerPoint exported all 19 slides to PNG; slides 1 to 9, 12, 14 to 19 were viewed. Checked: both logos at the original positions, dark and bright red bars, grey footer and slide number, the campus photos on slide 1, the tag, the tables, no title overlapping a logo (this was fixed once: long titles were shortened and the title box narrowed). Slides 10, 11, 13 are the same layout as the viewed iteration slides.
- Dashes: grep of both build scripts and the unzipped pptx slide and notes XML found none.

## 6. Questions for the main agent (conflicts or unclear points; nothing was guessed silently)

1. Department name. The pptx says "Department of Engineering Electronics & Telecommunication" (odd wording), the .doc footer says "Electronics and Telecommunication Engineering", the pdf says "Electronics and Computer Engineering" (cover, degree and footer). I used "Electronics and Computer Engineering" (newest file, September 2026) everywhere. Please confirm; it is one string in each build script (`DEPARTMENT`, `BRANCH`, `FOOTER_TEXT`) and in the paper's `\author` block.
2. Semester and batch wording. The .doc is a Semester VII template with "2020-24 Batch" and "2014-18 Batch" (both stale), the pdf says "Semester VIII 2026-27 Batch". I used "Semester VII 2026-27 Batch" in the footers and "end of Semester VII of LY B. Tech" in the certificate, because the submission is in October 2026 and the template is for Semester VII. Please confirm semester and the batch string (is "batch" the admission years, for example 2023-27?).
3. Margins. The .doc states Left 1.5", Right 1", Top 1.5", Bottom 1", but its actual page setup is about 1" left, 0.79" right and top, 0.49" bottom. I applied the written rule. Tell me if the examiners want the template's actual setup.
4. Title. The cover uses the working title "Multimodal Deepfake Detection for Images, Video and Audio" with the subtitle "A Hypothesis-Driven Study Under a Source-Disjoint Protocol". These are proposals; the old titles were not reused. Three proposals are listed in the paper source.
5. Student order. The pdf lists Vaibhav, Shivansh, Adithya; the pptx lists Vaibhav, Adithya, Shivansh. I used the pdf order in the report and paper, the pptx order on the title slide (as in that deck). Please give one order.
6. Slide footer "LY Project" kept as in the pptx. Confirm it is still wanted.
7. Acknowledgement position. The template lists it last; the old pdf has it after the Abstract. I put it after the Abstract and kept only a short factual text plus a placeholder. Confirm.
8. Director's name (not given anywhere) and the examiner lines are left blank. Date line left blank.
9. Paper: venue and style. I used IEEE conference style because no venue is named; switch to `[journal]` if the guide prefers. No e-mail addresses were supplied (placeholder in the author block). The guide is listed as the last author; confirm.
10. Report length. The template asks 30 to 60 pages and 10 to 15 references. The draft cites 18 papers (all READ) and is 26 pages with placeholders. Tell me whether to trim to 15.
11. rPPG slide and Chapter 3.7 describe the four-step feasibility test from O-010 as written in DECISIONS.md. They are structure, not a decision about thresholds.
12. ENV-DESKTOP-GPU figures (12 GB, 24 threads, about 31 GB RAM) in Chapter 4.1 were taken from docs/PLAN.md, not from docs/EXPERIMENTS.md, which I did not read. Please check them against the record.

## 7. What was not done or not possible

- The paper was compiled with MiKTeX only, not on Overleaf.
- No figures exist yet, so List of Figures and List of Tables are placeholders (Word can generate them once captions exist).
- The report's chapter 2 and the paper's related work were written from the matrix in docs/RESEARCH.md and the title pages and abstracts of the source files; I did not re-read the full papers. Statements were kept to what the matrix records. Statements that depend on the matrix text (for example "reports weak performance on content from outside the dataset" for DFDC) should be re-checked by whoever owns RESEARCH.md if they want a second verification.
- Venues for the arXiv-only entries are deliberately not stated (common knowledge such as ICCV 2019 for FaceForensics++ was not written because the file does not print it).
- No branding exists for a "bound thesis cover in gold"; that is a binding instruction.
- Git: not touched. `paper/main.pdf`, `paper/main.bbl` and the two preview PDFs are build outputs; the main agent may add them to `.gitignore` or keep them.
