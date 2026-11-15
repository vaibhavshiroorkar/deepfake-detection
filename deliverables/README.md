# Deliverables

The black book (project report), research paper, slides and poster for the final-year project. The report, paper and slide builders were restored from the October submission (tag `team/submission-final`, folder `deliverables_v2/`). They were made for the Somaiya Vidyavihar University format.

Every number in a deliverable is a placeholder filled from `results/numbers.json`, which `scripts/build_numbers.py` builds from the committed result files. To update all deliverables after a new result:

    .venv/Scripts/python.exe scripts/build_numbers.py
    .venv/Scripts/python.exe scripts/make_figures.py
    .venv/Scripts/python.exe deliverables/slides/build_deck.py      # add --draft to mark missing numbers [pending]
    powershell -File deliverables/slides/export.ps1                  # PDF of the deck, through PowerPoint

## Slides

`slides/build_deck.py` writes `build/Deepfake_Detection_Presentation.pptx`: 14 main slides that follow the paper, a divider and four backup slides. It draws on `slides/college-template.pptx`, which is the college slide master (logos, footer bands, layouts) taken from the October submission deck with its slides removed. `slides/deck_kit.py` holds the drawing helpers and `slides/slide_common.py` the grid and type scale. Every slide has speaker notes.

A deliverable that names a number missing from `results/numbers.json` fails to build. It never prints a gap or an old value.

Tools still needed for PDFs: a LaTeX distribution for the paper (MiKTeX or TeX Live), and LibreOffice or Microsoft Word to export the report and slides to PDF.
