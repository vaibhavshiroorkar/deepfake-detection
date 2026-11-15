# Deliverables

The black book (project report), research paper, slides and poster for the final-year project. The report, paper and slide builders were restored from the October submission (tag `team/submission-final`, folder `deliverables_v2/`). They were made for the Somaiya Vidyavihar University format.

Every number in a deliverable is a placeholder filled from `results/numbers.json`, which `scripts/build_numbers.py` builds from the committed result files. To update all deliverables after a new result:

    .venv/Scripts/python.exe scripts/build_numbers.py
    # then rebuild each deliverable (commands to be added per folder)

A deliverable that names a number missing from `results/numbers.json` fails to build. It never prints a gap or an old value.

Tools still needed for PDFs: a LaTeX distribution for the paper (MiKTeX or TeX Live), and LibreOffice or Microsoft Word to export the report and slides to PDF.
