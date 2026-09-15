"""The paper source book has to survive a half-finished project.

Its whole purpose is to be regenerated after every run, including runs that
have not happened yet. A renderer that raised on a missing artifact would make
the document impossible to rebuild exactly when it is most useful, so each one
is checked against an empty run directory as well as a real one.
"""

from pathlib import Path

from deepfake_detection.documentation.paper_source import (
    BLOCK_MARKERS,
    build_blocks,
    missing_blocks,
    render_mnw,
    render_streams,
    replace_block,
)

DOCUMENT = Path("docs/research/paper-source.md")


def test_the_document_carries_every_marker_the_generator_writes() -> None:
    assert missing_blocks(DOCUMENT) == ()


def test_every_block_renders_against_a_run_that_has_produced_nothing(
    tmp_path: Path,
) -> None:
    # Every path explicitly, including ones added later with a default that
    # points at the real filesystem. A default like that makes this test read
    # actual results while claiming to test an empty run.
    blocks = build_blocks(
        run_dir=tmp_path / "runs" / "absent",
        program_run=tmp_path / "runs" / "also-absent",
        registry=tmp_path / "missing.md",
        ffpp_run=tmp_path / "runs" / "absent-ffpp",
    )

    assert set(blocks) == set(BLOCK_MARKERS)
    for name, body in blocks.items():
        # Not an empty cell: the reader is told which command would fill it.
        assert "Not generated yet" in body, name
        assert "Run `" in body, name


def test_a_missing_block_says_what_to_run_rather_than_raising(tmp_path: Path) -> None:
    body = render_streams(tmp_path)

    assert "scripts/score_streams.py" in body


def test_mnw_counts_detections_per_generator(tmp_path: Path) -> None:
    predictions = tmp_path / "evaluation" / "visual-mnw-predictions.csv"
    predictions.parent.mkdir(parents=True)
    predictions.write_text(
        "clip_id,label,probability,predicted,method\n"
        "a,1,0.9,1,gen-one\n"
        "b,1,0.2,0,gen-one\n"
        "c,1,0.1,0,gen-two\n",
        encoding="utf-8",
    )

    body = render_mnw(tmp_path)

    assert "| `gen-two` | 0 | 1 |" in body
    assert "| `gen-one` | 1 | 2 |" in body
    assert "**1**" in body


def test_replacing_a_block_leaves_the_prose_around_it_alone() -> None:
    start, end = BLOCK_MARKERS["streams"]
    text = f"before\n{start}\nold\n{end}\nafter\n"

    updated = replace_block(text, "streams", "new")

    assert updated == f"before\n{start}\nnew\n{end}\nafter\n"
