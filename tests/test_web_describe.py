"""The description on top of an answer: short, and never saying the same thing twice."""

from deepfake_detection.webapp.describe import tidy


def test_the_description_keeps_two_sentences_and_drops_a_repeat():
    text = (
        "The video shows a person in a room with two monitors. The heatmap shows a warm area on the "
        "person's face, which covers the eyes and nose. In the video, the person's eyes and nose are "
        "visible, and the heatmap highlights the area around the eyes and nose."
    )
    assert tidy(text) == (
        "The video shows a person in a room with two monitors. The heatmap shows a warm area on the "
        "person's face, which covers the eyes and nose."
    )
    assert (
        tidy("A man talks. A man talks. He sits by a window.")
        == "A man talks. He sits by a window."
    )
