"""One renderer for the locked pages.

Streams, Fusion and Explainability were three copies of the same twenty lines,
differing only in which dict from stream_spec they read. Now they each hand that
dict to render() and the layout is defined once, so the three pages cannot drift
apart in wording or structure.

The lock is stated twice at most: the icon in the title, and the reason in the
callout. It is not also spelled out in the prose.
"""


def render(st, spec: dict):
    """Title, why it is locked, and what will land here."""
    st.title(f":material/lock: {spec['title']}")
    st.caption(spec["note"])
    st.info(spec["status"])

    st.subheader("What lands here")
    for name, description in spec["views"]:
        st.markdown(f"**{name}.** {description}")
