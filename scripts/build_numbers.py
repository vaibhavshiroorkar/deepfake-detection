"""Build docs/NUMBERS.json: one row per number of the logged sources, each with the ID of the entry it comes from.

    python scripts/build_numbers.py            (any Python 3; reads docs/ and data/_results, writes docs/NUMBERS.json, prints the table)

Sources (the IDs are the experiment, measurement or data-log entries; nothing is typed by hand except the two small curated files):
  docs/EXPERIMENTS.md   split into entries E-xxx, M-xxx, V-xxx (the generated comparison tables are attributed to E-001, E-002, E-003);
                        text outside an entry (environment, templates, protocol definitions) gets the ID EXPERIMENTS:<heading>
  data/_results/**      result files written by scripts (ranking, cost record, battery, M-004, I-1, S-1), attributed by FILE_IDS
  docs/DATA_LOG.md      dataset facts: ID DATA_LOG:<heading>
  docs/RESEARCH.md      rows R-001 to R-077: the numbers recorded for a paper keep its reference ID
  docs/NUMBERS_LIT.json further numbers quoted from the literature: each row needs a reference ID R-xxx and the place in the source
  docs/NUMBERS_DERIVED.json  numbers computed from logged ones (a share, a sum): formula and inputs, checked here
The header stores the SHA-256 of the text sources; scripts/check_docs.py fails when they no longer match (the file is stale).
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numbers_lib as NL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "NUMBERS.json"
FILE_IDS = [  # (glob under data/_results, ID)
    ("T-V/*.md", "E-001"), ("T-V/*.json", "E-001"), ("T-I/*.md", "E-002"), ("T-I/*.json", "E-002"), ("T-A/*.md", "E-003"), ("T-A/*.json", "E-003"),
    ("cost_record.json", "E-001"), ("I-0/battery.md", "E-004"), ("I-0/battery.json", "E-004"), ("I-0/metrics.json", "E-004"),
    ("I-0/parity.json", "E-004"), ("I-0/web_real_check.json", "E-004"), ("I-0/*/metrics.json", "E-004"),
    ("M-004/m004.json", "M-004"), ("I-1/result.md", "E-005"), ("I-1/selection.json", "E-005"), ("I-1/analysis.json", "E-005"),
    ("S-1/summary.json", "E-006"), ("S-1/result.md", "E-006"),
    ("M-005/summary.json", "M-005"), ("M-005/result_block.md", "M-005"), ("baseline/e013b.json", "E-013b"), ("baseline/e013b.md", "E-013b"),
    ("I-1b/e016.json", "E-016"), ("I-1c/e017.json", "E-017"),
]
TEXT_SOURCES = ["docs/EXPERIMENTS.md", "docs/DATA_LOG.md", "docs/RESEARCH.md", "docs/NUMBERS_LIT.json", "docs/NUMBERS_DERIVED.json"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.is_file() else "missing"


def experiments_rows():
    text = (ROOT / "docs/EXPERIMENTS.md").read_text(encoding="utf-8")
    cur, head, buf, rows = "EXPERIMENTS:preamble", "preamble", [], []

    def flush():
        if buf:
            rows.extend(NL.source_rows(chr(10).join(buf), cur, f"docs/EXPERIMENTS.md#{head}"))
            buf.clear()

    for line in text.splitlines():
        h = re.match(r"^(#{1,4})\s+(.*)", line)
        if h:
            flush()
            level, title = len(h.group(1)), h.group(2).strip()
            m = re.match(r"((?:E|M|V)-\d{3}[a-z]?)", title)
            t = re.match(r"(T-[VIA]).*generated", title)
            if m:
                cur, head = m.group(1), m.group(1)
            elif t:
                cur, head = {"T-V": "E-001", "T-I": "E-002", "T-A": "E-003"}[t.group(1)], f"comparison table {t.group(1)}"
            elif level <= 2 or not cur[:2] in ("E-", "M-", "V-"):
                cur, head = f"EXPERIMENTS:{title[:50]}", title[:50]
        buf.append(line)
    flush()
    return rows


def json_leaves(o, out):
    if isinstance(o, dict):
        for v in o.values():
            json_leaves(v, out)
    elif isinstance(o, list):
        for v in o:
            json_leaves(v, out)
    elif isinstance(o, bool) or o is None:
        pass
    elif isinstance(o, (int, float)):
        out.append(o)
    elif isinstance(o, str):
        out.append(o)


def result_rows():
    rows, base = [], ROOT / "data/_results"
    for pat, id_ in FILE_IDS:
        for f in sorted(base.glob(pat)):
            rel = f"data/_results/{f.relative_to(base).as_posix()}"
            if f.suffix == ".json":
                try:
                    leaves = []
                    json_leaves(json.loads(f.read_text(encoding="utf-8")), leaves)
                except Exception:
                    continue
                for v in leaves:
                    if isinstance(v, str):
                        rows.extend(NL.source_rows(v, id_, rel))
                    else:
                        s = repr(v) if isinstance(v, int) else format(v, ".10g")
                        rows.append(dict(value=s, id=id_, kind="number", src=rel, ctx=""))
                        if isinstance(v, float):
                            for d in (2, 3, 4):
                                rows.append(dict(value=format(round(v, d), f".{d}f"), id=id_, kind="number", src=rel, ctx=""))
            else:
                rows.extend(NL.source_rows(f.read_text(encoding="utf-8", errors="ignore"), id_, rel))
    return rows


def datalog_rows():
    rows, cur, buf = [], "preamble", []
    text = (ROOT / "docs/DATA_LOG.md").read_text(encoding="utf-8")

    def flush():
        if buf:
            rows.extend(NL.source_rows("\n".join(buf), f"DATA_LOG:{cur}", f"docs/DATA_LOG.md#{cur}"))
            buf.clear()

    for line in text.splitlines():
        h = re.match(r"^#{1,4}\s+(.*)", line)
        if h:
            flush()
            cur = h.group(1).strip()[:50]
        buf.append(line)
    flush()
    return rows


def research_rows():
    """Numbers recorded in a row of the literature matrix keep the reference ID of that row (R-xxx)."""
    rows = []
    for line in (ROOT / "docs/RESEARCH.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\|\s*(R-\d{3})\s*\|", line)
        if m:
            rows.extend(NL.source_rows(line, m.group(1), "docs/RESEARCH.md"))
    return rows


def lit_rows():
    f = ROOT / "docs/NUMBERS_LIT.json"
    if not f.is_file():
        return []
    rows = []
    for r in json.loads(f.read_text(encoding="utf-8")):
        assert re.fullmatch(r"R-\d{3}", r["id"]), f"literature row without a reference ID: {r}"
        assert r.get("where"), f"literature row without a place in the source: {r}"
        rows.append(dict(value=str(r["value"]), id=r["id"], kind="literature", src="docs/NUMBERS_LIT.json", ctx=r.get("what", "")))
    return rows


def derived_rows(known):
    f = ROOT / "docs/NUMBERS_DERIVED.json"
    if not f.is_file():
        return []
    rows = []
    for r in json.loads(f.read_text(encoding="utf-8")):
        val = eval(r["formula"], {"__builtins__": {}}, {})   # arithmetic on literals only, written by hand in the file
        assert abs(val - float(r["value"])) <= float(r.get("tol", 0.5 * 10 ** -(len(r["value"].split(".")[1]) if "." in r["value"] else 0))) + 1e-9, \
            f"derived number {r['value']} does not equal {r['formula']} = {val}"
        rows.append(dict(value=str(r["value"]), id=r["id"], kind="derived", src="docs/NUMBERS_DERIVED.json", ctx=f"{r['formula']} ({r.get('what', '')})"))
    return rows


def main() -> int:
    rows = experiments_rows() + result_rows() + datalog_rows() + research_rows() + lit_rows()
    rows += derived_rows(rows)
    seen, uniq = set(), []
    for r in rows:
        k = (r["value"], r["id"], r["src"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    head = dict(generated_by="scripts/build_numbers.py", source_sha256={p: sha(ROOT / p) for p in TEXT_SOURCES}, n_rows=len(uniq),
                note="one row per number of the logged sources; data/_results is not in git, so rows from it are copied here")
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write('{"header": ' + json.dumps(head) + ',\n"rows": [\n')
        fh.write(",\n".join(json.dumps(r, ensure_ascii=False) for r in uniq))
        fh.write("\n]}\n")
    by = collections.Counter(r["id"].split(":")[0] for r in uniq)
    print(f"{len(uniq)} rows written to {OUT.relative_to(ROOT)}")
    for k, v in sorted(by.items()):
        print(f"  {k:<14}{v:>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
