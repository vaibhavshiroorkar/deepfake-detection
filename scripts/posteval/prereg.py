"""Write the ordering record of a preregistered post-freeze block (E-033, E-034): SHA-256 of the block in docs/EXPERIMENTS.md and the time.

Usage: python scripts/posteval/prereg.py E-033|E-034
Span: from the line starting '## <ID>:' to the first LINE equal to '- Result: (to be appended below after the run)', inclusive, UTF-8, LF.
Same rule as scripts/modern_image/prereg.py. Refuses to overwrite an existing prereg.json.
"""
import hashlib, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
END = "- Result: (to be appended below after the run)"
eid = sys.argv[1]
lines = (ROOT / "docs" / "EXPERIMENTS.md").read_text(encoding="utf-8").split("\n")
start = next(i for i, l in enumerate(lines) if l.startswith(f"## {eid}:"))
end = next(i for i in range(start, len(lines)) if lines[i] == END)
block = "\n".join(lines[start:end + 1])
out = ROOT / "data" / "_results" / "posteval" / eid / "prereg.json"
out.parent.mkdir(parents=True, exist_ok=True)
if out.exists():
    sys.exit(f"{out} exists; not overwritten")
rec = {"experiment": eid, "written_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "committed": False,
       "block_sha256": hashlib.sha256(block.encode("utf-8")).hexdigest(), "block_chars": len(block), "block_lines": end - start + 1,
       "span_rule": f"from '## {eid}:' to the first LINE equal to '{END}' inclusive, UTF-8, lines joined with LF",
       "note": "written before any sample of the experiment was decoded or scored"}
out.write_text(json.dumps(rec, indent=1))
print(json.dumps(rec, indent=1))
