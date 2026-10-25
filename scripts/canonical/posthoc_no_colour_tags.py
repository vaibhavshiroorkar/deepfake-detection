"""E-019 post-hoc (not preregistered): metadata audit on the canonical DeepAction rows with the two
colour tags that survived protocol v1 (color_transfer, color_primaries) removed. Reads the CSV from reaudit.py."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import reaudit as ra  # noqa: E402

rows = list(csv.DictReader(open(ra.OUT / "deepaction_canonical_features.csv", encoding="utf-8")))
for r in rows:
    r["prompt_id"] = int(r["prompt_id"])
    for c in ra.e18.NUM + ra.PIX:
        r[c] = float(r[c]) if r[c] not in ("", "nan") else float("nan")
rng = np.random.default_rng(ra.e18.SEED)
res = ra.audit(rows, ra.e18.NUM, [c for c in ra.e18.CAT if c not in ("color_transfer", "color_primaries")], "posthoc", rng)
(ra.OUT / "e019_posthoc_no_colour_tags.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
