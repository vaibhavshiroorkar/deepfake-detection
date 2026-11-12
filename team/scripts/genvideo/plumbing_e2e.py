"""PLUMBING check of the generated_video module through the live API and a browser. Uses a TEST head (random-looking weights built here),
real frozen DINOv2-small and real ffmpeg, on non-sealed FF++ validation clips. It says NOTHING about detection quality."""
import json, sys, tempfile, threading, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "platform"))
import live_face_fixtures as LF
from dfd.platform.dashboard import serve
from dfd.platform.genvideo.model import save_head
from dfd.platform.modules.generated_video import GeneratedVideoModule
from dfd.platform.provenance.modules import build_provenance_modules

out = Path("data/_results/genvideo"); out.mkdir(parents=True, exist_ok=True)
td = Path(tempfile.mkdtemp())
rng = np.random.default_rng(0)
save_head(td / "TEST_head.npz", np.zeros(21), np.ones(21), [rng.normal(0, .1, (21, 8)), rng.normal(0, .1, (8, 4)), rng.normal(0, .1, (4, 1))],
          [np.zeros(8), np.zeros(4), np.zeros(1)], 0.5, {"head_id": "TEST-HEAD-PLUMBING", "evaluation": "not evaluated", "tested_scope": "plumbing test head, no evaluation",
                                                         "limitations": ["TEST HEAD: weights are random; the score means nothing"]})
fx, skipped = LF.build(td)
vids = [f for f in fx if str(f.get("path", "")).endswith(".mp4") and "ffpp" in f.get("name", f.get("fixture", ""))]
print("fixtures", [f.get("name", f.get("fixture")) for f in fx], skipped)
app = serve.create_app(live=True, modules_factory=lambda: [GeneratedVideoModule(td / "TEST_head.npz")] + build_provenance_modules())
c = app.test_client()
rows = []
for f in vids[:2]:
    p = Path(f["path"])
    with open(p, "rb") as fh:
        jid = c.post("/api/analyse", data={"file": (fh, p.name)}, content_type="multipart/form-data").get_json()["job_id"]
    t = time.time()
    while True:
        j = c.get(f"/api/jobs/{jid}").get_json()
        if j["state"] in ("done", "failed"): break
        time.sleep(0.3)
    g = next((m for m in (j["report"] or {"modules": []})["modules"] if m["module_id"] == "generated_video"), None)
    rows.append({"fixture": f.get("name", f.get("fixture")), "state": j["state"], "error": j["error"], "seconds": round(time.time() - t, 1),
                 "generated_video": None if g is None else {k: g[k] for k in ("status", "finding", "source", "version", "units_analysed", "reason")} | {"score": g["score"] and g["score"]["value"], "model": g["model"] and g["model"]["name"]}})
(out / "plumbing_e2e.json").write_text(json.dumps({"purpose": "plumbing only, TEST head, no detection claim", "rows": rows}, indent=1))
print(json.dumps(rows, indent=1))
