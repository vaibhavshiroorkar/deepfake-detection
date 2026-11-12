"""Evaluation groups: predeclared sample groups of a post-freeze evaluation, offered by the lab as dataset -> group.

A descriptor `<name>.lab.json` sits next to its score file (default folder data/_evallab/imports) and points to a frozen
groups file (for example data/_manifests/posteval/e033_lavdf.groups.json). The groups file holds, per dataset key, the
predeclared groups with their sample ids; the descriptor records its SHA-256, so a changed groups file is refused.
Descriptors for a final (sealed) split are refused unless they declare "sealed_reanalysis": true (D-096): such a group only
re-analyses the stored scores of a logged final evaluation, and every run on it still passes the ledger guard.
Read only: nothing here writes a file, runs a model or changes a threshold.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np

from dfd.eval.scorefile import FINAL_SPLITS, POST_SPLITS, read_scores
from dfd.eval.scorefile import file_sha256 as sha256_file

from .common import (ENV_GROUP_DIRS, FULL_GROUP, GROUP_DIRS, POST_FREEZE_STATUS, SEALED_REANALYSIS_STATUS, THRESHOLD_SOURCES,
                     LabError, file_id, repo_rel)

SCHEMA = "evallab-groups/1"
REQUIRED = ("schema", "dataset", "title", "experiment", "iteration", "modality", "split", "run_ids", "score_file",
            "groups_file", "groups_sha256", "threshold")
PUBLIC_GROUP_KEYS = ("id", "title", "kind", "description", "overlap", "group_unit")


def configured_group_dirs(root: Path, dirs=None) -> list:
    if dirs is None:
        env = os.environ.get(ENV_GROUP_DIRS)
        dirs = [d for d in env.split(";") if d.strip()] if env else list(GROUP_DIRS)
    return [Path(d) if Path(d).is_absolute() else Path(root) / d for d in dirs]


class GroupRegistry:
    """Loads the descriptors and their groups files; counts each group against its score file."""

    def __init__(self, root: Path, dirs=None):
        self.root = Path(root)
        self._dirs = dirs
        self._lock = threading.Lock()
        self._counts: dict = {}

    def dirs(self) -> list:
        return configured_group_dirs(self.root, self._dirs)

    # ------------------------------------------------------------------ loading
    def _load(self) -> tuple[list, list]:
        found, skipped = [], []
        seen = set()
        for d in self.dirs():
            if not d.is_dir():
                continue
            for p in sorted(d.rglob("*.lab.json")):
                try:
                    found.append(self._load_one(p))
                except LabError as e:
                    skipped.append({"path": repo_rel(p, self.root), "reason": e.message})
        out = []
        for ent in found:
            if ent["dataset"] in seen:
                skipped.append({"path": ent["descriptor"], "reason": f"dataset key {ent['dataset']!r} is declared twice"})
                continue
            seen.add(ent["dataset"])
            out.append(ent)
        return out, skipped

    def _load_one(self, p: Path) -> dict:
        def bad(msg):
            raise LabError("GROUPS_INVALID", msg)
        try:
            desc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            bad(f"unreadable descriptor: {e}")
        if not isinstance(desc, dict) or desc.get("schema") != SCHEMA:
            bad(f"not a {SCHEMA} descriptor")
        missing = [k for k in REQUIRED if k not in desc]
        if missing:
            bad(f"descriptor lacks {missing}")
        sealed = desc["split"] in FINAL_SPLITS and desc.get("sealed_reanalysis") is True
        if not sealed and (desc["split"] in FINAL_SPLITS or desc["split"] not in POST_SPLITS):
            bad(f"split {desc['split']!r}: evaluation groups are offered only for post-freeze evaluations {list(POST_SPLITS)}, "
                "never for a sealed split unless the descriptor is a declared re-analysis of stored scores")
        thr = desc["threshold"]
        src = thr.get("source", "fixed") if isinstance(thr, dict) else None
        if src not in THRESHOLD_SOURCES or not thr.get("provenance"):
            bad("threshold must give its source and its provenance")
        if src == "fixed" and not isinstance(thr.get("value"), (int, float)):
            bad("a fixed threshold must give a value")
        gpath = self._abs(desc["groups_file"], p.parent)
        spath = self._abs(desc["score_file"], p.parent)
        if not gpath.is_file():
            bad(f"groups file {desc['groups_file']!r} not found")
        if not spath.is_file():
            bad(f"score file {desc['score_file']!r} not found")
        gsha = sha256_file(gpath)
        if gsha != desc["groups_sha256"]:
            bad(f"groups file changed: sha256 {gsha[:12]}... but the descriptor records {str(desc['groups_sha256'])[:12]}...")
        groups_doc = json.loads(gpath.read_text(encoding="utf-8"))
        ds = next((d for d in groups_doc.get("datasets", []) if d.get("dataset") == desc["dataset"]), None)
        if ds is None:
            bad(f"groups file has no dataset {desc['dataset']!r}")
        if ds.get("split") != desc["split"]:
            bad("groups file and descriptor disagree on the split")
        groups = list(ds.get("groups") or [])
        ids = [g.get("id") for g in groups]
        if not groups or len(set(ids)) != len(ids) or FULL_GROUP in ids:
            bad("groups must be a non-empty list with unique ids other than FULL")
        srel = repo_rel(spath, self.root)
        return {"dataset": desc["dataset"], "descriptor": repo_rel(p, self.root), "desc": desc, "groups": groups,
                "full": ds.get("full"), "groups_path": repo_rel(gpath, self.root), "groups_sha256": gsha,
                "score_path": srel, "score_abs": spath, "score_file_id": file_id(srel)}

    def _abs(self, rel: str, base: Path) -> Path:
        q = Path(rel)
        if q.is_absolute():
            return q
        return (self.root / q) if (self.root / q).exists() else (base / q)

    # ------------------------------------------------------------------ counts
    def _present(self, ent: dict) -> dict:
        """sample_id -> label of the scored rows of the descriptor's runs (cached by file stat and block: one score file can hold several)."""
        st = ent["score_abs"].stat()
        d = ent["desc"]
        key = (str(ent["score_abs"]), st.st_mtime_ns, st.st_size, d["dataset"], d["split"], tuple(d["run_ids"]))
        with self._lock:
            if key in self._counts:
                return self._counts[key]
        t = read_scores(ent["score_abs"])
        m = (t.dataset == d["dataset"]) & (t.split == d["split"]) & np.isin(t.run_id, d["run_ids"])
        present = {str(s): int(y) for s, y in zip(t.sample_id[m], t.label[m])}
        with self._lock:
            self._counts[key] = present
        return present

    def _public_group(self, g: dict, present: dict) -> dict:
        ids = g.get("sample_ids") or []
        scored = [present[s] for s in ids if s in present]
        out = {k: g.get(k) for k in PUBLIC_GROUP_KEYS}
        out.update({"n_samples": len(scored), "n_fake": int(sum(scored)), "n_real": len(scored) - int(sum(scored)),
                    "n_groups": g.get("n_groups"), "n_declared": len(ids), "n_missing_scores": len(ids) - len(scored)})
        return out

    # ------------------------------------------------------------------ public
    def listing(self) -> dict:
        ents, skipped = self._load()
        out = []
        for e in ents:
            d = e["desc"]
            try:
                present = self._present(e)
            except Exception as ex:  # noqa: BLE001 - an unreadable score file makes the dataset unavailable, not the lab
                skipped.append({"path": e["descriptor"], "reason": f"score file unreadable: {ex}"})
                continue
            full = self._public_group({**e["full"], "id": FULL_GROUP}, present) if e.get("full") else None
            if full:
                full["warning"] = ("Whole eligible pool: re-analyses every stored score of this dataset. Model inference was run "
                                   "once in the experiment and is not repeated, but the bootstrap over all rows takes longer.")
                full["estimate"] = f"{full['n_samples']} stored scores x bootstrap resamples (typically under a minute)"
            out.append({
                "dataset": e["dataset"], "title": d["title"], "experiment": d["experiment"], "iteration": d["iteration"],
                "modality": d["modality"], "head": d.get("head"), "split": d["split"],
                "status": SEALED_REANALYSIS_STATUS if d["split"] in FINAL_SPLITS else POST_FREEZE_STATUS,
                "sealed_reanalysis": d["split"] in FINAL_SPLITS,
                "score_file_id": e["score_file_id"], "score_file": e["score_path"],
                "threshold": d["threshold"], "score_kind": d.get("score_kind"), "eligibility": d.get("eligibility"),
                "caveats": d.get("caveats", []), "provenance": d.get("provenance", {}),
                "groups_file": e["groups_path"], "groups_sha256": e["groups_sha256"], "descriptor": e["descriptor"],
                "groups": [self._public_group(g, present) for g in e["groups"]], "full": full,
            })
        return {"datasets": out, "skipped": skipped, "dirs": [repo_rel(x, self.root) for x in self.dirs()]}

    def resolve(self, dataset: str, gid: str) -> tuple[dict, dict]:
        """(entry, group with sample_ids) for a dataset key and a group id, or LabError."""
        ents, _ = self._load()
        ent = next((e for e in ents if e["dataset"] == dataset), None)
        if ent is None:
            raise LabError("UNKNOWN_DATASET", f"no evaluation groups for dataset {dataset!r}")
        if gid == FULL_GROUP:
            if not ent.get("full"):
                raise LabError("UNKNOWN_GROUP", f"dataset {dataset!r} offers no full-pool evaluation")
            return ent, {**ent["full"], "id": FULL_GROUP}
        g = next((g for g in ent["groups"] if g.get("id") == gid), None)
        if g is None:
            raise LabError("UNKNOWN_GROUP", f"dataset {dataset!r} has no group {gid!r}")
        return ent, g
