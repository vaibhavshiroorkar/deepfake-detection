"""Run configuration, the frozen run record and the background runner.

The record is written to disk in state "queued" with the configuration, its hash, the score-file SHA-256 and the
code state BEFORE any metric is computed. Runs execute one at a time on a background thread (or inline in
synchronous mode, used by the tests).
"""

from __future__ import annotations

import datetime as _dt
import math
import platform
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dfd.eval.ledger import DEFAULT_LEDGER
from dfd.eval.scorefile import FINAL_SPLITS, MODALITIES, POST_SPLITS, SPLITS, ScoreFileError

from . import analysis
from .catalogue import Catalogue
from .common import (BREAKDOWN_COLUMNS, N_BOOT_DEFAULT, N_BOOT_MAX, N_BOOT_MIN, PURPOSES, REPO_ROOT, ROLES,
                     SEED_DEFAULT, THRESHOLD_SOURCES, LabError, canonical_json, sha256_text)
from .store import RunStore
from .tables import TableCache, read_extras

CONFIG_KEYS = ("score_file_id", "iteration", "modality", "dataset", "split", "run_ids", "threshold_source", "threshold",
               "threshold_provenance", "n_boot", "seed", "label", "purpose", "breakdown_column", "eval_group")


def _bad(msg: str, code: str = "INVALID_CONFIG"):
    raise LabError(code, msg)


def _str(raw, key):
    v = raw.get(key)
    if not isinstance(v, str) or not v.strip():
        _bad(f"{key} is required and must be a non-empty string")
    return v.strip()


def _int(raw, key, default, lo, hi):
    v = raw.get(key, default)
    if v is None:
        v = default
    if isinstance(v, bool) or not isinstance(v, int):
        _bad(f"{key} must be an integer")
    if not lo <= v <= hi:
        _bad(f"{key} must be between {lo} and {hi}")
    return v


def validate_config(raw, entry: dict) -> dict:
    """Normalise and validate a run configuration against the catalogue entry of its score file."""
    if not isinstance(raw, dict):
        _bad("the request body must be a JSON object")
    unknown = sorted(set(raw) - set(CONFIG_KEYS))
    if unknown:
        _bad(f"unknown configuration keys: {unknown}")
    cfg = {"score_file_id": _str(raw, "score_file_id"), "iteration": _str(raw, "iteration"),
           "modality": _str(raw, "modality"), "dataset": _str(raw, "dataset"), "split": _str(raw, "split")}
    if cfg["split"] not in SPLITS:
        _bad(f"split must be one of {list(SPLITS)}")
    if cfg["modality"] not in MODALITIES:
        _bad(f"modality must be one of {list(MODALITIES)}")
    cand = next((c for c in entry["candidates"]
                 if c["iteration"] == cfg["iteration"] and c["modality"] == cfg["modality"]), None)
    if cand is None:
        _bad(f"the score file has no runs for iteration {cfg['iteration']!r} and modality {cfg['modality']!r}")
    if [cfg["dataset"], cfg["split"]] not in cand["blocks"]:
        _bad(f"the score file has no rows for dataset {cfg['dataset']!r} and split {cfg['split']!r} "
             f"in {cfg['iteration']}/{cfg['modality']}")
    rids = raw.get("run_ids")
    if rids is None:
        rids = list(cand["run_ids"])
    if not isinstance(rids, list) or not rids or not all(isinstance(r, str) for r in rids):
        _bad("run_ids must be a non-empty list of run id strings")
    if len(set(rids)) != len(rids):
        _bad("run_ids lists a run twice")
    missing = [r for r in rids if r not in cand["run_ids"]]
    if missing:
        _bad(f"run ids not in {cfg['iteration']}/{cfg['modality']}: {missing}")
    cfg["run_ids"] = [r for r in cand["run_ids"] if r in rids]          # candidate order (by seed)
    src = raw.get("threshold_source", "val_fit")
    if src not in THRESHOLD_SOURCES:
        _bad(f"threshold_source must be one of {list(THRESHOLD_SOURCES)}")
    cfg["threshold_source"] = src
    thr, prov = raw.get("threshold"), raw.get("threshold_provenance")
    if src == "fixed":
        if isinstance(thr, bool) or not isinstance(thr, (int, float)) or not math.isfinite(thr) or not 0.0 <= thr <= 1.0:
            _bad("threshold_source 'fixed' needs a numeric threshold in [0, 1]")
        if not isinstance(prov, str) or not prov.strip():
            _bad("threshold_source 'fixed' needs threshold_provenance: where the value comes from")
        cfg["threshold"], cfg["threshold_provenance"] = float(thr), prov.strip()
    else:
        if thr is not None or prov:
            _bad("threshold and threshold_provenance are only for threshold_source 'fixed'")
        cfg["threshold"], cfg["threshold_provenance"] = None, None
    cfg["n_boot"] = _int(raw, "n_boot", N_BOOT_DEFAULT, N_BOOT_MIN, N_BOOT_MAX)
    cfg["seed"] = _int(raw, "seed", SEED_DEFAULT, 0, 2**63 - 1)
    label = raw.get("label", "")
    if label is None:
        label = ""
    if not isinstance(label, str) or len(label) > 120:
        _bad("label must be a string of at most 120 characters")
    cfg["label"] = label.strip()
    purpose = raw.get("purpose", "development")
    if purpose not in PURPOSES:
        _bad(f"purpose must be one of {list(PURPOSES)}")
    if purpose == "final_reanalysis" and cfg["split"] not in FINAL_SPLITS:
        _bad("purpose 'final_reanalysis' applies only to the final splits test_in and test_x")
    if (purpose == "post_freeze_evaluation") != (cfg["split"] in POST_SPLITS):
        _bad("purpose 'post_freeze_evaluation' goes with the post-freeze split post_x, and only with it")
    cfg["purpose"] = purpose
    bc = raw.get("breakdown_column")
    if bc is not None and bc not in BREAKDOWN_COLUMNS:
        _bad(f"breakdown_column must be one of {list(BREAKDOWN_COLUMNS)}")
    cfg["breakdown_column"] = bc
    eg = raw.get("eval_group")
    if eg is not None and (not isinstance(eg, str) or not eg.strip()):
        _bad("eval_group must be a group id string")
    cfg["eval_group"] = eg.strip() if eg else None
    return cfg


def config_hash(cfg: dict) -> str:
    return sha256_text(canonical_json(cfg))


def code_state(root: Path = REPO_ROOT) -> dict:
    """Python version, git HEAD and the number of dirty entries. Read only; nulls if git is unavailable."""
    env = {"python": platform.python_version(), "git_head": None, "dirty_entries": None}

    def git(*args):
        r = subprocess.run(["git", "--no-optional-locks", *args], cwd=root, capture_output=True, text=True, timeout=20)
        return r.stdout if r.returncode == 0 else None
    try:
        head = git("rev-parse", "HEAD")
        status = git("status", "--porcelain")
        env["git_head"] = head.strip() if head else None
        env["dirty_entries"] = len([ln for ln in status.splitlines() if ln.strip()]) if status is not None else None
    except (OSError, subprocess.SubprocessError):
        pass
    return env


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


class Runner:
    def __init__(self, catalogue: Catalogue, store: RunStore, tables: TableCache, ledger_path=DEFAULT_LEDGER,
                 root: Path = REPO_ROOT, synchronous: bool = False, groups=None):
        self.catalogue, self.store, self.tables, self.groups = catalogue, store, tables, groups
        self.ledger_path, self.root, self.synchronous = ledger_path, Path(root), synchronous
        self._pool = None
        self._lock = threading.Lock()

    def submit(self, raw) -> dict:
        """Validate (LabError on a bad configuration), freeze, persist as queued, then compute. Returns the record."""
        if not isinstance(raw, dict):
            _bad("the request body must be a JSON object")
        entry = self.catalogue.find(_str(raw, "score_file_id"))
        cfg = validate_config(raw, entry)
        group = self._freeze_group(cfg, entry)
        rid = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        rec = {
            "id": rid, "created": _now(), "started": None, "finished": None, "state": "queued",
            "progress": {"stage": "queued", "done": 0, "total": 6},
            "config": cfg, "config_sha256": config_hash(cfg),
            "score_file": {"path": entry["path"], "sha256": entry["sha256"], "n_rows": entry["n_rows"]},
            "role": ROLES[cfg["split"]], "purpose": cfg["purpose"],
            "final_guard": {"required": cfg["split"] in FINAL_SPLITS, "ledger_entries": []},
            "threshold": {"value": None, "source": cfg["threshold_source"],
                          "provenance": cfg["threshold_provenance"], "fitted_on_these_rows": cfg["split"] == "val"
                          and cfg["threshold_source"] == "val_fit"},
            "data": None, "metrics": None, "confusion": None, "roc": None, "breakdown": None, "protocol": None,
            "runtime_s": None, "error": None,
            "eval_group": group,
            "environment": code_state(self.root),
            "limitations": "see export; the lab re-analyses stored scores and cannot verify training-data exclusion "
                           "beyond what the score file records",
        }
        self.store.save(rec)                         # frozen before anything is computed
        if self.synchronous:
            self._execute(rid)
        else:
            with self._lock:
                if self._pool is None:
                    self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="evallab")
            self._pool.submit(self._execute, rid)
        return self.store.get(rid)

    def _freeze_group(self, cfg: dict, entry: dict):
        """Resolve cfg['eval_group'] against the registry and record what identifies it (part of the config hash)."""
        gid = cfg.get("eval_group")
        if gid is None:
            return None
        if self.groups is None:
            _bad("this lab has no evaluation-group registry", "UNKNOWN_DATASET")
        if cfg["split"] in FINAL_SPLITS:                # only a declared re-analysis of stored sealed scores (D-096)
            try:
                ent, g = self.groups.resolve(cfg["dataset"], gid)
            except LabError:
                ent = None
            if ent is None or ent["desc"].get("sealed_reanalysis") is not True or cfg["purpose"] != "final_reanalysis":
                _bad("evaluation groups on a final (sealed) split exist only as a declared re-analysis of the stored scores",
                     "GROUP_ON_FINAL")
        ent, g = self.groups.resolve(cfg["dataset"], gid)
        if ent["score_file_id"] != cfg["score_file_id"]:
            _bad(f"group {gid!r} of {cfg['dataset']!r} belongs to score file {ent['score_path']}, not to {entry['path']}")
        if cfg["split"] != ent["desc"]["split"] or cfg["iteration"] != ent["desc"]["iteration"]:
            _bad("the run's split or iteration does not match the evaluation-group descriptor")
        cfg["eval_group_sha256"] = ent["groups_sha256"]
        return {"dataset": cfg["dataset"], "id": gid, "title": g.get("title"), "kind": g.get("kind"),
                "description": g.get("description"), "overlap": g.get("overlap"), "group_unit": g.get("group_unit"),
                "n_declared": len(g.get("sample_ids") or []), "experiment": ent["desc"]["experiment"],
                "groups_file": ent["groups_path"], "groups_sha256": ent["groups_sha256"], "descriptor": ent["descriptor"],
                "provenance": ent["desc"].get("provenance", {}), "caveats": ent["desc"].get("caveats", []),
                "score_kind": ent["desc"].get("score_kind")}

    def _group_ids(self, cfg: dict):
        gid = cfg.get("eval_group")
        if gid is None:
            return None
        ent, g = self.groups.resolve(cfg["dataset"], gid)
        if ent["groups_sha256"] != cfg.get("eval_group_sha256"):
            raise analysis.Refused("GROUPS_CHANGED", "the groups file changed after the configuration was frozen")
        return list(g.get("sample_ids") or [])

    def _update(self, rid: str, **kw) -> dict:
        rec = self.store.get(rid)
        rec.update(kw)
        self.store.save(rec)
        return rec

    def _execute(self, rid: str) -> None:
        rec = self.store.get(rid)
        t0 = time.monotonic()
        self._update(rid, state="running", started=_now())

        def progress(stage, done, total):
            self._update(rid, progress={"stage": stage, "done": done, "total": total})

        try:
            cfg = rec["config"]
            progress("reading", 0, 6)
            table = self.tables.get(self.abs_path(rec))
            extras = read_extras(self.abs_path(rec), cfg["run_ids"], cfg["modality"])
            group_ids = self._group_ids(cfg)
            out = analysis.compute(cfg, table, extras, ledger_path=self.ledger_path,
                                   frozen_sha256=rec["score_file"]["sha256"], progress=progress, group_ids=group_ids)
            self._update(rid, state="done", finished=_now(), runtime_s=round(time.monotonic() - t0, 3), **out)
        except analysis.Refused as e:
            self._update(rid, state="refused", finished=_now(), runtime_s=round(time.monotonic() - t0, 3),
                         error={"code": e.code, "message": e.message})
        except ScoreFileError as e:
            self._update(rid, state="refused", finished=_now(), runtime_s=round(time.monotonic() - t0, 3),
                         error={"code": "SCORE_FILE_INVALID", "message": str(e)})
        except Exception as e:                       # noqa: BLE001 - a failed run must be recorded, not lost
            self._update(rid, state="failed", finished=_now(), runtime_s=round(time.monotonic() - t0, 3),
                         error={"code": "RUN_FAILED", "message": f"{type(e).__name__}: {e}"})

    def abs_path(self, rec: dict) -> str:
        p = Path(rec["score_file"]["path"])
        return str(p if p.is_absolute() else self.root / p)
