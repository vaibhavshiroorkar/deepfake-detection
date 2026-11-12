"""ForenSynths / CNNDetection image sets and the diffusion test set.

Layouts (inspected on the real extracted folders, 2026-10-24):

  progan_val:        data/forensynths/progan_val/<class>/{0_real,1_fake}/*.png            20 LSUN classes
  progan_test:       data/forensynths/progan_test/progan/<class>/{0_real,1_fake}/*.png    20 LSUN classes
  cnn_synth_subset:  data/forensynths/cnn_synth_subset/<generator>/{0_real,1_fake}/*       biggan, deepfake,
                     gaugan, stargan, whichfaceisreal (no class level; whichfaceisreal is .jpeg)
                     data/forensynths/cnn_synth_subset/<generator>/<class>/{0_real,1_fake}/*   stylegan
                     (bedroom, car, cat), stylegan2 (car, cat, church, horse), cyclegan (apple, horse,
                     orange, summer, winter, zebra)
  diffusion_test:    data/diffusion_test/diffusion_datasets/<folder>/{0_real,1_fake}/*   a folder holds ONE
                     class only: ten fake folders (dalle, glide_100_10, glide_100_27, glide_50_27, guided,
                     ldm_100, ldm_200, ldm_200_cfg) hold 1_fake only; imagenet and laion hold 0_real only.

Groups and why.
  * ProGAN val and test: group = the LSUN class. These images have no identity and no source video, so the
    class is the coarsest source grouping available (images of one class share scene statistics and, for the
    real images, the LSUN source collection). group_basis = class, identity_known = false. The group id is
    namespaced by dataset, so a class in progan_val and the same class name in progan_test are different
    groups: the two sets are different images, and class-level disjointness is only enforced inside one set.
  * CNN synthesis subset and diffusion test set: test_x only. group = generator folder, plus the class
    folder when there is one (group_basis = class then, else generator). Real and fake images of one
    generator folder share the group.
  * Diffusion: the real folders (imagenet, laion) are their own groups (group_basis = generator, since the
    folder names the source collection); the dataset README pairs the real set with the generators, but the
    pairing is not recorded in the folder names, so it is not assumed here.

Label from the folder name: 0_real -> 0, 1_fake -> 1. method = the generator (progan for ProGAN, the
generator folder for the CNN subset and the folder for diffusion), or ``real``.
"""

from __future__ import annotations

from pathlib import Path

from ..manifest import Record, rel_posix

IMAGE_EXT = {".png", ".jpg", ".jpeg"}
LABEL_DIRS = {"0_real": 0, "1_fake": 1}

EXPECTED_PROGAN = {"images": 8000, "real": 4000, "fake": 4000, "classes": 20, "per_class_label": 200}


def scan_labelled_tree(base: Path) -> list[tuple[tuple[str, ...], int, Path]]:
    """Find every image below base whose parent folder is 0_real or 1_fake.

    Returns (folder parts above the label folder, label, file) sorted by path. The parts are relative to base.
    Images not under a label folder are ignored (and would show as a count mismatch).
    """
    out: list[tuple[tuple[str, ...], int, Path]] = []
    if not base.is_dir():
        return out
    for p in sorted(base.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXT and p.parent.name in LABEL_DIRS:
            parts = p.parent.parent.relative_to(base).parts
            out.append((parts, LABEL_DIRS[p.parent.name], p))
    return out


def _records(
    key: str, family: str, base: Path, data_root: Path, make_group, method_of, notes: str = ""
) -> list[Record]:
    recs = []
    for parts, label, p in scan_labelled_tree(base):
        group_local, basis = make_group(parts)
        rel_in_base = p.relative_to(base).as_posix()
        recs.append(
            Record(
                sample_id=f"{key}:{rel_in_base}",
                dataset=key,
                family=family,
                path=rel_posix(p, data_root),
                modality="image",
                label=label,
                method="real" if label == 0 else method_of(parts),
                group_id=f"{key}:{group_local}",
                group_basis=basis,
                identity_known=False,
                official_split="",
                notes=notes,
            )
        )
    return recs


def _progan(key: str, base: Path, data_root: Path) -> tuple[list[Record], dict]:
    recs = _records(key, "forensynths_progan", base, data_root,
                    make_group=lambda parts: (parts[0], "class"), method_of=lambda parts: "progan")
    per_group: dict[str, dict[str, int]] = {}
    for r in recs:
        g = per_group.setdefault(r.group_id, {"real": 0, "fake": 0})
        g["real" if r.label == 0 else "fake"] += 1
    n_real = sum(r.label == 0 for r in recs)
    report = {
        "dataset": key,
        "rows": len(recs),
        "real": n_real,
        "fake": len(recs) - n_real,
        "classes": len(per_group),
        "per_class_counts_unique": sorted({(v["real"], v["fake"]) for v in per_group.values()}),
        "expected": EXPECTED_PROGAN,
        "complete": len(recs) == 8000 and n_real == 4000 and len(per_group) == 20
        and {(v["real"], v["fake"]) for v in per_group.values()} == {(200, 200)},
    }
    return recs, report


def build_progan_val(data_root: Path) -> tuple[list[Record], dict]:
    return _progan("forensynths_progan_val", data_root / "forensynths" / "progan_val", data_root)


def build_progan_test(data_root: Path) -> tuple[list[Record], dict]:
    base = data_root / "forensynths" / "progan_test"
    if (base / "progan").is_dir():
        base = base / "progan"
    return _progan("forensynths_progan_test", base, data_root)


def build_cnn_subset(data_root: Path) -> tuple[list[Record], dict]:
    key = "forensynths_cnn_subset"
    base = data_root / "forensynths" / "cnn_synth_subset"

    def group(parts):
        return (parts[0], "generator") if len(parts) == 1 else (":".join(parts[:2]), "class")

    recs = _records(key, "forensynths_cnn_synth", base, data_root, make_group=group,
                    method_of=lambda parts: parts[0],
                    notes="test_x only")
    per_gen: dict[str, dict[str, int]] = {}
    for r in recs:
        gen = r.group_id.split(":")[1]
        g = per_gen.setdefault(gen, {"real": 0, "fake": 0})
        g["real" if r.label == 0 else "fake"] += 1
    report = {
        "dataset": key,
        "rows": len(recs),
        "generators": per_gen,
        "groups": len({r.group_id for r in recs}),
        "complete": None,  # no documented per-generator count; the archive lists 56,003 members (DATA_LOG.md)
    }
    return recs, report


def build_diffusion_test(data_root: Path) -> tuple[list[Record], dict]:
    key = "diffusion_test"
    base = data_root / "diffusion_test"
    if (base / "diffusion_datasets").is_dir():
        base = base / "diffusion_datasets"
    recs = _records(key, "diffusion_test", base, data_root,
                    make_group=lambda parts: (parts[0], "generator"), method_of=lambda parts: parts[0],
                    notes="test_x only")
    per_folder: dict[str, dict[str, int]] = {}
    for r in recs:
        g = per_folder.setdefault(r.group_id.split(":")[1], {"real": 0, "fake": 0})
        g["real" if r.label == 0 else "fake"] += 1
    report = {
        "dataset": key,
        "rows": len(recs),
        "folders": per_folder,
        "documented": "1,000 images per folder (DATA_LOG.md, I2)",
        "complete": len(per_folder) == 10 and all(v["real"] + v["fake"] == 1000 for v in per_folder.values()),
    }
    return recs, report
