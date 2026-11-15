"""build_numbers.py reads H2 and the branch metrics into numbers.json keys."""

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_numbers.py"


def _module(tmp_path: Path):
    spec = importlib.util.spec_from_file_location("build_numbers", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT, module.RESULTS = tmp_path, tmp_path / "results"
    return module


def test_h2_keys(tmp_path):
    module = _module(tmp_path)
    out = tmp_path / "results" / "p001-h2"
    out.mkdir(parents=True)
    target = {"estimate": 0.0213, "lower": -0.004, "upper": 0.051, "p_one_sided": 0.061,
              "clips": 2900, "identities": 190, "per_seed": [0.02, 0.025, 0.019]}
    (out / "h2.json").write_text(json.dumps({
        "seeds": [17, 29, 43],
        "targets": {"dfdc": target},
        "holm": {"dfdc": {"p_one_sided": 0.061, "rejected_at_0.05": False}},
        "h2_success": False,
        "weights": {"17": {"branches": ["visual", "audio", "sync"], "coef": [1.2, 0.3, 0.4]}},
    }))
    numbers: dict = {}
    module.h2(numbers)
    assert numbers["h2.dfdc.estimate"]["value"] == 0.021
    assert numbers["h2.dfdc.seed29"]["value"] == 0.025
    assert numbers["h2.dfdc.holm"]["value"] == "no"
    assert numbers["h2.success"]["value"] == "no"
    assert numbers["h2.weights.seed17.sync"]["value"] == 0.4
    assert numbers["h2.dfdc.p"]["source"] == "results/p001-h2/h2.json"


def test_branch_keys_split_branch_seed_and_part(tmp_path):
    module = _module(tmp_path)
    out = tmp_path / "results" / "p001-branches"
    out.mkdir(parents=True)
    report = {"overall": {"metrics": {"roc_auc": 0.91234, "fpr": 0.2, "balanced_accuracy": 0.8}, "total": 500},
              "confidence_intervals": {"roc_auc": {"lower": 0.88, "upper": 0.94}}}
    (out / "visual-seed17-test-identity-strict-metrics.json").write_text(json.dumps(report))
    numbers: dict = {}
    module.branches(numbers)
    assert numbers["visual.test-identity-strict.seed17.auc"]["value"] == 0.912
    assert numbers["visual.test-identity-strict.seed17.n"]["value"] == 500
