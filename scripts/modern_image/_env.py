"""Shared paths for the modern-image research scripts (W1-I). pyarrow lives in a separate target folder, not in the product venv."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
PYLIB = ROOT / "data" / "cf_eval" / "_pylib"
if str(PYLIB) not in sys.path:
    sys.path.insert(0, str(PYLIB))
EVAL_REPO, EVAL_REV = "OwensLab/CommunityForensics-Eval", "7d4a74a88d2cac93b513c0853bf92c260eaceea0"
SMALL_REPO, SMALL_REV = "OwensLab/CommunityForensics-Small", "6c539a534c07917307c381f5af4053c6091b5278"
CF_EVAL, CF_SMALL = ROOT / "data" / "cf_eval", ROOT / "data" / "cf_small"
RES = ROOT / "data" / "_results" / "modern"
META_COLS = ["image_name", "format", "resolution", "mode", "model_name", "nsfw_flag", "real_source", "subset", "split", "label", "architecture"]
