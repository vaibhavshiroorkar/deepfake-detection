#!/usr/bin/env bash
# The environment `scripts/check_encoder_ports.py reference` runs in: the
# official AV-HuBERT code on fairseq 0.12.2, and timm 0.9 for HSEmotion's pickle.
#
# CPU only and separate from the project venv. fairseq pins a PyTorch too old
# for this GPU, and installing it into the project would take the CUDA stack
# down (docs/obstacles.md). fairseq is used from a source checkout on
# PYTHONPATH rather than built: building needs a C++ toolchain, and nothing the
# reference calls uses the compiled extensions.
#
# Linux or WSL, with git and uv:
#     bash scripts/setup_encoder_reference.sh ~/avhubert-ref
# then, from the repository root in that shell:
#     . ~/avhubert-ref/.venv/bin/activate
#     python scripts/check_encoder_ports.py reference \
#         --av-hubert ~/avhubert-ref/av_hubert --fairseq ~/avhubert-ref/fairseq-src
set -euo pipefail

root="${1:?usage: setup_encoder_reference.sh <directory>}"
mkdir -p "$root"
cd "$root"

[ -d av_hubert ] || git clone --depth 1 https://github.com/facebookresearch/av_hubert.git
[ -d fairseq-src ] || git clone --depth 1 --branch v0.12.2 \
    https://github.com/facebookresearch/fairseq.git fairseq-src

uv venv --python 3.9 .venv
# shellcheck disable=SC1091
. .venv/bin/activate
uv pip install "torch==2.0.1" --index-url https://download.pytorch.org/whl/cpu
uv pip install "numpy<1.24" "omegaconf==2.0.6" "hydra-core==1.0.7" bitarray \
    sacrebleu regex cffi tqdm python_speech_features scipy \
    opencv-python-headless sentencepiece "timm==0.9.16"

PYTHONPATH="$root/fairseq-src" python -c \
    "import fairseq; print('fairseq', fairseq.__version__, 'from source')"
