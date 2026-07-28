#!/usr/bin/env bash
# One-command setup and proof: creates a venv, fetches Galbot's SDK examples
# (sparse) and the public S1 model, then runs the full 10-tutorial suite
# against this backend. Takes about 5 minutes on a clean machine.
set -euo pipefail
cd "$(dirname "$0")"

# open3d (needed by tutorial example4) ships wheels for Python 3.10-3.12 only;
# prefer an interpreter in that range over a newer default python3.
PY=""
for cand in python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
        if "$cand" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)'; then
            PY="$cand"
            break
        fi
    fi
done
if [ -z "$PY" ]; then
    echo "No Python 3.10-3.12 found on PATH (needed for the open3d wheel used by example4)." >&2
    exit 1
fi
echo "Using $PY ($($PY --version 2>&1))"

"$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q mujoco numpy opencv-python-headless open3d

mkdir -p third_party
if [ ! -d third_party/GalbotSDK ]; then
    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/GalaxyGeneralRobotics/GalbotSDK third_party/GalbotSDK
    git -C third_party/GalbotSDK sparse-checkout set examples/g1/python/tutorials
fi
if [ ! -d third_party/galbot_s1_description ]; then
    git clone --depth 1 \
        https://github.com/GalaxyGeneralRobotics/galbot_s1_description \
        third_party/galbot_s1_description
fi

# GALBOTSIM_MODEL_XML is left unset on purpose: the MJCF filename is discovered
# from third_party/galbot_s1_description/mjcf/, so an upstream rename does not
# break the quickstart. Set it explicitly to point at a different model.
GALBOTSIM_PYTHON="$PWD/.venv/bin/python" \
GALBOT_SDK_EXAMPLES="$PWD/third_party/GalbotSDK/examples/g1/python/tutorials" \
bash tests/run_tutorials.sh
