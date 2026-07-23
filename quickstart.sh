#!/usr/bin/env bash
# One-command setup and proof: creates a venv, fetches Galbot's SDK examples
# (sparse) and the public S1 model, then runs the full 10-tutorial suite
# against this backend. Takes about 5 minutes on a clean machine.
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
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

GALBOTSIM_PYTHON="$PWD/.venv/bin/python" \
GALBOT_SDK_EXAMPLES="$PWD/third_party/GalbotSDK/examples/g1/python/tutorials" \
GALBOTSIM_MODEL_XML="$PWD/third_party/galbot_s1_description/mjcf/galbot_s1_v1_1_0.xml" \
bash tests/run_tutorials.sh
