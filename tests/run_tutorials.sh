#!/usr/bin/env bash
# run_tutorials.sh — run all 10 official G1 tutorials against GalbotSim.
#
# Environment variables (all optional):
#   GALBOTSIM_ROOT        Path to galbot-sim repo root
#   GALBOT_SDK_EXAMPLES   Path to tutorials directory
#   GALBOTSIM_PYTHON      Python interpreter to use

set -uo pipefail

GALBOTSIM_ROOT="${GALBOTSIM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
GALBOT_SDK_EXAMPLES="${GALBOT_SDK_EXAMPLES:-$GALBOTSIM_ROOT/third_party/GalbotSDK/examples/g1/python/tutorials}"
GALBOTSIM_PYTHON="${GALBOTSIM_PYTHON:-python3}"
TIMEOUT_SECS="${TIMEOUT_SECS:-90}"

# Portable timeout: prefer `timeout`, fall back to `gtimeout`, else perl alarm
if command -v timeout &>/dev/null; then
    TIMEOUT_CMD="timeout"
elif command -v gtimeout &>/dev/null; then
    TIMEOUT_CMD="gtimeout"
else
    timeout() { perl -e "alarm $1; exec @ARGV[1..$#ARGV]" "$@"; }
    TIMEOUT_CMD="timeout"
fi

TUTORIALS=(
    example1_galbot_control_started.py
    example2_arm_manipulation.py
    example3_robot_navigation.py
    example4_sensor_data_collect.py
    example5_pick_and_place.py
    example6_execute_vla.py
    example7_arm_manipulation2.py
    example8_real_time_control_loop.py
    example9_multiple_waypoints_planning.py
    example10_final_task.py
)

passed=0
failed=0
failed_names=()

echo "=== GalbotSim tutorial runner ==="
echo "Root:      $GALBOTSIM_ROOT"
echo "Examples:  $GALBOT_SDK_EXAMPLES"
echo "Python:    $GALBOTSIM_PYTHON"
echo ""

for tutorial in "${TUTORIALS[@]}"; do
    script="$GALBOT_SDK_EXAMPLES/$tutorial"
    if [[ ! -f "$script" ]]; then
        echo "SKIP  $tutorial (file not found)"
        continue
    fi

    printf "%-50s " "$tutorial"
    if echo y | $TIMEOUT_CMD "$TIMEOUT_SECS" env PYTHONPATH="$GALBOTSIM_ROOT" "$GALBOTSIM_PYTHON" "$script" > /dev/null 2>&1; then
        echo "PASS"
        passed=$((passed + 1))
    else
        echo "FAIL"
        failed=$((failed + 1))
        failed_names+=("$tutorial")
    fi
done

echo ""
echo "=== Results: $passed passed, $failed failed out of ${#TUTORIALS[@]} ==="

if [[ $failed -gt 0 ]]; then
    echo "Failed tutorials:"
    for name in "${failed_names[@]}"; do
        echo "  - $name"
    done
    exit 1
fi
