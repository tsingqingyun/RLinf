#! /usr/bin/bash
# Restart embodied training after failure: halve actor batch sizes in the yaml, then rerun.
# Same environment and CLI as run_embodiment.sh.
#
# Usage:
#   cd examples/embodiment && bash run_embodiment_auto_halve.sh robocasa365_searingmeat_ppo_openpi_pi05
#
# Optional env:
#   WORLD_SIZE / NUM_GPUS  — default 8 (for halve_actor_batch_yaml.py FSDP alignment)
#   MAX_AUTO_RESTARTS     — default 30; set to -1 for unlimited
#
# Notes:
# - Clean exit (code 0) stops the loop (e.g. training finished all epochs).
# - Ctrl+C / SIGTERM exits the wrapper without editing yaml.
# - If the process hangs (no exit), this script cannot help until the process is killed.
# - Each restart writes a new runner.logger.log_path under REPO_PATH/logs/...

set -euo pipefail

export EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"
HALVE_PY="${EMBODIED_PATH}/halve_actor_batch_yaml.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}
export NVIDIA_DRIVER_CAPABILITIES="all"
export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH

export OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}
export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
export ISAAC_PATH=${ISAAC_PATH:-/path/to/isaac-sim}
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}

if [ -z "${1:-}" ]; then
    CONFIG_NAME="maniskill_ppo_openvlaoft"
else
    CONFIG_NAME=$1
fi

ROBOT_PLATFORM=${2:-${ROBOT_PLATFORM:-"LIBERO"}}
export ROBOT_PLATFORM

export LIBERO_TYPE=${LIBERO_TYPE:-"standard"}
if [ "$LIBERO_TYPE" == "pro" ]; then
    export LIBERO_PERTURBATION="all"
    echo "Evaluation Mode: LIBERO-PRO | Perturbation: $LIBERO_PERTURBATION"
elif [ "$LIBERO_TYPE" == "plus" ]; then
    export LIBERO_SUFFIX="all"
    echo "Evaluation Mode: LIBERO-PLUS | Suffix: $LIBERO_SUFFIX"
else
    echo "Evaluation Mode: Standard LIBERO"
fi

echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"
echo "Using Python at $(which python)"

CONFIG_YAML="${EMBODIED_PATH}/config/${CONFIG_NAME}.yaml"
if [[ ! -f "$CONFIG_YAML" ]]; then
    echo "Expected config at $CONFIG_YAML" >&2
    exit 1
fi

MAX_AUTO_RESTARTS=${MAX_AUTO_RESTARTS:-30}
WORLD_SIZE=${WORLD_SIZE:-${NUM_GPUS:-8}}
export WORLD_SIZE

trap 'echo "[auto_halve] interrupted, exiting"; exit 130' INT TERM

restart_count=0
while true; do
    LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
    mkdir -p "${LOG_DIR}"
    MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
    CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
    echo "${CMD}" >"${MEGA_LOG_FILE}"
    echo "[auto_halve] start (restart_count=${restart_count}) log=${LOG_DIR}"

    set +e
    set -o pipefail
    ${CMD} 2>&1 | tee -a "${MEGA_LOG_FILE}"
    ec=${PIPESTATUS[0]}
    set -e

    if [[ "$ec" -eq 0 ]]; then
        echo "[auto_halve] training exited 0 — stopping wrapper."
        break
    fi
    if [[ "$ec" -eq 130 ]] || [[ "$ec" -eq 143 ]]; then
        echo "[auto_halve] signal exit $ec — stopping wrapper."
        break
    fi

    echo "[auto_halve] training failed with exit code $ec"
    if [[ "$MAX_AUTO_RESTARTS" -ge 0 ]] && [[ "$restart_count" -ge "$MAX_AUTO_RESTARTS" ]]; then
        echo "[auto_halve] MAX_AUTO_RESTARTS=$MAX_AUTO_RESTARTS reached — abort."
        exit "$ec"
    fi

    echo "[auto_halve] halving actor batch sizes in ${CONFIG_YAML} ..."
    if ! WORLD_SIZE="$WORLD_SIZE" python3 "$HALVE_PY" "$CONFIG_YAML"; then
        echo "[auto_halve] halve script failed — abort."
        exit 1
    fi

    restart_count=$((restart_count + 1))
    sleep 5
done
