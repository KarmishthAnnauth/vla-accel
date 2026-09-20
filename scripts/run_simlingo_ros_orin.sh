#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
source /home/USER/vla_benchmarking/benchmarking/alpamayo-autoware/install/setup.bash

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/USER/cyclone_zerotier.xml
export ROS_DOMAIN_ID=0

SIMLINGO_PATH=/home/USER/vla_benchmarking/benchmarking/simlingo
CKPT=/home/USER/models/simlingo_hf/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt

export PYTHONPATH="${SIMLINGO_PATH}:${SIMLINGO_PATH}/team_code:${PYTHONPATH:-}"

echo "[orin] domain=$ROS_DOMAIN_ID rmw=$RMW_IMPLEMENTATION"
echo "[orin] ckpt=$CKPT"
echo "[orin] simlingo=$SIMLINGO_PATH"
exec ros2 launch simlingo_ros simlingo.launch.py \
  checkpoint_path:="$CKPT" \
  simlingo_path:="$SIMLINGO_PATH"
