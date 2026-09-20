#!/usr/bin/env bash
set -eo pipefail

IMAGE=karmishthannauth/simlingo:humble-cyclonedd
NAME=sim_ros

if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "[orin] starting container $NAME"
  docker rm -f "$NAME" 2>/dev/null || true
  docker run -d --runtime nvidia --network host --shm-size=8g \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    -e ROS_DOMAIN_ID=0 \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e CYCLONEDDS_URI=file:///cyclone.xml \
    -v /home/USER/cyclone_zerotier.xml:/cyclone.xml:ro \
    -v /home/USER/vla_benchmarking/benchmarking/:/benchmarking \
    -v /home/USER/models/simlingo_hf:/models/simlingo:ro \
    --name "$NAME" --entrypoint bash "$IMAGE" -c 'sleep infinity' >/dev/null
  sleep 2
fi

exec docker exec -i "$NAME" bash -c '
set -e
source /opt/ros/humble/setup.bash
source /opt/ros_ws/install/setup.bash      # rmw_cyclonedds_cpp
source /opt/sim_ws/install/setup.bash      # carla_msgs, autoware_*, ackermann_msgs, simlingo_ros
export PYTHONPATH="/benchmarking/simlingo:/benchmarking/simlingo/team_code:$PYTHONPATH"
echo "[ctr] domain=$ROS_DOMAIN_ID rmw=$RMW_IMPLEMENTATION"
exec ros2 launch simlingo_ros simlingo.launch.py \
  checkpoint_path:=/models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt \
  simlingo_path:=/benchmarking/simlingo
'
