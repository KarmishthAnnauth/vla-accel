# orion_ros parity tests

Manual tests that check the ROS node still matches the reference agent
(`Orion/team_code/orion_b2d_agent.py`). Run inside the ORION container, with the
workspace sourced and `/root/Orion/ckpts -> /models/Orion` linked:

```bash
docker exec orion_ros bash -c '
  source /opt/ros/humble/setup.bash
  source /opt/orion_ws/install/setup.bash
  ln -sfn /models/Orion /root/Orion/ckpts
  cd /root/Orion
  ROS_DOMAIN_ID=77 python3 test_cameras.py   # use a spare domain: it publishes
  python3 test_pid_parity.py
  python3 test_build_batch.py
  python3 test_route_reset.py
  python3 test_memory_continuity.py'
```

- `test_cameras.py` — each ORION camera slot gets its OWN image; a snapshot is
  withheld until all six have arrived; skew against CAM_FRONT is reported.
- `test_pid_parity.py` — `_compute_control` equals the agent's `run_step`
  control block over 200 frames, including the 5 m/s throttle cap.
- `test_build_batch.py` — six distinct views survive ORION's real
  `inference_only_pipeline` + collate, with six distinct calibration matrices,
  and the per-route scene_token reaches img_metas. No checkpoint needed.
- `test_route_reset.py` — a new route arms a context reset, a replayed identical
  plan does not, and the reset clears ORION's temporal memory, the PID windows
  and the frame counter.
- `test_memory_continuity.py` — drives ORION's own memory-retention rule with
  the timestamp sequence each `timestamp_mode` produces. Shows that in async
  real-time at >2 s/inference the memory is zeroed on every frame under
  "sensor", kept under "agent", and never dropped in synchronous mode.

Copy them in with `docker cp` (they are not installed by setup.py).
