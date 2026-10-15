# daimon_tj_fx_description

This package stores the TJ_FX dual-arm URDF assets.

## Contents

- `urdf/raw/left_arm_raw.urdf`: original left arm URDF exported from CAD
- `urdf/raw/right_arm_raw.urdf`: original right arm URDF exported from CAD
- `urdf/left_arm.urdf`: left arm URDF with package-local mesh paths
- `urdf/right_arm.urdf`: right arm URDF with package-local mesh paths
- `urdf/daimon_tj_fx_dual_arm.urdf`: assembled dual-arm URDF with a center box torso

## Layout assumptions

- The center torso is modeled as a `0.10 x 0.10 x 1.50 m` box.
- `base_link` is placed at the midpoint between the two arm bases.
- The left and right arm mount points are at `(0, -0.15, 0)` and `(0, 0.15, 0)` relative to `base_link`.
- Both arms are mounted horizontally in a T-pose-like layout: the left arm points toward `+y` and the right arm points toward `-y` when all arm joints are zero.
- ROS frame convention is `x` forward, `y` left, `z` up.

## Preview (ROS 2)

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select daimon_tj_fx_description
source install/setup.bash
ros2 launch daimon_tj_fx_description display.launch.py
```

By default this starts `robot_state_publisher` and RViz, plus a one-shot zero-valued `/joint_states` seed so the full arm is visible even when the normal `joint_state_publisher` stays off.
The default seed delay is `2.0` seconds.
If you want to drive the model from your own `/joint_states` publisher later, keep `use_joint_state_publisher:=false`; the one-shot seed publishes once at startup and then exits.

To change the `joint_states` publish rate, pass `joint_state_rate` at launch time:

```bash
roslaunch daimon_tj_fx_description display.launch joint_state_rate:=30
```

```bash
roslaunch daimon_tj_fx_description display.launch use_joint_state_publisher:=false
```

```bash
roslaunch daimon_tj_fx_description display.launch use_initial_joint_state_seed:=false
```

## ROS2 legacy note

The original ROS2 launch file is kept at `launch/display.launch.py` for reference, but this package is now configured for ROS1 catkin builds in this workspace.
