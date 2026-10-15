# DM25PD04-EG Tianji gripper description

ROS 2 description package generated from the 2026-08-07 CAD export.

## Build

```bash
cd <your-ros2-workspace>
source /opt/ros/lyrical/setup.bash
colcon build --packages-select dm25pd04_eg_urdf_tianji_20260807
source install/setup.bash
```

## Display in RViz2

```bash
ros2 launch dm25pd04_eg_urdf_tianji_20260807 display.launch.py
```

RViz2 can be disabled for a headless smoke test:

```bash
ros2 launch dm25pd04_eg_urdf_tianji_20260807 display.launch.py use_rviz:=false
```

The original SolidWorks-exported ROS 1 launch files are retained as source
references. Use `display.launch.py` with ROS 2.
