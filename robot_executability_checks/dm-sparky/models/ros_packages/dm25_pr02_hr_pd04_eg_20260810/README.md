# DM25 PR02 mobile robot with DM25PD04-EG grippers

This package uses `dm25_pr02_hr` as its parent robot. It keeps the mobile
chassis, wheel joints, movable knee joint, torso, head, and both seven-axis
arms from that model.

Both legacy grippers are replaced by the gripper exported in
`DM25PD04_EG_URDF_tianji_20260807`:

- `arm_left_ee_link` and `arm_right_ee_link` remain the arm kinematic tips.
- The source gripper's `Link2` flange adapter and `Link2_joint` are omitted.
- The body, camera, two fingers, and two pads are instantiated on each arm.
- `left_endeffector_joint` and `right_endeffector_joint` remain the commanded
  names expected by the motion system.
- Each opposing finger is represented by the existing `*_gripper_state` mimic
  joint. Its opposite axis and `+1` multiplier produce symmetric closure.
- The CAD maximum-open pose is `q=0`; pad contact is `q=0.385773 rad` per
  finger. Normalized hardware state is converted for TF/RViz.

`config/self_collision_model.yaml` contains the always-selected sparse WBC
self-collision model. It checks both arms and full-stroke PD04-EG envelopes
against the torso/head, each gripper against the movable thigh, and the two
gripper envelopes against each other. Chassis and wheel geometry are not part
of this fixed-bottom WBC collision model.

The mount transforms account for the former Link2 axis offset
`x=-0.010346 m` and mirror the right-side installation.

Validate with:

```bash
check_urdf urdf/dm25_pr02_hr_pd04_eg_20260810.urdf
```


检查结果：TCP v2 与当前 URDF 一致，前提是这里的 TCP 指的是 gripper_base_link 坐标系。

| 变换 | TCP v2 | 当前 URDF `arm_7_link → gripper_base_link` | 误差 |
|---|---:|---:|---:|
| 左侧 Y | `+0.010350 m` | `+0.010346 m` | `4 μm` |
| 右侧 Y | `-0.010350 m` | `-0.010346 m` | `4 μm` |
| 两侧 Z | `-0.129710 m` | `-0.129711803 m` | `1.803 μm` |
| 旋转 | 左右镜像 ±90° | 完全一致 | 可忽略 |


旋转矩阵也完全对应：
左侧：相机/Link3 朝左外侧
右侧：相机/Link3 朝右外侧
两侧夹爪 Z 轴都沿机械臂末端向外
