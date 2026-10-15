# DM-Sparky LeRobot Evaluation and Replay

## 目录与使用方式

本仓库将三个机器人版本放在同一 `main` 分支的 `robot_executability_checks/` 下，按机器人分成独立目录，无需切换 Git 分支。

| 目录 | 功能 |
| --- | --- |
| [tianji](../tianji/) | Tianji：Aik7 轨迹回放、离线可达性/可操作性评估、可达空间与视频可视化 |
| [dm-sparky](../dm-sparky/) | DM-Sparky：固定躯干评估，以及双臂、底盘、膝/腰的 18 DoF 联合求解 |
| [realman](../realman/) | RealMan RM75-B：双臂模型、DM-TR01 支架、UMI 夹爪、专用评估与可视化 |

**本页命令均在当前机器人目录内运行。** 每个目录单独保留 `config/`、`models/`、`src/`、`scripts/` 和 `.venv/`。切换机器人时先 `deactivate`，再进入对应目录并激活其环境；不要将三个 `src/` 同时加入 `PYTHONPATH`。源版本与迁移说明见 [CODE_IMPORT.md](../../docs/CODE_IMPORT.md)。

本目录保留 DM-Sparky 的固定躯干评估、18 自由度联合求解和可视化，以及原有天机兼容功能。**`evaluate.sh` 的兼容默认值仍为 Tianji；评估 DM-Sparky 时必须显式指定 `--robot mobile-humanoid`：**

```bash
# 完成下文安装后，在 dm-sparky/ 内运行
./evaluate.sh /path/to/dataset outputs/dm_sparky_quality --robot mobile-humanoid
```

底盘、膝和腰参与联合求解时，使用下文的 `scripts/evaluate_humanoid_mobility.py`。

这是一个自包含的天机双臂 LeRobot v3 工具仓库，现在同时提供：

- **MuJoCo 轨迹回放**：用天机 ROS `/start_wbc` 路径的 Aik7 解析 IK 驱动机器人，可视化左右 UMI TCP 目标和实际末端。
- **离线可操纵性验证**：从天机整机 URDF 重建左右 7 轴运动链，对每帧 `gripper_base_link` TCP 求解，输出可达性、关节限位余量、Jacobian 奇异值、条件数、isotropy 和 manipulability。
- **跨构型无碰撞可达空间**：对 Tianji、DM-SPARKY（full-body）的左右七轴手臂分别做 Sobol 采样、碰撞过滤、共享体素化和 manipulability 热图显示。

同事原来的移动底盘人形评估仍保留为可选模式，但评估命令默认已改为天机。

## 使用前提：轨迹必须统一到本项目的标定板坐标系

**本项目的轨迹回放、可达性和可操作性评估，要求输入的左右 TCP 位姿已经统一到此前确认的标定板坐标系。LeRobot v3 只是数据格式，不代表坐标系满足要求。** 此限制对所有支持的机器人构型均适用。

- **世界原点**：位于此前的斜立标定板板面宽度中央，距上边 14.6 cm、距下边 15.6 cm；不是任意 AprilTag 的中心，也不是 Ego 首帧位置。
- **世界轴向**：`+X` 沿板面宽度向右，`+Y` 沿板面长边向下，`+Z` 朝板内。原点与轴向必须同时一致。
- **TCP 定义**：左右位姿均表示夹爪 URDF 的 `base_link`（天机对应 `gripper_base_link_L/R`），不是法兰、相机或 IMU 位姿。位置单位为米，四元数顺序为 `x, y, z, w`。手持夹爪水平朝前时，其局部轴为 `+X` 向上、`+Y` 向右、`+Z` 向前。
- **场景配准**：所选 `--scene-config` 必须正确描述该标定板参考系、桌子和机器人基座之间的关系；仅将原点设在某块标定板上仍不够。

`head_world_frame`、独立的 `slam_world_frame`、墙面其他 AprilGrid 或任意未配准的世界系，**不能直接使用当前场景配置回放或评估**。其他来源的轨迹须先通过已知的完整刚体变换（位置和姿态一起变换）转换到上述板系；不得靠猜测轴向、左右补偿或高度平移配准。即使报告写着 `apriltag_frame` 或 `calibrated`，也必须确认物理标定板、原点、轴向及场景配准一致。

当前普通回放／评估入口不会自动识别或转换这些不同世界系。使用者必须先核对本节前提；不满足时应停止机器人指标计算，而不是把配准错误解释成不可达或低可操作性。输入检查方法见下文“已固定的位姿约定”。

## 一键评估 LeRobot v3 数据

以下命令仅适用于满足上述标定板坐标系前提的数据。

```bash
./evaluate.sh /path/to/lerobot_v3_dataset
```

这会默认使用仓库内的天机 URDF、`config/tianji_joints_params.xml` 和 `config/default_scene.json`。首次运行会创建 `.venv`并安装依赖。第二个参数可指定输出目录，其余参数原样传给评估程序：

```bash
./evaluate.sh /path/to/dataset /path/to/output --episodes 0,2-4
```

全量 episode 可使用多进程（例如 8 个 worker）：

```bash
./evaluate.sh /path/to/dataset /path/to/output --episodes all --jobs 8
```

每次评估同时给出两套互相独立的结论：

1. `independent`：每帧不继承前后状态，从关节中点及两个确定性补充种子求解，回答单帧理论可达性。
2. `continuous_midpoint`：先用上一帧解作为连续种子，失败后从关节中点复求，回答连续轨迹求解结果。

天机模式的运动关节是 `Joint1_L…Joint7_L` 和 `Joint1_R…Joint7_R`，目标 link 是 `gripper_base_link_L/R`（数据中的 TCP），不是 `Link7` 法兰或 `ee_link`。有效关节范围取 URDF 限位和天机 ROS 运行时限位的交集。每臂雅可比是 `base_link` 下的 `6×7` TCP 几何 Jacobian。输出目录包含：

- `summary.json`：两套 PASS/FAIL、阈值、模型和范围声明。
- `report.md`：两套方案的中文并列结论。
- `independent_frames.csv`、`continuous_midpoint_frames.csv`：逐帧 IK、关节裕量、奇异值、条件数、isotropy 和 manipulability。
- `quality_dashboard.html`：可离线打开的可视化仪表板，可切换求解策略和 episode，查看逐帧左右臂可达性、isotropy、关节裕量和 TCP 轨迹点。三维轨迹可拖拽旋转、滚轮缩放、双击恢复视角，并保留 X-Y / X-Z 精确投影；叉号表示不可达，前 10 帧用橙色圈标出，第 0 帧另用黑圈强调。

评估完成后可直接打开仪表板：

```bash
xdg-open outputs/tianji_arm_quality/quality_dashboard.html
```

## 人形底盘与腰部自适应上限

普通 `--robot mobile-humanoid` 评估会固定底盘以及膝、腰关节。若要评估轮式
人形前后移动并使用膝、腰俯仰和腰偏航后的运动学上限，运行：

```bash
python scripts/evaluate_humanoid_mobility.py \
  --dataset "$LEROBOT_DATASET" \
  --episodes all \
  --base-y-min -0.50 \
  --base-y-max -0.05 \
  --jobs 8 \
  --output outputs/humanoid_adaptive_mobility
```

这里的 `base-y` 是底盘 `base_link` 在桌子坐标系中的前后位置。求解器把底盘
前后平移、`knee_joint`、`hip_pitch_joint`、`hip_yaw_joint` 和左右 14 个手臂
关节放进同一个双臂位姿问题，并在 episode 内继承上一帧解。输出会记录每帧选到
的底盘位置、膝腰角度、仅手臂 Jacobian 指标和包含移动自由度的全身 Jacobian
指标。

该模式允许底盘和腰部逐帧调整，但尚未限制其速度/加速度，也不检查碰撞或动力学，
因此结果必须标记为“运动学理论上限”，不能直接作为真机 replay 门禁。

评估明确不检查自碰撞、环境碰撞或速度超限；不可达帧的可操作性写为 `null/N/A`，不会在错误关节解上计算雅可比结论。

如需继续评估移动人形，显式指定：

```bash
./evaluate.sh /path/to/dataset /path/to/output --robot mobile-humanoid
```

该模式使用 `config/mobile_humanoid_scene.json` 和 `config/dm25_joints_params.xml`，并固定手臂链上的非手臂关节。

MuJoCo 回放窗口用 Aik7 显示机器人和 TCP 目标；HTML 仪表板显示离线数值 IK 的可达性与可操纵性。两者共用同一份场景坐标配准和天机 TCP 定义，但 IK 实现相互独立，便于交叉核对。

## 已固定的位姿约定

**跨数据集注意：以下是旧批次确认过的约定，不代表所有 LeRobot v3 数据。**
`calibration_status=calibrated` 不足以判断世界坐标系。新批次需要核对源 episode 的
`pose_data/trajectory_quality_report.json` 中 `coordinate_frame`：
`apriltag_frame` 还需确认标签原点、朝向及其场景位置与配置一致；
`head_world_frame` 是该 episode 首帧头部世界系，需要额外的刚体配准到标定板/桌面；
`slam_world_frame` 不保证左右手已处于同一全局系，不能直接按当前标定板配置评估。
当前默认回放/评估仍使用旧板系配置，不能自动推断新数据的配准。

先运行无猜测的诊断，可生成原始 TCP 轨迹/局部轴截图、本目录两种机器人的
URDF 与 MuJoCo FK/Jacobian 一致性报告：

```bash
python scripts/validate_dataset_transfer.py \
  --dataset /path/to/dataset \
  --output outputs/dataset_transfer_check
```

未确认世界系时不输出机器人可达性结论。只有确认与旧板系及 TCP 约定一致时，
才追加 `--board-frame-confirmed`，对首/中/末 episode 各抽样 12 帧做独立 IK
诊断。该结果不是全轨迹可达率，也不验证连续性、碰撞或真实场景配准。
录制新结果应显式指定本次的 `--quality-csv` 和 `--dataset`；录制脚本会检查
帧数、帧序号、时间戳和 TCP 目标位置，拒绝串用不同数据或不同场景的 CSV。

当前默认配置对应已经确认的 UMI 采集约定：

- 数据中的左右 pose 是 UMI 夹爪 URDF 的 `base_link`，即 TCP。
- pose 四元数顺序为 `qx, qy, qz, qw`。
- 世界坐标原点位于标定板板面宽度中央，距上边 14.6 cm、距下边 15.6 cm；因此它比板面几何中心向上 5 mm。
- 世界 `+X` 向右（板面 21 cm 宽方向、桌面宽方向且远离机械臂），`+Y` 沿板面长边向下，`+Z` 朝板内。
- 两侧 UMI TCP 都直接对齐到天机 `gripper_base_link`；没有左/右手隐藏旋转补偿，也不会使用 IMU 外参。

默认的白桌尺寸是长 1.40 m、宽 0.70 m、高 0.80 m。标定板的实测面板为 `21.0 × 30.2 cm`，支撑侧面三边为 `8.0 / 28.5 / 30.2 cm`。代码按三边余弦律保留这个小钝角：支撑夹角为 `94.5016°`，板面相对竖直的倾角为 `19.8141°`。标定板、桌子和天机底座的相对摆位全部明确写在 [config/default_scene.json](config/default_scene.json) 中；它是可替换的仿真场景配置，并不是任意 LeRobot 数据集自带的属性。

## 安装

项目自带完整的天机和夹爪模型资产。克隆后可在任意目录运行，不依赖开发者机器上的绝对路径：

```bash
git clone https://github.com/daimon-robotics/TacClaw-Infinity.git
cd TacClaw-Infinity/robot_executability_checks/dm-sparky
./setup.sh
```

`setup.sh` 会在 `dm-sparky/` 内创建 `.venv`。随后可激活它：

```bash
source .venv/bin/activate
```

依赖包括 MuJoCo、NumPy、PyArrow，以及编译 Aik7 后端需要的 CMake、C++17 编译器和 Eigen3 开发头文件。Debian/Ubuntu 如缺 Eigen3，可安装 `libeigen3-dev` 后重新运行 `./setup.sh`。也可以使用已经激活、且具备这些依赖的现有 Python 环境；下面的文档统一以 `python` 表示当前选定的解释器。

可先确认环境：

```bash
python -c "import mujoco, numpy, pyarrow, scipy; print(mujoco.__version__)"
```

## 天机与人形的无碰撞可达空间

正式计算对每只手臂使用固定种子的 scrambled Sobol 序列，采样数为
`2^20 = 1,048,576`，默认共享体素边长为 4 cm。每个样本都通过 MuJoCo
`mj_fwdPosition` 更新位置与碰撞；采样臂的自碰撞、躯干/底座碰撞和静态桌面环境
碰撞会被剔除。另一条手臂保持 `qpos0`，与其碰撞只单独计数，不会从 individual
workspace 中删除样本。因此 overlap 的严格含义是“左右 individual collision-free
workspace 的空间交集”，不是同时满足双臂碰撞约束的 14-DoF workspace。

依次计算本目录的两种构型：

```bash
for robot in tianji humanoid; do
  python scripts/compute_workspace.py \
    --robot "$robot" \
    --samples 1048576 \
    --voxel-size 0.04 \
    --jobs 8 \
    --output ".workspace_cache/$robot/workspace.npz"
done
```

该命令不修改原 URDF/MJCF。它在 `.workspace_cache/<robot>/` 中生成碰撞可用的
派生模型、`workspace.npz`、JSON 元数据、左右/重叠区 OBJ 和包络版派生 MJCF。
RealMan 的模型、装配脚本、专用场景与评估入口在相邻的 [realman/](../realman/) 目录提供。

缓存包含左右 raw occupancy、仅供显示的 closing occupancy、每体素样本数、左右
最大 \(\eta\) 和 overlap dexterity score。这里
`η = σ_min([Jp; 0.30 Jr]) / σ_max([Jp; 0.30 Jr])`，只取当前手臂七个
DoF 的 Jacobian 列。所有体积和 coverage 统计始终使用 raw occupancy；1-voxel
closing 只用于生成更连续的视觉外壳。

### 交互查看

本目录两种构型均可用同一个被动查看器显示已经求解的 episode 轨迹：

```bash
python scripts/view_workspace.py --robot tianji --episode 0 --paused
python scripts/view_workspace.py --robot humanoid --episode 0 --paused
```

默认读取各构型已生成的逐帧 quality CSV；也可用 `--quality-csv` 和
`--workspace-cache` 指定其他结果。窗口按键：

天机原有的 LeRobot v3/Aik7 replay 也可以原样启用同一叠加层，不会改变轨迹、
相机、时间推进或 episode 切换：

```bash
python scripts/replay_lerobot.py \
  --dataset "$LEROBOT_DATASET" \
  --episode 0 \
  --workspace-cache .workspace_cache/tianji/workspace.npz \
  --paused
```

不传 `--workspace-cache` 时，原 replay 行为与按键保持不变。

- `Space`：播放/暂停；`R`：回到首帧。
- `W`：`OFF → Envelope → Dexterity heatmap → OFF`。
- `O`：左右+重叠区与仅重叠区之间切换。

Mode A 是附着在各构型 reference body 上的半透明左右/重叠包络；Mode B 只画体素
外壳，并用统一的 p1–p99 robust 区间按 viridis 映射 \(\eta\)。当
`viewer.user_scn.maxgeom` 不足时，程序会优先保留 overlap 再做确定性的空间子采样，
并明确打印实际显示数量；缓存与数值统计不会被降采样。

### 视频、截图与验证报告

```bash
for robot in tianji humanoid; do
  python scripts/render_workspace_visualization.py \
    --robot "$robot" \
    --output-dir "outputs/collision_free_workspace/$robot"
done

python scripts/validate_workspace.py \
  --output outputs/collision_free_workspace/validation_report.md
```

每种构型会输出一个 1280×720 presentation-style workspace 视频：默认按源时间戳
逐帧播放完整轨迹，同时让相机缓慢环绕蓝/橙色左右包络和紫色交集；英文图例会
明确说明三种区域的定义，并显示当前帧、总帧数和完整轨迹时间。
同时输出 front-oblique、side、top-oblique 三个视角的 Envelope 和 Heatmap 图片。
验证报告包含 raw/tolerance-aware
轨迹覆盖率、左右与交集体积、最大半径、\(\eta\) 分位数、碰撞剔除明细、
`2^18`/`2^20` 与 4/5 cm 体素敏感性，以及 1000 帧叠加更新基准。无桌面
GLX 上下文时无法可靠测 `viewer.sync()`；报告会把它明确记为不可用，而不会用
无窗口渲染数值冒充交互显示 FPS。

### 叠加录制场景的双目点云

点云版视频使用外部 `stereo_depth` 项目生成米制深度。该项目和深度模型权重未包含在本仓库中；请使用自己的安装路径设置 `STEREO_DEPTH_ROOT`。预处理程序将 LeRobot 中的
左目 RGB 帧与原始 Ego 左目视频做多帧视觉匹配，以确定原始视频的截取起点；这避免了
仅由设备元数据或头部轨迹推测起点所造成的秒级时间误差。截取后的每个 30 Hz 图像帧
直接配对同一 LeRobot 帧的 `head_*` 位姿，使点云与左右 TCP 轨迹使用完全相同的时间轴。
正确坐标链为
`rectified depth camera -> raw RGB camera -> Ego IMU/head -> calibration-board world -> table -> robot base`。
其中 RGB camera 与 IMU 之间使用原始 `calibration.json` 的完整旋转和平移，双目校正
旋转来自 `run_report.json`。原始 `sensor.mcap` 的 `head_pose` 仅用于校验视觉匹配后的
时间与位姿一致性，不再作为视频起点的唯一依据。RGB 颜色不是按相同像素索引直接拷贝：程序将每个深度点变换到
原始 RGB 相机并通过 `meta/info.json` 中的鱼眼内参与畸变参数重新投影，再以有效掩码
剔除视野外点。各机器人视频共享同一段完成配准的录制场景点云。

```bash
export LEROBOT_DATASET=/path/to/your_lerobot_v3_dataset
export STEREO_DEPTH_ROOT=/absolute/path/to/stereo_depth

# 从原始 SXR Ego 录制中截取与 LeRobot episode 0 对齐的双目片段。
# 脚本把 LeRobot 左目图像序列直接匹配到原始视频，并保留 4656x1748 双目画面和标定。
# 若多帧匹配误差超过阈值，脚本会拒绝把不同采集批次的点云与轨迹混合。
.venv/bin/python scripts/prepare_raw_ego_stereo.py \
  --raw-episode /path/to/raw/episode_20260818_0001 \
  --dataset "$LEROBOT_DATASET" \
  --episode 0 \
  --output-dir outputs/pointcloud/episode_000_raw_ego/stereo_input

# 使用 stereo_depth 的 Python/CUDA 环境运行 Fast-FoundationStereo。
python "$STEREO_DEPTH_ROOT/run_fast_foundationstereo_video.py" \
  --episode outputs/pointcloud/episode_000_raw_ego/stereo_input \
  --out-dir outputs/pointcloud/episode_000_raw_ego/depth_30hz \
  --stream rgb --layout horizontal \
  --extrinsics-convention cam_from_imu \
  --frame-step 1 --batch-size 1 --scale 0.25 \
  --valid-iters 4 --max-disp 192 --max-depth 3.0 \
  --save-depth-npy --save-left-rgb --save-registered-rgb \
  --no-preview-video --no-depth-video \
  --model-dir weights/23-36-37/model_best_bp2_serialize.pth \
  --ffs-root "$STEREO_DEPTH_ROOT/external/Fast-FoundationStereo"

for robot in tianji humanoid; do
  MUJOCO_GL=egl .venv/bin/python scripts/render_workspace_visualization.py \
    --robot "$robot" --episode 0 \
    --dataset "$LEROBOT_DATASET" \
    --pointcloud-run outputs/pointcloud/episode_000_raw_ego/depth_30hz \
    --pointcloud-max-points 5000 --pointcloud-voxel-size 0.008 \
    --pointcloud-point-radius 0.0035 \
    --pointcloud-color-saturation 1.8 --pointcloud-color-gamma 0.75 \
    --pointcloud-color-priority 0.25 \
    --pointcloud-min-luminance 0.08 \
    --pointcloud-workcell-margin 0.03 \
    --pointcloud-max-height-above-table 0.50 \
    --pointcloud-stabilize-static \
    --output-dir "outputs/collision_free_workspace_pointcloud_raw_ego/$robot"
done
```

最终视频以 30 FPS 播放全部轨迹帧；上例为每个源帧重建一次点云，因此动态手部、
夹爪和机器人轨迹共享 30 Hz 时间采样，不会因低频点云保持而产生额外滞后。
若只需快速预览，可改用 `--frame-step 3` 得到 10 Hz 点云。`--scale 0.25` 只降低深度推理分辨率以控制显存，
原始图像、相机内外参和颜色注册仍来自 2328x1748 单目原始 Ego 数据，而不是
LeRobot 的 224x224 训练视频。示例使用 8 mm 体素并显示最多 5000 个 RGB 点，
以便在可达空间包络之外保留更清晰的桌面与场景结构。由于采集棚主体是黑色、
桌面是白色，示例还对显示颜色使用 1.8 倍饱和度和 0.75 gamma，并为高色度点
预留 25\% 的点预算；这只改变 MuJoCo 展示，不修改已配准 RGB 或深度文件。
`--pointcloud-stabilize-static` 将多帧中检测到的中性白色桌面和标定板点融合到
标定板世界系，并投影回其已知平面；这些固定表面不再随每帧独立深度估计闪烁，
手、夹爪和任务物体等非静态点仍按当前帧更新。
默认示例保留任务桌范围内的场景点。若要制作只强调手部/TCP 对齐的近景诊断版，
可额外使用 `--pointcloud-hide-table-surface --pointcloud-tcp-radius 0.32`；不要将这种
诊断裁剪用于需要展示完整任务上下文的最终视频。

如果没有与轨迹严格同源的原始录制，不要降低
`--max-visual-alignment-mse` 来强行通过。可先直接使用 LeRobot 中同一 episode 的
左右 Ego 视频；它只有训练分辨率，但帧序号与 TCP 天然一致：

```bash
.venv/bin/python scripts/prepare_lerobot_stereo.py \
  --dataset "$LEROBOT_DATASET" --episode 0 \
  --output-dir outputs/pointcloud/episode_000/stereo_input --overwrite

python "$STEREO_DEPTH_ROOT/run_fast_foundationstereo_video.py" \
  --episode outputs/pointcloud/episode_000/stereo_input \
  --out-dir outputs/pointcloud/episode_000/depth_30hz \
  --stream rgb --layout horizontal --extrinsics-convention cam_from_imu \
  --frame-step 1 --batch-size 1 --scale 1.0 \
  --valid-iters 4 --max-disp 192 --max-depth 3.0 \
  --save-depth-npy --save-left-rgb --save-registered-rgb \
  --no-preview-video --no-depth-video \
  --model-dir weights/23-36-37/model_best_bp2_serialize.pth \
  --ffs-root "$STEREO_DEPTH_ROOT/external/Fast-FoundationStereo"
```

## 回放当前数据集

```bash
# 从 TacClaw-Infinity 仓库根目录进入；若已在 dm-sparky/ 内则无需重复
cd robot_executability_checks/dm-sparky

# 选择本机 LeRobot v3 数据集根目录
export LEROBOT_DATASET=/path/to/your_lerobot_v3_dataset

python scripts/replay_lerobot.py \
  --dataset "$LEROBOT_DATASET" \
  --episode 0 \
  --paused
```

不激活环境时，把上例中的 `python` 替换为 `.venv/bin/python` 即可。

窗口内操作：

- `Space`：播放或暂停。
- `R`：回到当前 episode 第 1 帧。
- `[` 或 `,`：上一条 episode。
- `]` 或 `.`：下一条 episode。
- 鼠标：使用 MuJoCo 原生相机控制；关闭窗口即可退出。

红色和蓝色半透明球分别是录制的左、右 TCP 目标；每个球旁的 RGB 小点分别表示该 TCP 的局部 `+X/+Y/+Z` 方向。黄色板面和两侧橙色线框是实测标定板与三角支撑。夹爪在所有帧都固定为厂商 CAD 的 `q=0` 参考造型；这样不会再由旧合并 URDF 的人工手指预旋转造成交错显示。

首次打开一个 episode 时会以 Aik7 为该条轨迹选择连续的解析关节分支，切换到另一条 episode 时会按需加载并求解。终端显示的毫米/角度残差只是“天机模型当前是否贴住目标”的画面核对信息，不是可达性或安全结论。若某一帧 Aik7 没有返回解析分支，查看器会明确写出“无解析分支，保持前帧”；它不会悄悄切换到数值 IK 或给出安全结论。

## 逆解与 TCP → 法兰转换

这里使用的是 ROS 工作区第二套方案：`Aik7_ow`（`/start_wbc` 的解析 IK）。第一套 Pinocchio/HQP 是遥操作的实时增量控制器，本项目不会使用它。

记录的 UMI pose 和天机目标均是夹爪 `gripper_base_link` TCP；Aik7 的输入端却是 `ee_link`。启动时项目从**当前已导入的 MuJoCo URDF**读取每只手的固定链，而不是沿用旧脚本的硬编码四元数：

```text
gripper_base_link (TCP) → Link7 (法兰) → ee_link (Aik7 端点)
```

每帧先把 TCP 目标变到 `Link7`，再变到 `ee_link` 后交给 Aik7；最终把得到的七轴关节角回填 MuJoCo，并重新以 `gripper_base_link` 对原始 TCP 计算显示残差。启动时还会做一次 Aik7 正解与当前 URDF/MuJoCo `ee_link` 正解的交叉校验；若模型版本不一致，程序会停止，而不会把旧模型关节角静默写入新 URDF。

分支连续性遵循解析 WBC 的思路：先固定使用上一帧的第 7 轴；该冗余配置没有解析分支时，才在第 7 轴限位内按 1° 网格扫描，最后选择距上一帧关节向量最近的候选。默认肘部根沿用 WBC 的左负、右正；若该根在当前 URDF 限位下完全没有解析分支，才尝试同一 Aik7 求解器的另一肘部根，查看器会明确标注“备用肘部分支”。这避免了旧数值 IK 的随机重启导致的跨帧跳支路。

## 指定任意 LeRobot v3.0 数据集

运行时用 `--dataset` 指向 LeRobot v3 根目录：

```text
my_dataset/
├── meta/info.json
├── meta/episodes/chunk-*/file-*.parquet
└── data/chunk-*/file-*.parquet
```

先检查它被识别到的 episode 和 TCP 字段布局：

```bash
python scripts/replay_lerobot.py \
  --dataset "$LEROBOT_DATASET" \
  --list-episodes
```

默认只接受 `observation.state` 中明确命名的以下字段，而不会把任意向量的前 14 个数字误当作 TCP：

```text
left_x left_y left_z left_qx left_qy left_qz left_qw
right_x right_y right_z right_qx right_qy right_qz right_qw
```

如果你的 v3 数据集把 TCP 放在其他 feature 或字段名中，提供可审计的布局文件：

```bash
python scripts/replay_lerobot.py \
  --dataset "$LEROBOT_DATASET" \
  --pose-feature observation.state \
  --pose-layout config/pose_layout.example.json \
  --episode 12
```

布局文件必须显式声明 `quaternion_order: "xyzw"`。它可以使用字段名，也可使用整数索引；详见 [config/pose_layout.example.json](config/pose_layout.example.json)。LeRobot v3 只规定文件容器，不规定“某个向量一定是左右夹爪 TCP”，所以对不同录制格式必须显式确认这个布局。

## 不打开窗口的检查

以下命令会读取一个 episode、准备 MuJoCo 场景并完成 Aik7 解析回放，但不要求 OpenGL：

```bash
python scripts/replay_lerobot.py \
  --dataset "$LEROBOT_DATASET" \
  --episode 0 \
  --dry-run
```

这适合先检查数据格式、URDF asset 路径和坐标变换是否可以运行。每次运行会在 `outputs/replay/`（或 `--output` 指定处）生成：

- `scene/scene.xml`：当前 MuJoCo 场景；
- `scene/tianji_prepared.urdf` 和必要的唯一化 mesh 副本：MuJoCo 导入用的副本，不改原厂 URDF；
- `replay_manifest.json`：数据集 hash、字段布局、场景坐标约定和“仅回放”的记录。
- `virtual_scene_registration.json`：由当前仿真场景导出的 `base_link ← umi_world` 完整 6-DoF 配准。

不会输出旧项目中的 `results.json`、碰撞/可达性 CSV、NPZ 分析结果或 PASS/STOP 判定。

## 更换桌子或机器人摆位

仓库还提供一个经过全量数据验证的候选摆位
`config/base_up_60mm_farther_100mm_scene.json`。相对默认配置，它保持机器人朝向和
桌子/标定板不变，只把天机底座抬高 60 mm、沿桌面前后方向拉远 100 mm：

```bash
python scripts/replay_lerobot.py \
  --dataset "$LEROBOT_DATASET" \
  --scene-config config/base_up_60mm_farther_100mm_scene.json \
  --episode 0 \
  --paused
```

这个配置是针对当前杯子数据的可达性和关节余量候选，不代表所有数据集的通用最优
安装位。`root_in_table_m` 的第二项从 `-0.20` 改为 `-0.30` 表示机器人离桌子更远
100 mm；数值越接近 0 表示越靠近桌子。

复制并编辑默认配置，然后把新文件传给 `--scene-config`：

```bash
cp config/default_scene.json config/my_scene.json
# 编辑 config/my_scene.json

python scripts/replay_lerobot.py \
  --dataset "$LEROBOT_DATASET" \
  --scene-config config/my_scene.json \
  --episode 0
```

`rotation_robot_from_table` 必须是合法的右手 `SO(3)` 矩阵（正交且 `det=+1`）。`rotation_table_from_umi` 不再手写：它由 `calibration_board` 的三条实测边、以及 `umi_world` 中显式给出的 `+X` 和 `+Y` 水平投影自动推导。程序会拒绝镜像、三角形不闭合或不完整的配置，避免静默改变左右手或姿态。

## 仿真场景配准（不是物理手眼标定）

旧的真实机器人 YAML 不会被这个项目读取。当前仿真里没有相机观测，因此不能凭空“重新估计”真实世界的手眼外参；项目做的是可重复的**虚拟场景配准**：根据当前桌子、机器人根坐标、标定板尺寸和世界原点，生成完整的：

```text
p_base = R_base_from_umi @ p_umi + t_base_from_umi
R_base_tcp = R_base_from_umi @ R_umi_tcp
```

单独导出并检查该配准：

```bash
python scripts/calibrate_simulated_scene.py
```

默认配置会得到 `parent_frame=base_link`、`child_frame=umi_world` 的 JSON；它基于非共面虚拟板点拟合，并会写入残差和闭环误差。文件明确标记 `physical_handeye: false`：它保证“当前 MuJoCo 场景内部的机器人、桌子、板和轨迹使用同一坐标链”，但不能替代真实相机/机器人测得的物理手眼标定。

## 模型处理说明

天机合并 URDF 中有重名 STL，并且过去的合并夹爪关节含有与厂商 CAD 不一致的手指预旋转。启动时项目只在输出目录生成一份 MuJoCo 专用 URDF 副本：

- 解析 `package://` 模型路径；
- 只对重名 mesh 生成唯一文件名，防止夹爪误显示为支架；
- 将四个夹爪手指恢复为厂商 q=0 CAD 参考定义并锁在极小范围内。

原始模型文件不会被修改。

若使用自己的天机 URDF，可传入 `--urdf`；若该 URDF 使用额外的 `package://` mesh，重复提供 `--package-root <ros-package-root>`。

## 图形窗口故障

如果看到 `GLX: Failed to create context` 或 `could not create window`，这通常是当前终端/远程会话没有可用的桌面 OpenGL 上下文，而不是 TCP 数据或 URDF 的读取错误。先用 `--dry-run` 验证回放核心；打开窗口时请在具有可用 `DISPLAY`、X11/桌面 OpenGL 的本机终端运行。被动交互查看器需要真实图形上下文，不能用无窗口的 EGL/OSMesa 替代。

## 测试

```bash
PYTHONPATH=src python -m unittest -v
```

测试覆盖坐标变换、LeRobot v3 named-field 读取、夹爪 CAD 零位 URDF、天机 TCP Jacobian 有限差分对照、已知 TCP 位姿的数值 IK 回收，以及 Aik7 与 MuJoCo 的 TCP → 法兰 → `ee_link` 运动学闭环。实际数据集的 `--dry-run` 是完整场景与解析 IK 的额外 smoke test。
