import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    model_path = LaunchConfiguration("model").perform(context)
    rviz_config_path = LaunchConfiguration("rviz_config").perform(context)
    use_rviz = LaunchConfiguration("use_rviz").perform(context).lower() in ("1", "true", "yes", "on")
    use_joint_state_publisher = LaunchConfiguration("use_joint_state_publisher").perform(context).lower() in ("1", "true", "yes", "on")
    use_arm_ik_markers = LaunchConfiguration("use_arm_ik_markers").perform(context).lower() in ("1", "true", "yes", "on")
    use_initial_joint_state_seed = LaunchConfiguration("use_initial_joint_state_seed").perform(context).lower() in ("1", "true", "yes", "on")
    wait_for_initial_joint_state = LaunchConfiguration("wait_for_initial_joint_state").perform(context).lower() in ("1", "true", "yes", "on")
    joint_state_rate = LaunchConfiguration("joint_state_rate")
    joint_state_topic = LaunchConfiguration("joint_state_topic")
    joint_command_topic = LaunchConfiguration("joint_command_topic")
    initial_joint_state_delay = max(
        0.01,
        float(LaunchConfiguration("initial_joint_state_delay").perform(context)),
    )
    package_share = get_package_share_directory("daimon_tj_fx_description")
    default_rviz_config = os.path.join(package_share, "rviz", "display.rviz")
    arm_ik_rviz_config = os.path.join(package_share, "rviz", "display_arm_ik_markers.rviz")

    if use_arm_ik_markers and rviz_config_path == default_rviz_config:
        rviz_config_path = arm_ik_rviz_config

    with open(model_path, "r", encoding="utf-8") as model_file:
        robot_description = model_file.read()

    if use_rviz:
        rviz_dir = Path.home() / ".rviz2"
        rviz_dir.mkdir(parents=True, exist_ok=True)
        (rviz_dir / "persistent_settings").touch(exist_ok=True)

    common_parameters = [{"robot_description": robot_description, "rate": joint_state_rate}]
    actions = []
    initial_joint_state_seed_node = None

    if use_joint_state_publisher and not use_arm_ik_markers:
        actions.append(
            Node(
                package="joint_state_publisher",
                executable="joint_state_publisher",
                name="joint_state_publisher",
                parameters=common_parameters,
            )
        )
    elif use_initial_joint_state_seed and not use_arm_ik_markers:
        initial_joint_state_seed_node = Node(
            package="daimon_tj_fx_description",
            executable="initial_joint_state_seed_node",
            name="initial_joint_state_seed_node",
            parameters=[
                {
                    "joint_topic": "/joint_states",
                    "publish_delay_sec": initial_joint_state_delay,
                }
            ],
        )

    actions.append(
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            parameters=common_parameters,
        )
    )

    if use_arm_ik_markers:
        actions.append(
            Node(
                package="daimon_tj_fx_description",
                executable="wbc_arm_ik_goal_markers_node",
                name="wbc_arm_ik_goal_markers_node",
                parameters=[
                    {
                        "urdf_path": model_path,
                        "frame_id": "base_link",
                        "goal_topic": "/wbc_goal_pose",
                    }
                ],
            )
        )
        actions.append(
            Node(
                package="daimon_arm_ik_ros2",
                executable="wbc_arm_ik_joint_state_node",
                name="wbc_arm_ik_joint_state_node",
                parameters=[
                    {
                        "urdf_path": model_path,
                        "joint_state_topic": joint_state_topic,
                        "pose_expand_topic": "/wbc_goal_pose",
                        "joint_command_topic": joint_command_topic,
                        "wait_for_initial_joint_state": wait_for_initial_joint_state,
                    }
                ],
            )
        )

    rviz_node = None
    if use_rviz:
        rviz_node = Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config_path, "-f", "base_link"],
        )
        actions.append(rviz_node)

    if initial_joint_state_seed_node is not None:
        if rviz_node is not None:
            actions.append(
                RegisterEventHandler(
                    OnProcessStart(
                        target_action=rviz_node,
                        on_start=[initial_joint_state_seed_node],
                    )
                )
            )
        else:
            actions.append(initial_joint_state_seed_node)

    return actions


def generate_launch_description():
    package_share = get_package_share_directory("daimon_tj_fx_description")
    default_model = os.path.join(package_share, "urdf", "daimon_tj_fx_dual_arm_with_gripper.urdf")
    default_rviz_config = os.path.join(package_share, "rviz", "display.rviz")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model",
                default_value=default_model,
                description="Absolute path to the URDF model file.",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=default_rviz_config,
                description="Absolute path to the RViz config file.",
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description="Whether to launch RViz2.",
            ),
            DeclareLaunchArgument(
                "use_joint_state_publisher",
                default_value="false",
                description="Whether to launch joint_state_publisher. Ignored when use_arm_ik_markers is true.",
            ),
            DeclareLaunchArgument(
                "use_initial_joint_state_seed",
                default_value="true",
                description="When joint_state_publisher is disabled, publish one zero joint-state command after startup. If RViz is enabled, this waits for the RViz process to start first.",
            ),
            DeclareLaunchArgument(
                "use_arm_ik_markers",
                default_value="false",
                description="Launch ROS2 end-effector drag markers plus the differential IK joint-state publisher.",
            ),
            DeclareLaunchArgument(
                "joint_state_topic",
                default_value="/joint_states",
                description="Joint-state feedback topic used by the ROS2 IK node when use_arm_ik_markers is true.",
            ),
            DeclareLaunchArgument(
                "joint_state_rate",
                default_value="1000",
                description="Publish rate for joint_state_publisher in Hz when it is enabled.",
            ),
            DeclareLaunchArgument(
                "joint_command_topic",
                default_value="/joint_states",
                description="Joint-state output topic used by the ROS2 IK node when use_arm_ik_markers is true.",
            ),
            DeclareLaunchArgument(
                "wait_for_initial_joint_state",
                default_value="false",
                description="When use_arm_ik_markers is true, wait for the first joint-state feedback before solving IK.",
            ),
            DeclareLaunchArgument(
                "initial_joint_state_delay",
                default_value="2.0",
                description="Delay in seconds before publishing the one-shot zero joint-state seed. When RViz is enabled, the delay starts after the RViz process launches.",
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
