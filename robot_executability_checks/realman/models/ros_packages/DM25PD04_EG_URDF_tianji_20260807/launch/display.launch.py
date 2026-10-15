from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


PACKAGE_NAME = "dm25pd04_eg_urdf_tianji_20260807"


def generate_launch_description():
    package_share = Path(get_package_share_directory(PACKAGE_NAME))
    default_model = package_share / "urdf" / "DM25PD04_EG_URDF_tianji_20260807.urdf"
    default_rviz_config = package_share / "rviz" / "display.rviz"

    model = LaunchConfiguration("model")
    use_rviz = LaunchConfiguration("use_rviz")
    use_joint_state_publisher = LaunchConfiguration("use_joint_state_publisher")
    robot_description = ParameterValue(Command(["cat ", model]), value_type=str)

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "model",
                default_value=str(default_model),
                description="Absolute path to the URDF model file.",
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description="Start RViz2 with the gripper display configuration.",
            ),
            DeclareLaunchArgument(
                "use_joint_state_publisher",
                default_value="true",
                description="Publish default values for the movable gripper joints.",
            ),
            Node(
                package="joint_state_publisher",
                executable="joint_state_publisher",
                name="joint_state_publisher",
                condition=IfCondition(use_joint_state_publisher),
                parameters=[{"robot_description": robot_description}],
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                parameters=[{"robot_description": robot_description}],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                condition=IfCondition(use_rviz),
                arguments=["-d", str(default_rviz_config)],
            ),
        ]
    )
