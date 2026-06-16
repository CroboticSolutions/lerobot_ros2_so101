#!/usr/bin/env python3
"""
SO-101 leader + Piper real robot teleoperation.

Pokreće:
  1. robot_state_publisher   — URDF za RViz vizualizaciju
  2. piper_ctrl_single_node  — CAN SDK driver (joint_ctrl_single → piper)
  3. MoveIt move_group       — planiranje + collision checking
  4. RViz                    — vizualizacija
  5. SO-101 leader node      — čita encoder pozicije
  6. TeleopBridge (real)     — mapira SO-101 → JointState na joint_ctrl_single

Preduvjeti:
  - CAN bus aktivan: sudo ip link set can0 up type can bitrate 1000000
    (ili: bash ~/arms_ws/src/piper_ros/can_activate.sh)
  - SO-101 spojen na USB (default /dev/ttyACM0)
  - Piper robot uključen i spreman

Pokretanje:
  ros2 launch lerobot_ros2_so101 so101_piper_real_teleop.launch.py

Opcionalni argumenti:
  can_port:=can0          CAN port (default can0)
  port:=/dev/ttyACM0      SO-101 USB port
  scale:=1.0              Skaliranje pokreta (0.5 = upola sporije)
  launch_rviz:=true       Pokreni RViz
  auto_enable:=true       Automatski enable Piper motora
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder

os.environ["RCUTILS_COLORIZED_OUTPUT"] = "1"


def generate_launch_description() -> LaunchDescription:
    moveit_config = MoveItConfigsBuilder(
        "piper", package_name="piper_with_gripper_moveit"
    ).to_moveit_configs()

    return LaunchDescription([
        # ── args ──────────────────────────────────────────────────────────────
        DeclareLaunchArgument("can_port",     default_value="can0",
                              description="CAN port za Piper robot"),
        DeclareLaunchArgument("port",         default_value="/dev/ttyACM0",
                              description="USB port SO-101 leader arma"),
        DeclareLaunchArgument("scale",        default_value="1.0",
                              description="Globalni scale pokreta (0.0–1.0)"),
        DeclareLaunchArgument("launch_rviz",  default_value="true",
                              description="Pokreni RViz"),
        DeclareLaunchArgument("auto_enable",  default_value="true",
                              description="Automatski enable Piper motora"),
        DeclareLaunchArgument("gripper_exist", default_value="true",
                              description="Piper ima gripper"),

        # ── 1. robot_state_publisher ──────────────────────────────────────────
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[moveit_config.robot_description],
        ),

        # ── 2. Piper CAN SDK driver ───────────────────────────────────────────
        # Subscribes to joint_ctrl_single (JointState, radians + metres)
        # Publishes joint state on joint_states_single i joint_states_feedback
        Node(
            package="piper",
            executable="piper_single_ctrl",
            name="piper_ctrl_single_node",
            output="screen",
            parameters=[{
                "can_port":            LaunchConfiguration("can_port"),
                "auto_enable":         LaunchConfiguration("auto_enable"),
                "gripper_exist":       LaunchConfiguration("gripper_exist"),
                "gripper_val_mutiple": 1,
            }],
        ),

        # ── 3. MoveIt move_group ──────────────────────────────────────────────
        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            name="move_group",
            output="screen",
            parameters=[
                moveit_config.to_dict(),
                {
                    "publish_robot_description_semantic": True,
                    "allow_trajectory_execution": True,
                    "publish_planning_scene": True,
                    "publish_geometry_updates": True,
                    "publish_state_updates": True,
                    "publish_transforms_updates": True,
                    "use_sim_time": False,
                },
            ],
        ),

        # ── 4. RViz ───────────────────────────────────────────────────────────
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="log",
            arguments=["-d", str(moveit_config.package_path / "config/moveit.rviz")],
            parameters=[
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
                moveit_config.planning_pipelines,
                moveit_config.robot_description_kinematics,
                {"use_sim_time": False},
            ],
            condition=IfCondition(LaunchConfiguration("launch_rviz")),
        ),

        # ── 5. SO-101 leader node ─────────────────────────────────────────────
        Node(
            package="lerobot_ros2_so101",
            executable="leader_node.py",
            name="so101_leader",
            parameters=[{
                "port":         LaunchConfiguration("port"),
                "joint_states": "/so101/joint_states",
            }],
            output="screen",
        ),

        # ── 6. Teleop bridge (real mode) ──────────────────────────────────────
        # Reads /so101/joint_states → publishes JointState to joint_ctrl_single
        Node(
            package="lerobot_ros2_so101",
            executable="so101_piper_teleop_bridge.py",
            name="so101_piper_teleop_bridge",
            prefix="/usr/bin/python3",
            parameters=[{
                "mode":                    "real",
                "so101_joint_states_topic": "/so101/joint_states",
                "joint_ctrl_topic":         "joint_ctrl_single",
                "scale":                    LaunchConfiguration("scale"),
            }],
            output="screen",
        ),
    ])
