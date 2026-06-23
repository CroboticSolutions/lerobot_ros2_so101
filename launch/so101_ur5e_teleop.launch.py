#!/usr/bin/env python3
"""
SO-101 leader → UR5e (+ Robotiq 2F-85) teleoperation for ur_simulation_gz.

Assumes the UR sim is already running in another terminal, e.g.:

  ros2 launch ur_simulation_gz ur_sim_moveit.launch.py \\
      ur_type:=ur5e use_robotiq_gripper:=true \\
      semantic_description_file:=srdf/ur_robotiq.srdf.xacro moveit_start_delay_s:=12.0

This launch starts:
  1. SO-101 leader node  — publishes encoder positions on /so101/joint_states
  2. UR5eTeleopBridge     — maps SO-101 → UR scaled_joint_trajectory_controller
                            (arm) + robotiq_gripper_controller (gripper action)

Teleop is relative (safe-start): on the first reading it captures the UR's
current pose, so the arm does not jump.  Move the SO-101 leader and the UR
follows; the gripper tracks the SO-101 gripper.

Usage:
  ros2 launch lerobot_ros2_so101 so101_ur5e_teleop.launch.py port:=/dev/ttyACM0

Optional args:
  port            -- SO-101 USB port (default /dev/ttyACM0)
  scale           -- global motion scale 0.0–1.0 (default 1.0; 0.5 = half range)
  trajectory_ms   -- fixed waypoint horizon fallback [ms] (default 100)
  arm_topic       -- UR arm trajectory topic
                     (default /scaled_joint_trajectory_controller/joint_trajectory)
  gripper_action  -- Robotiq gripper action (default /robotiq_gripper_controller/gripper_cmd)
  bridge_python   -- Python executable for the bridge (default /usr/bin/python3)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("port",          default_value="/dev/ttyACM0",
                              description="SO-101 USB port"),
        DeclareLaunchArgument("scale",         default_value="1.0",
                              description="Global motion scale (0.0–1.0)"),
        DeclareLaunchArgument("trajectory_ms", default_value="100",
                              description="Fallback trajectory waypoint horizon [ms]"),
        DeclareLaunchArgument("arm_topic",
                              default_value="/scaled_joint_trajectory_controller/joint_trajectory",
                              description="UR arm controller trajectory topic"),
        DeclareLaunchArgument("gripper_action",
                              default_value="/robotiq_gripper_controller/gripper_cmd",
                              description="Robotiq ParallelGripperCommand action"),
        DeclareLaunchArgument("bridge_python", default_value="/usr/bin/python3",
                              description="Python executable with matching rclpy"),

        # SO-101 leader — publishes on /so101/joint_states so it doesn't collide
        # with the UR sim's joint_state_broadcaster on /joint_states.
        Node(
            package="lerobot_ros2_so101",
            executable="leader_node.py",
            name="so101_leader",
            parameters=[{
                "port": LaunchConfiguration("port"),
                "joint_states": "/so101/joint_states",
            }],
            output="screen",
        ),

        # Bridge: /so101/joint_states → UR arm trajectory + gripper action.
        Node(
            package="lerobot_ros2_so101",
            executable="so101_ur5e_teleop_bridge.py",
            name="so101_ur5e_teleop_bridge",
            prefix=LaunchConfiguration("bridge_python"),
            parameters=[{
                "so101_joint_states_topic": "/so101/joint_states",
                "ur_feedback_topic":        "/joint_states",
                "arm_trajectory_topic":     LaunchConfiguration("arm_topic"),
                "gripper_action":           LaunchConfiguration("gripper_action"),
                "trajectory_duration_ms":   LaunchConfiguration("trajectory_ms"),
                "scale":                    LaunchConfiguration("scale"),
            }],
            output="screen",
        ),
    ])
