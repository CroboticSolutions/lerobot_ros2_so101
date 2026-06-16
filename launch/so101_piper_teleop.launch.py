#!/usr/bin/env python3
"""
Launch SO-101 leader node + SO-101→Piper bridge together.

Usage:
  ros2 launch lerobot_ros2_so101 so101_piper_teleop.launch.py port:=/dev/ttyACM0

Optional args:
  scale            -- global motion scale (default 1.0, use 0.5 to halve speed)
  trajectory_ms    -- waypoint horizon in milliseconds (default 100)
  arm_topic        -- arm controller topic (default /arm_controller/joint_trajectory)
  gripper_topic    -- gripper controller topic (default /gripper_controller/joint_trajectory)
  bridge_python    -- Python executable for the bridge (default /usr/bin/python3)
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
                              description="Trajectory waypoint horizon [ms]"),
        DeclareLaunchArgument("arm_topic",
                              default_value="/arm_controller/joint_trajectory",
                              description="Piper arm controller trajectory topic"),
        DeclareLaunchArgument("gripper_topic",
                              default_value="/gripper_controller/joint_trajectory",
                              description="Piper gripper controller trajectory topic"),
        DeclareLaunchArgument("bridge_python", default_value="/usr/bin/python3",
                              description="Python executable with matching rclpy"),

        # SO-101 leader — publishes on /so101/joint_states (not /joint_states)
        # so it doesn't collide with Piper sim's joint_state_broadcaster.
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

        # Bridge: /so101/joint_states → arm_controller + gripper_controller
        #
        Node(
            package="lerobot_ros2_so101",
            executable="so101_piper_teleop_bridge.py",
            name="so101_piper_teleop_bridge",
            prefix=LaunchConfiguration("bridge_python"),
            parameters=[{
                "so101_joint_states_topic": "/so101/joint_states",
                "arm_trajectory_topic":     LaunchConfiguration("arm_topic"),
                "gripper_trajectory_topic": LaunchConfiguration("gripper_topic"),
                "trajectory_duration_ms":   LaunchConfiguration("trajectory_ms"),
                "scale":                    LaunchConfiguration("scale"),
            }],
            output="screen",
        ),
    ])
