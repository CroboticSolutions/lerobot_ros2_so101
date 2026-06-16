import math

import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# (piper_joint, scale, offset_rad, piper_min, piper_max)
#
# scale  — SO-101 radians are multiplied by this before adding offset
# offset — added after scale so that SO-101 center (0 rad) lands at a valid
#          Piper home position:
#            joint2 must be in [0, π]    → offset π/2 centres it
#            joint3 must be in [-π, 0]   → offset -π/4 centres it
#
# All limits taken from piper_description.urdf.
_JOINT_MAP: dict[str, tuple[str, float, float, float, float]] = {
    "shoulder_pan":  ("joint1", -1.0,     0.0,          -2.618,  2.168),
    "shoulder_lift": ("joint2",  1.0,     math.pi / 2,   0.0,    3.14),
    "elbow_flex":    ("joint3",  1.0,    -math.pi / 4,  -2.967,  0.0),
    "wrist_flex":    ("joint5",  1.0,     0.0,          -1.22,   1.22),
    "wrist_roll":    ("joint4", -1.0,     0.0,          -1.745,  1.745),
}

# joint6 (end-effector rotation) is kept at 0 — SO-101 has no equivalent DOF
_JOINT6_HOME = 0.0

# Gripper: SO-101 "gripper" joint (rad, centred at 0) → metres
# Physical SO-101 gripper range is ~±0.5 rad (not the full ±π of the encoder).
# These defaults match the typical calibration range from the lerobot calibration file.
# Override with gripper_open_rad / gripper_closed_rad params if your calibration differs.
_GRIPPER_OPEN_M   = 0.0
_GRIPPER_CLOSED_M = 0.035
_GRIPPER_SO101_OPEN_RAD   = -0.035  # range_min=2025 → (2025-2048)*2π/4096
_GRIPPER_SO101_CLOSED_RAD =  2.012  # range_max=3359 → (3359-2048)*2π/4096


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _make_duration(nanoseconds: int) -> Duration:
    d = Duration()
    d.sec = nanoseconds // 1_000_000_000
    d.nanosec = nanoseconds % 1_000_000_000
    return d


def _gripper_metres(raw: float, open_rad: float, closed_rad: float) -> float:
    """Convert SO-101 gripper raw value (rad) to Piper metres [0, 0.035].

    Normalises over the actual physical SO-101 gripper range instead of the
    theoretical ±π encoder range, so the full Piper gripper stroke is used.
    """
    norm = _clamp((raw - open_rad) / (closed_rad - open_rad), 0.0, 1.0)
    return _GRIPPER_OPEN_M + norm * (_GRIPPER_CLOSED_M - _GRIPPER_OPEN_M)


class TeleopBridge(Node):
    """
    Subscribes to SO-101 leader JointState and forwards position commands
    to Piper.

    mode = "sim"  (default)
        Publishes JointTrajectory to arm_controller and gripper_controller
        (ros2_control, Gazebo simulation).

    mode = "real"
        Publishes a single JointState to joint_ctrl_single
        (piper_ctrl_single_node, real CAN-based robot).
    """

    def __init__(self) -> None:
        super().__init__("so101_piper_teleop_bridge")

        # ── parameters ────────────────────────────────────────────────────────
        self.declare_parameter("mode", "sim")
        self.declare_parameter("so101_joint_states_topic", "/so101/joint_states")
        # sim-mode topics
        self.declare_parameter("arm_trajectory_topic",
                               "/arm_controller/joint_trajectory")
        self.declare_parameter("gripper_trajectory_topic",
                               "/gripper_controller/joint_trajectory")
        self.declare_parameter("trajectory_duration_ms", 100)
        # real-mode topic
        self.declare_parameter("joint_ctrl_topic", "joint_ctrl_single")
        # common
        self.declare_parameter("scale", 1.0)
        self.declare_parameter("gripper_open_rad",   _GRIPPER_SO101_OPEN_RAD)
        self.declare_parameter("gripper_closed_rad", _GRIPPER_SO101_CLOSED_RAD)

        self._mode       = self.get_parameter("mode").value
        so101_topic      = self.get_parameter("so101_joint_states_topic").value
        self._scale            = self.get_parameter("scale").value
        self._grip_open_rad   = self.get_parameter("gripper_open_rad").value
        self._grip_closed_rad = self.get_parameter("gripper_closed_rad").value
        duration_ms           = self.get_parameter("trajectory_duration_ms").value
        self._traj_dur_ns: int = duration_ms * 1_000_000

        # ── publishers ────────────────────────────────────────────────────────
        if self._mode == "real":
            ctrl_topic = self.get_parameter("joint_ctrl_topic").value
            self._ctrl_pub = self.create_publisher(JointState, ctrl_topic, 10)
            self.get_logger().info(
                f"TeleopBridge ready  |  mode=real  |  input: {so101_topic}"
                f"  |  cmd → {ctrl_topic}  |  scale={self._scale}"
            )
        else:
            arm_topic  = self.get_parameter("arm_trajectory_topic").value
            grip_topic = self.get_parameter("gripper_trajectory_topic").value
            self._arm_pub  = self.create_publisher(JointTrajectory, arm_topic, 10)
            self._grip_pub = self.create_publisher(JointTrajectory, grip_topic, 10)
            self.get_logger().info(
                f"TeleopBridge ready  |  mode=sim  |  input: {so101_topic}"
                f"  |  arm → {arm_topic}  |  gripper → {grip_topic}"
                f"  |  scale={self._scale}  duration_ms={duration_ms}"
            )

        self._sub = self.create_subscription(
            JointState, so101_topic, self._on_joint_state, 10
        )

    # ── callback ───────────────────────────────────────────────────────────────

    def _on_joint_state(self, msg: JointState) -> None:
        positions: dict[str, float] = dict(zip(msg.name, msg.position))
        if self._mode == "real":
            self._publish_real(positions)
        else:
            self._publish_sim_arm(positions)
            self._publish_sim_gripper(positions)

    # ── sim mode ───────────────────────────────────────────────────────────────

    def _publish_sim_arm(self, positions: dict[str, float]) -> None:
        names: list[str] = []
        pos: list[float] = []

        for so101_name, (piper_name, scale, offset, lo, hi) in _JOINT_MAP.items():
            raw = positions.get(so101_name)
            if raw is None:
                self.get_logger().warn(
                    f"Expected SO-101 joint '{so101_name}' not in message — skipping",
                    throttle_duration_sec=5.0,
                )
                return
            names.append(piper_name)
            pos.append(_clamp(raw * scale * self._scale + offset, lo, hi))

        names.append("joint6")
        pos.append(_JOINT6_HOME)

        traj = JointTrajectory()
        # stamp = 0 → controller uses its own "now" on receipt,
        # avoiding sim vs wall clock mismatch.
        traj.joint_names = names
        pt = JointTrajectoryPoint()
        pt.positions = pos
        pt.time_from_start = _make_duration(self._traj_dur_ns)
        traj.points = [pt]
        self._arm_pub.publish(traj)

    def _publish_sim_gripper(self, positions: dict[str, float]) -> None:
        raw = positions.get("gripper")
        if raw is None:
            return
        opening_m = _gripper_metres(raw, self._grip_open_rad, self._grip_closed_rad)

        traj = JointTrajectory()
        traj.joint_names = ["joint7", "joint8"]
        pt = JointTrajectoryPoint()
        pt.positions = [opening_m, -opening_m]
        pt.time_from_start = _make_duration(self._traj_dur_ns)
        traj.points = [pt]
        self._grip_pub.publish(traj)

    # ── real mode ──────────────────────────────────────────────────────────────

    def _publish_real(self, positions: dict[str, float]) -> None:
        names: list[str] = []
        pos: list[float] = []

        for so101_name, (piper_name, scale, offset, lo, hi) in _JOINT_MAP.items():
            raw = positions.get(so101_name)
            if raw is None:
                self.get_logger().warn(
                    f"Expected SO-101 joint '{so101_name}' not in message — skipping",
                    throttle_duration_sec=5.0,
                )
                return
            names.append(piper_name)
            pos.append(_clamp(raw * scale * self._scale + offset, lo, hi))

        names.append("joint6")
        pos.append(_JOINT6_HOME)

        # Gripper — piper_ctrl_single_node expects metres at index 6
        raw_grip = positions.get("gripper")
        names.append("gripper")
        pos.append(_gripper_metres(raw_grip, self._grip_open_rad, self._grip_closed_rad) if raw_grip is not None else _GRIPPER_OPEN_M)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        msg.position = pos
        self._ctrl_pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = TeleopBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
