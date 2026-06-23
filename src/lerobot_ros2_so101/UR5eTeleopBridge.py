"""
Bridge SO-101 leader commands to a UR5e (+ Robotiq 2F-85) running in
ur_simulation_gz with MoveIt.

Differences from the Piper bridge (TeleopBridge):
  - Arm is driven via JointTrajectory to the UR ``scaled_joint_trajectory_controller``
    (6 joints, position command interface).
  - Gripper is NOT a JointTrajectory topic.  It is a
    ``parallel_gripper_action_controller/GripperActionController`` exposing the
    ``control_msgs/action/ParallelGripperCommand`` action at
    ``/robotiq_gripper_controller/gripper_cmd``.  We drive it with an action client.

Teleop is RELATIVE (safe-start) by default: on the first valid SO-101 message we
capture the leader pose AND the UR's current joint pose (from /joint_states), then
command UR = ur_home + sign*scale*(leader_now - leader_home).  This guarantees the
arm never jumps at startup and removes the need for hand-tuned absolute offsets.

Features carried over from TeleopBridge:
  - Exponential low-pass filter on the mapped command
  - Deadband: skip publishing if nothing moved enough
  - Adaptive time_from_start scaled with movement magnitude
  - Watchdog: holds last command if the SO-101 signal is lost
"""

import time

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import ParallelGripperCommand
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# SO-101 leader joint  →  (ur_joint, sign)
# wrist_2_joint has no leader counterpart: it stays at its captured home.
_JOINT_MAP: dict[str, tuple[str, float]] = {
    "shoulder_pan":  ("shoulder_pan_joint",  1.0),
    "shoulder_lift": ("shoulder_lift_joint", 1.0),
    "elbow_flex":    ("elbow_joint",         1.0),
    "wrist_flex":    ("wrist_1_joint",       1.0),
    "wrist_roll":    ("wrist_3_joint",       1.0),
}

# UR joints in publish order.  wrist_2_joint is the passive joint (held at home).
_UR_JOINT_ORDER = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# UR5e position limits [rad] (elbow artificially limited to +-pi, see ur_description).
_UR_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan_joint":  (-6.28, 6.28),
    "shoulder_lift_joint": (-6.28, 6.28),
    "elbow_joint":         (-3.14, 3.14),
    "wrist_1_joint":       (-6.28, 6.28),
    "wrist_2_joint":       (-6.28, 6.28),
    "wrist_3_joint":       (-6.28, 6.28),
}

_GRIPPER_JOINT = "robotiq_85_left_knuckle_joint"
_GRIPPER_OPEN_M = 0.0
_GRIPPER_CLOSED_M = 0.043
# SO-101 gripper raw range (radians), from the existing calibration.
_GRIPPER_SO101_OPEN_RAD = -0.57
_GRIPPER_SO101_CLOSED_RAD = +1.54

_MAX_VEL_RAD_S = 1.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _make_duration(nanoseconds: int) -> Duration:
    d = Duration()
    d.sec = nanoseconds // 1_000_000_000
    d.nanosec = nanoseconds % 1_000_000_000
    return d


def _gripper_metres(raw: float, open_rad: float, closed_rad: float) -> float:
    norm = _clamp((raw - open_rad) / (closed_rad - open_rad), 0.0, 1.0)
    return _GRIPPER_OPEN_M + norm * (_GRIPPER_CLOSED_M - _GRIPPER_OPEN_M)


class UR5eTeleopBridge(Node):
    def __init__(self) -> None:
        super().__init__("so101_ur5e_teleop_bridge")

        # ── parameters ────────────────────────────────────────────────────────
        self.declare_parameter("so101_joint_states_topic", "/so101/joint_states")
        self.declare_parameter("ur_feedback_topic", "/joint_states")
        self.declare_parameter(
            "arm_trajectory_topic", "/scaled_joint_trajectory_controller/joint_trajectory"
        )
        self.declare_parameter("gripper_action", "/robotiq_gripper_controller/gripper_cmd")
        self.declare_parameter("scale", 1.0)
        self.declare_parameter("trajectory_duration_ms", 100)
        self.declare_parameter("gripper_open_rad", _GRIPPER_SO101_OPEN_RAD)
        self.declare_parameter("gripper_closed_rad", _GRIPPER_SO101_CLOSED_RAD)
        self.declare_parameter("gripper_max_effort", 50.0)
        self.declare_parameter("lp_alpha", 0.3)
        self.declare_parameter("deadband", 0.002)
        self.declare_parameter("max_vel", _MAX_VEL_RAD_S)
        self.declare_parameter("min_traj_ms", 50)
        self.declare_parameter("watchdog_s", 0.5)
        self.declare_parameter("gripper_deadband_m", 0.0008)
        # Per-joint sign override (e.g. flip a joint that moves the wrong way).
        for so101_name, (_, sign) in _JOINT_MAP.items():
            self.declare_parameter(f"sign.{so101_name}", sign)

        so101_topic = self.get_parameter("so101_joint_states_topic").value
        feedback_topic = self.get_parameter("ur_feedback_topic").value
        arm_topic = self.get_parameter("arm_trajectory_topic").value
        gripper_action = self.get_parameter("gripper_action").value
        self._scale = float(self.get_parameter("scale").value)
        self._grip_open_rad = float(self.get_parameter("gripper_open_rad").value)
        self._grip_closed_rad = float(self.get_parameter("gripper_closed_rad").value)
        self._grip_max_effort = float(self.get_parameter("gripper_max_effort").value)
        self._lp_alpha = _clamp(float(self.get_parameter("lp_alpha").value), 0.0, 1.0)
        self._deadband = float(self.get_parameter("deadband").value)
        self._max_vel = float(self.get_parameter("max_vel").value)
        self._grip_deadband_m = float(self.get_parameter("gripper_deadband_m").value)
        self._min_traj_ns = int(float(self.get_parameter("min_traj_ms").value) * 1_000_000)
        self._fixed_traj_ns = int(
            float(self.get_parameter("trajectory_duration_ms").value) * 1_000_000
        )
        watchdog_s = float(self.get_parameter("watchdog_s").value)

        self._signs = {
            so101_name: float(self.get_parameter(f"sign.{so101_name}").value)
            for so101_name in _JOINT_MAP
        }

        self._validate_parameters(watchdog_s)

        # ── state ─────────────────────────────────────────────────────────────
        self._filtered: dict[str, float] | None = None
        self._prev: dict[str, float] | None = None
        self._prev_grip_m: float | None = None
        self._last_grip_m: float | None = None
        self._last_msg_time: float | None = None
        # safe-start capture
        self._ur_feedback: dict[str, float] | None = None
        self._leader_home: dict[str, float] | None = None
        self._ur_home: dict[str, float] | None = None

        # ── ROS interfaces ────────────────────────────────────────────────────
        self._arm_pub = self.create_publisher(JointTrajectory, arm_topic, 10)
        self._grip_client = ActionClient(self, ParallelGripperCommand, gripper_action)
        self._feedback_sub = self.create_subscription(
            JointState, feedback_topic, self._on_ur_feedback, 10
        )
        self._sub = self.create_subscription(
            JointState, so101_topic, self._on_joint_state, 10
        )
        self._watchdog_timer = self.create_timer(watchdog_s, self._watchdog_cb)

        self.get_logger().info(
            f"UR5eTeleopBridge ready  |  {so101_topic} → {arm_topic} (+ gripper action "
            f"{gripper_action})  |  scale={self._scale}  alpha={self._lp_alpha}  "
            f"deadband={self._deadband}  watchdog={watchdog_s}s  (relative/safe-start)"
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _validate_parameters(self, watchdog_s: float) -> None:
        errors: list[str] = []
        if not 0.0 <= self._scale <= 1.0:
            errors.append("scale must be between 0.0 and 1.0")
        if self._grip_closed_rad <= self._grip_open_rad:
            errors.append("gripper_closed_rad must be greater than gripper_open_rad")
        if self._deadband < 0.0:
            errors.append("deadband must be >= 0.0")
        if self._grip_deadband_m < 0.0:
            errors.append("gripper_deadband_m must be >= 0.0")
        if self._max_vel < 0.0:
            errors.append("max_vel must be >= 0.0")
        if self._min_traj_ns <= 0:
            errors.append("min_traj_ms must be > 0")
        if self._fixed_traj_ns <= 0:
            errors.append("trajectory_duration_ms must be > 0")
        if watchdog_s <= 0.0:
            errors.append("watchdog_s must be > 0.0")
        if errors:
            msg = "; ".join(errors)
            self.get_logger().fatal(f"Invalid UR5eTeleopBridge parameters: {msg}")
            raise ValueError(msg)

    def _on_ur_feedback(self, msg: JointState) -> None:
        self._ur_feedback = dict(zip(msg.name, msg.position))

    def _capture_home(self, raw: dict[str, float]) -> bool:
        """Capture leader + UR home once UR feedback for all arm joints is available."""
        fb = self._ur_feedback
        missing_ur = [j for j in _UR_JOINT_ORDER if fb is None or j not in fb]
        missing_leader = [n for n in _JOINT_MAP if n not in raw]
        if missing_ur or missing_leader:
            self.get_logger().warn(
                "Waiting for safe-start references — "
                f"UR joints missing: {missing_ur or 'none'}; "
                f"SO-101 joints missing: {missing_leader or 'none'}",
                throttle_duration_sec=2.0,
            )
            return False
        self._leader_home = {n: raw[n] for n in _JOINT_MAP}
        self._ur_home = {j: fb[j] for j in _UR_JOINT_ORDER}
        self.get_logger().info(
            "Safe-start captured current UR pose; teleop is now relative to it"
        )
        return True

    def _map_positions(self, raw: dict[str, float]) -> dict[str, float]:
        """UR target = ur_home + sign*scale*(leader_now - leader_home), clamped."""
        out: dict[str, float] = {}
        for so101_name, (ur_name, _) in _JOINT_MAP.items():
            delta = raw[so101_name] - self._leader_home[so101_name]
            target = self._ur_home[ur_name] + self._signs[so101_name] * self._scale * delta
            lo, hi = _UR_LIMITS[ur_name]
            out[ur_name] = _clamp(target, lo, hi)
        # passive joint holds its captured home
        out["wrist_2_joint"] = self._ur_home["wrist_2_joint"]
        return out

    def _gripper_from_raw(self, raw: dict[str, float]) -> float:
        raw_grip = raw.get("gripper")
        if raw_grip is None:
            return self._last_grip_m if self._last_grip_m is not None else _GRIPPER_OPEN_M
        return _gripper_metres(raw_grip, self._grip_open_rad, self._grip_closed_rad)

    def _low_pass(self, new: dict[str, float]) -> dict[str, float]:
        if self._filtered is None:
            return dict(new)
        a = self._lp_alpha
        return {k: a * new[k] + (1.0 - a) * self._filtered[k] for k in new}

    def _max_delta(self, curr: dict[str, float], prev: dict[str, float]) -> float:
        return max(abs(curr[k] - prev[k]) for k in curr)

    def _adaptive_duration_ns(self, max_delta: float) -> int:
        if self._max_vel <= 0.0 or max_delta <= 0.0:
            return self._fixed_traj_ns
        ns = int((max_delta / self._max_vel) * 1_000_000_000)
        return max(ns, self._min_traj_ns)

    # ── main callback ──────────────────────────────────────────────────────────

    def _on_joint_state(self, msg: JointState) -> None:
        raw = dict(zip(msg.name, msg.position))

        if self._ur_home is None:
            if not self._capture_home(raw):
                return

        mapped = self._map_positions(raw)
        grip_m = self._gripper_from_raw(raw)

        self._last_msg_time = time.monotonic()
        self._watchdog_timer.reset()

        filtered = self._low_pass(mapped)
        self._filtered = filtered

        if self._prev is not None:
            arm_delta = self._max_delta(filtered, self._prev)
            grip_delta = (
                abs(grip_m - self._prev_grip_m)
                if self._prev_grip_m is not None
                else 0.0
            )
            arm_moved = arm_delta >= self._deadband
            grip_moved = grip_delta >= self._grip_deadband_m
            if not arm_moved and not grip_moved:
                return
        else:
            arm_moved = grip_moved = True

        dur_ns = self._adaptive_duration_ns(
            self._max_delta(filtered, self._prev) if self._prev is not None else 0.0
        )

        self._prev = dict(filtered)

        if arm_moved:
            self._publish_arm(filtered, dur_ns)
        if grip_moved or self._last_grip_m is None:
            self._send_gripper(grip_m)
        self._prev_grip_m = grip_m
        self._last_grip_m = grip_m

    # ── publishers ───────────────────────────────────────────────────────────

    def _publish_arm(self, filtered: dict[str, float], dur_ns: int) -> None:
        traj = JointTrajectory()
        traj.joint_names = _UR_JOINT_ORDER
        pt = JointTrajectoryPoint()
        pt.positions = [filtered[n] for n in _UR_JOINT_ORDER]
        # velocities intentionally omitted — JTC rejects single-point trajectories
        # whose final point carries non-zero velocity.
        pt.time_from_start = _make_duration(dur_ns)
        traj.points = [pt]
        self._arm_pub.publish(traj)

    def _send_gripper(self, grip_m: float) -> None:
        if not self._grip_client.server_is_ready():
            self.get_logger().warn(
                "Gripper action server not ready — skipping gripper command",
                throttle_duration_sec=5.0,
            )
            return
        goal = ParallelGripperCommand.Goal()
        goal.command.name = [_GRIPPER_JOINT]
        goal.command.position = [grip_m]
        goal.command.effort = [self._grip_max_effort]
        # Fire-and-forget: the controller preempts the previous goal.
        self._grip_client.send_goal_async(goal)

    # ── watchdog ───────────────────────────────────────────────────────────────

    def _watchdog_cb(self) -> None:
        if self._last_msg_time is None or self._filtered is None:
            return
        self.get_logger().warn(
            "SO-101 signal lost — holding position", throttle_duration_sec=2.0
        )
        self._publish_arm(self._filtered, self._fixed_traj_ns)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = UR5eTeleopBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
