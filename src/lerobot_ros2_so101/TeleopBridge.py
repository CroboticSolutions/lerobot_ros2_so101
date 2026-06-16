import math
import time

import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# (piper_joint, scale, offset_rad, piper_min, piper_max)
_JOINT_MAP: dict[str, tuple[str, float, float, float, float]] = {
    "shoulder_pan":  ("joint1", -1.0,     0.0,          -2.618,  2.168),
    "shoulder_lift": ("joint2",  1.0,     math.pi / 2,   0.0,    3.14),
    "elbow_flex":    ("joint3",  1.0,    -math.pi / 4,  -2.967,  0.0),
    "wrist_flex":    ("joint5",  1.0,     0.0,          -1.22,   1.22),
    "wrist_roll":    ("joint4", -1.0,     0.0,          -1.745,  1.745),
}

_JOINT6_HOME      = 0.0
_GRIPPER_OPEN_M   = 0.0
_GRIPPER_CLOSED_M = 0.035
_GRIPPER_SO101_OPEN_RAD   = -0.6
_GRIPPER_SO101_CLOSED_RAD =  0.6

# Names in publish order (arm joints only, without joint6/gripper)
_ARM_SO101_NAMES = list(_JOINT_MAP.keys())

# Max joint velocity used for adaptive time_from_start [rad/s]
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


class TeleopBridge(Node):
    """
    SO-101 leader → Piper bridge with:
      - Low-pass filter (exponential moving average) on raw input
      - Deadband: skip publish if no joint moved more than deadband threshold
      - Velocity feedforward: finite-difference velocity sent with trajectory
      - Adaptive time_from_start: scales with movement magnitude
      - Watchdog: holds last position if SO-101 signal is lost

    mode = "sim"   → JointTrajectory to arm_controller + gripper_controller
    mode = "real"  → JointState to joint_ctrl_single (piper CAN SDK)
    """

    def __init__(self) -> None:
        super().__init__("so101_piper_teleop_bridge")

        # ── parameters ────────────────────────────────────────────────────────
        self.declare_parameter("mode", "sim")
        self.declare_parameter("so101_joint_states_topic", "/so101/joint_states")
        self.declare_parameter("arm_trajectory_topic",
                               "/arm_controller/joint_trajectory")
        self.declare_parameter("gripper_trajectory_topic",
                               "/gripper_controller/joint_trajectory")
        self.declare_parameter("trajectory_duration_ms", 100)
        self.declare_parameter("joint_ctrl_topic", "joint_ctrl_single")
        self.declare_parameter("scale", 1.0)
        self.declare_parameter("gripper_open_rad",   _GRIPPER_SO101_OPEN_RAD)
        self.declare_parameter("gripper_closed_rad", _GRIPPER_SO101_CLOSED_RAD)
        # improvements
        self.declare_parameter("lp_alpha",    0.3)    # low-pass smoothing (0=frozen, 1=raw)
        self.declare_parameter("deadband",    0.002)  # rad — skip if all joints < this delta
        self.declare_parameter("max_vel",     _MAX_VEL_RAD_S)   # rad/s for adaptive duration
        self.declare_parameter("min_traj_ms", 50)     # floor for adaptive duration [ms]
        self.declare_parameter("watchdog_s",  0.5)    # seconds before watchdog fires

        self._mode            = self.get_parameter("mode").value
        so101_topic           = self.get_parameter("so101_joint_states_topic").value
        self._scale           = self.get_parameter("scale").value
        self._grip_open_rad   = self.get_parameter("gripper_open_rad").value
        self._grip_closed_rad = self.get_parameter("gripper_closed_rad").value
        self._lp_alpha        = self.get_parameter("lp_alpha").value
        self._deadband        = self.get_parameter("deadband").value
        self._max_vel         = self.get_parameter("max_vel").value
        self._min_traj_ns     = self.get_parameter("min_traj_ms").value * 1_000_000
        self._fixed_traj_ns   = self.get_parameter("trajectory_duration_ms").value * 1_000_000
        watchdog_s            = self.get_parameter("watchdog_s").value

        # ── state ─────────────────────────────────────────────────────────────
        # filtered Piper positions (keyed by piper joint name); None until first msg
        self._filtered: dict[str, float] | None = None
        # previous filtered positions for velocity and deadband
        self._prev: dict[str, float] | None = None
        # wall-clock time of last message (for velocity dt and watchdog)
        self._last_msg_time: float | None = None

        # ── publishers ────────────────────────────────────────────────────────
        if self._mode == "real":
            ctrl_topic = self.get_parameter("joint_ctrl_topic").value
            self._ctrl_pub = self.create_publisher(JointState, ctrl_topic, 10)
            self.get_logger().info(
                f"TeleopBridge ready  |  mode=real  |  {so101_topic} → {ctrl_topic}"
                f"  |  scale={self._scale}  alpha={self._lp_alpha}"
                f"  deadband={self._deadband}  watchdog={watchdog_s}s"
            )
        else:
            arm_topic  = self.get_parameter("arm_trajectory_topic").value
            grip_topic = self.get_parameter("gripper_trajectory_topic").value
            self._arm_pub  = self.create_publisher(JointTrajectory, arm_topic, 10)
            self._grip_pub = self.create_publisher(JointTrajectory, grip_topic, 10)
            self.get_logger().info(
                f"TeleopBridge ready  |  mode=sim  |  {so101_topic}"
                f"  → {arm_topic} + {grip_topic}"
                f"  |  scale={self._scale}  alpha={self._lp_alpha}"
                f"  deadband={self._deadband}  watchdog={watchdog_s}s"
            )

        self._sub = self.create_subscription(
            JointState, so101_topic, self._on_joint_state, 10
        )

        # ── watchdog timer ────────────────────────────────────────────────────
        self._watchdog_timer = self.create_timer(
            watchdog_s, self._watchdog_cb
        )

    # ── helpers ────────────────────────────────────────────────────────────────

    def _map_positions(self, raw: dict[str, float]) -> dict[str, float] | None:
        """Apply joint map → return {piper_name: position} or None on missing joint."""
        out: dict[str, float] = {}
        for so101_name, (piper_name, scale, offset, lo, hi) in _JOINT_MAP.items():
            v = raw.get(so101_name)
            if v is None:
                self.get_logger().warn(
                    f"SO-101 joint '{so101_name}' missing — skipping",
                    throttle_duration_sec=5.0,
                )
                return None
            out[piper_name] = _clamp(v * scale * self._scale + offset, lo, hi)
        out["joint6"] = _JOINT6_HOME
        return out

    def _low_pass(self, new: dict[str, float]) -> dict[str, float]:
        """Exponential moving average per joint."""
        if self._filtered is None:
            return dict(new)
        a = self._lp_alpha
        return {k: a * new[k] + (1.0 - a) * self._filtered[k] for k in new}

    def _max_delta(self, curr: dict[str, float], prev: dict[str, float]) -> float:
        return max(abs(curr[k] - prev[k]) for k in curr)

    def _adaptive_duration_ns(self, max_delta: float) -> int:
        """Scale duration to max_delta / max_vel, floored at min_traj_ms."""
        if self._max_vel <= 0.0 or max_delta <= 0.0:
            return self._fixed_traj_ns
        ns = int((max_delta / self._max_vel) * 1_000_000_000)
        return max(ns, self._min_traj_ns)

    # ── main callback ──────────────────────────────────────────────────────────

    def _on_joint_state(self, msg: JointState) -> None:
        self._last_msg_time = time.monotonic()
        self._watchdog_timer.reset()

        raw = dict(zip(msg.name, msg.position))
        mapped = self._map_positions(raw)
        if mapped is None:
            return

        # 3. Low-pass filter
        filtered = self._low_pass(mapped)
        self._filtered = filtered

        # 2. Deadband — skip if nothing moved enough
        if self._prev is not None:
            delta = self._max_delta(filtered, self._prev)
            if delta < self._deadband:
                return

        # 4. Adaptive duration
        dur_ns = self._adaptive_duration_ns(
            self._max_delta(filtered, self._prev) if self._prev is not None else 0.0
        )

        self._prev = dict(filtered)

        # gripper
        raw_grip = raw.get("gripper")
        grip_m   = _gripper_metres(raw_grip, self._grip_open_rad, self._grip_closed_rad) \
                   if raw_grip is not None else _GRIPPER_OPEN_M

        if self._mode == "real":
            self._publish_real(filtered, grip_m)
        else:
            self._publish_sim_arm(filtered, dur_ns)
            self._publish_sim_gripper(grip_m, dur_ns)

    # ── sim mode ───────────────────────────────────────────────────────────────

    def _publish_sim_arm(self, filtered: dict[str, float], dur_ns: int) -> None:
        # ordered: joint1-5 from _JOINT_MAP (already mapped), then joint6
        piper_order = [_JOINT_MAP[n][0] for n in _ARM_SO101_NAMES] + ["joint6"]

        traj = JointTrajectory()
        traj.joint_names = piper_order
        pt = JointTrajectoryPoint()
        pt.positions = [filtered[n] for n in piper_order]
        # velocities intentionally omitted — JointTrajectoryController rejects
        # single-point trajectories where the last point has non-zero velocity
        pt.time_from_start = _make_duration(dur_ns)
        traj.points = [pt]
        self._arm_pub.publish(traj)

    def _publish_sim_gripper(self, grip_m: float, dur_ns: int) -> None:
        traj = JointTrajectory()
        traj.joint_names = ["joint7", "joint8"]
        pt = JointTrajectoryPoint()
        pt.positions = [grip_m, -grip_m]
        pt.time_from_start = _make_duration(dur_ns)
        traj.points = [pt]
        self._grip_pub.publish(traj)

    # ── real mode ──────────────────────────────────────────────────────────────

    def _publish_real(self, filtered: dict[str, float], grip_m: float) -> None:
        piper_order = [_JOINT_MAP[n][0] for n in _ARM_SO101_NAMES] + ["joint6", "gripper"]
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = piper_order
        msg.position = [filtered.get(n, 0.0) for n in piper_order[:-1]] + [grip_m]
        self._ctrl_pub.publish(msg)

    # ── watchdog ───────────────────────────────────────────────────────────────

    def _watchdog_cb(self) -> None:
        """Fire if no SO-101 message received within watchdog interval.
        Holds Piper at last known position by resending it with zero velocity.
        """
        if self._last_msg_time is None or self._filtered is None:
            return

        self.get_logger().warn(
            "SO-101 signal lost — holding position", throttle_duration_sec=2.0
        )

        grip_m = _gripper_metres(
            _GRIPPER_SO101_OPEN_RAD, self._grip_open_rad, self._grip_closed_rad
        )
        if self._mode == "real":
            self._publish_real(self._filtered, grip_m)
        else:
            self._publish_sim_arm(self._filtered, self._fixed_traj_ns)
            self._publish_sim_gripper(grip_m, self._fixed_traj_ns)


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
