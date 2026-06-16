import math
import time

import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# (piper_joint, scale, offset_rad, piper_min, piper_max)
# Offsets corrected for new SO-101 calibration (2026-06-16):
#   shoulder_pan  homing 1596→1575 (Δ−21 steps = +0.032 rad on raw)
#   shoulder_lift homing −1346→−1239 (Δ+107 steps = −0.164 rad on raw)
#   elbow_flex    homing 1636→1562 (Δ−74 steps = +0.114 rad on raw)
#   wrist_flex    homing 1785→1756 (Δ−29 steps = +0.044 rad on raw)
#   wrist_roll    homing 1851→1893 (Δ+42 steps = −0.064 rad on raw)
_JOINT_MAP: dict[str, tuple[str, float, float, float, float]] = {
    "shoulder_pan":  ("joint1", -1.0,  0.032,                    -2.618,  2.168),
    "shoulder_lift": ("joint2",  1.0,  math.pi / 2 + 0.164,       0.0,    3.14),
    "elbow_flex":    ("joint3",  1.0, -math.pi / 4 - 0.114,      -2.967,  0.0),
    "wrist_flex":    ("joint5",  1.0, -0.044,                     -1.22,   1.22),
    "wrist_roll":    ("joint4", -1.0, -0.064,                     -1.745,  1.745),
}

_JOINT6_HOME      = 0.0
_GRIPPER_OPEN_M   = 0.0
_GRIPPER_CLOSED_M = 0.035
_GRIPPER_PIPER_MAX_M = 0.08
# Gripper range derived from new calibration (range_min=1679, range_max=3056):
#   open  = (1679−2048)×2π/4096 = −0.566 rad  → use −0.57 (at hardware open limit)
#   closed = (3056−2048)×2π/4096 = +1.545 rad  → use +1.54 (at hardware close limit)
_GRIPPER_SO101_OPEN_RAD   = -0.57
_GRIPPER_SO101_CLOSED_RAD = +1.54

# Names in publish order (arm joints only, without joint6/gripper)
_ARM_SO101_NAMES = list(_JOINT_MAP.keys())
_PIPER_ARM_ORDER = [_JOINT_MAP[n][0] for n in _ARM_SO101_NAMES] + ["joint6"]
_PIPER_REAL_ORDER = _PIPER_ARM_ORDER + ["gripper"]
_PIPER_LIMITS = {
    piper_name: (lo, hi)
    for _, (piper_name, _, _, lo, hi) in _JOINT_MAP.items()
}
_PIPER_LIMITS["joint6"] = (-3.14, 3.14)

# Max joint velocity used for adaptive time_from_start [rad/s]
_MAX_VEL_RAD_S = 1.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _make_duration(nanoseconds: int) -> Duration:
    d = Duration()
    d.sec = nanoseconds // 1_000_000_000
    d.nanosec = nanoseconds % 1_000_000_000
    return d


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _gripper_metres(raw: float, open_rad: float, closed_rad: float) -> float:
    norm = _clamp((raw - open_rad) / (closed_rad - open_rad), 0.0, 1.0)
    return _GRIPPER_OPEN_M + norm * (_GRIPPER_CLOSED_M - _GRIPPER_OPEN_M)


class TeleopBridge(Node):
    """
    Bridge SO-101 leader commands to Piper control interfaces.

    Features:
      - Low-pass filter (exponential moving average) on raw input
      - Deadband: skip publish if no joint moved more than deadband threshold
      - Real-mode speed limiting through Piper driver's velocity[6] convention
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
        self.declare_parameter("gripper_deadband_m", 0.0005)
        self.declare_parameter("real_speed_percent", 30)
        self.declare_parameter("safe_start", True)
        self.declare_parameter("piper_feedback_topic", "joint_states_feedback")

        self._mode            = self.get_parameter("mode").value
        so101_topic           = self.get_parameter("so101_joint_states_topic").value
        self._scale           = float(self.get_parameter("scale").value)
        self._grip_open_rad   = float(self.get_parameter("gripper_open_rad").value)
        self._grip_closed_rad = float(self.get_parameter("gripper_closed_rad").value)
        self._lp_alpha        = _clamp(float(self.get_parameter("lp_alpha").value), 0.0, 1.0)
        self._deadband        = float(self.get_parameter("deadband").value)
        self._max_vel         = float(self.get_parameter("max_vel").value)
        self._grip_deadband_m = float(self.get_parameter("gripper_deadband_m").value)
        self._real_speed_pct  = int(self.get_parameter("real_speed_percent").value)
        self._safe_start      = _as_bool(self.get_parameter("safe_start").value)
        self._min_traj_ns     = int(float(self.get_parameter("min_traj_ms").value) * 1_000_000)
        self._fixed_traj_ns   = int(float(self.get_parameter("trajectory_duration_ms").value) * 1_000_000)
        watchdog_s            = float(self.get_parameter("watchdog_s").value)
        feedback_topic        = self.get_parameter("piper_feedback_topic").value

        self._validate_parameters(watchdog_s)

        # ── state ─────────────────────────────────────────────────────────────
        # filtered Piper positions (keyed by piper joint name); None until first msg
        self._filtered: dict[str, float] | None = None
        # previous filtered positions for velocity and deadband
        self._prev: dict[str, float] | None = None
        self._prev_grip_m: float | None = None
        self._last_grip_m: float | None = None
        # wall-clock time of last message (for velocity dt and watchdog)
        self._last_msg_time: float | None = None
        self._feedback_positions: dict[str, float] | None = None
        self._safe_start_offsets: dict[str, float] | None = None
        self._safe_start_grip_offset_m = 0.0

        # ── publishers ────────────────────────────────────────────────────────
        if self._mode == "real":
            ctrl_topic = self.get_parameter("joint_ctrl_topic").value
            self._ctrl_pub = self.create_publisher(JointState, ctrl_topic, 10)
            self._feedback_sub = self.create_subscription(
                JointState, feedback_topic, self._on_piper_feedback, 10
            )
            self.get_logger().info(
                f"TeleopBridge ready  |  mode=real  |  {so101_topic} → {ctrl_topic}"
                f"  |  scale={self._scale}  alpha={self._lp_alpha}"
                f"  deadband={self._deadband}  watchdog={watchdog_s}s"
                f"  real_speed={self._real_speed_pct}%"
                f"  safe_start={self._safe_start}"
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

    def _validate_parameters(self, watchdog_s: float) -> None:
        errors: list[str] = []
        if self._mode not in {"sim", "real"}:
            errors.append("mode must be 'sim' or 'real'")
        if not 0.0 <= float(self._scale) <= 1.0:
            errors.append("scale must be between 0.0 and 1.0")
        if float(self._grip_closed_rad) <= float(self._grip_open_rad):
            errors.append("gripper_closed_rad must be greater than gripper_open_rad")
        if float(self._deadband) < 0.0:
            errors.append("deadband must be >= 0.0")
        if float(self._grip_deadband_m) < 0.0:
            errors.append("gripper_deadband_m must be >= 0.0")
        if float(self._max_vel) < 0.0:
            errors.append("max_vel must be >= 0.0")
        if self._min_traj_ns <= 0:
            errors.append("min_traj_ms must be > 0")
        if self._fixed_traj_ns <= 0:
            errors.append("trajectory_duration_ms must be > 0")
        if float(watchdog_s) <= 0.0:
            errors.append("watchdog_s must be > 0.0")
        if not 1 <= self._real_speed_pct <= 100:
            errors.append("real_speed_percent must be between 1 and 100")

        if errors:
            msg = "; ".join(errors)
            self.get_logger().fatal(f"Invalid TeleopBridge parameters: {msg}")
            raise ValueError(msg)

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

    def _on_piper_feedback(self, msg: JointState) -> None:
        self._feedback_positions = dict(zip(msg.name, msg.position))

    def _gripper_from_raw(self, raw: dict[str, float]) -> float:
        raw_grip = raw.get("gripper")
        if raw_grip is None:
            self.get_logger().warn(
                "SO-101 joint 'gripper' missing — keeping last gripper command",
                throttle_duration_sec=5.0,
            )
            return self._last_grip_m if self._last_grip_m is not None else _GRIPPER_OPEN_M
        return _gripper_metres(raw_grip, self._grip_open_rad, self._grip_closed_rad)

    def _apply_safe_start(
        self, mapped: dict[str, float], grip_m: float
    ) -> tuple[dict[str, float], float] | None:
        if self._mode != "real" or not self._safe_start:
            return mapped, grip_m

        if self._safe_start_offsets is None:
            feedback = self._feedback_positions
            missing = [
                name for name in _PIPER_REAL_ORDER
                if feedback is None or name not in feedback
            ]
            if missing:
                self.get_logger().warn(
                    "Waiting for Piper feedback before teleop safe-start: "
                    + ", ".join(missing),
                    throttle_duration_sec=2.0,
                )
                return None

            self._safe_start_offsets = {
                name: feedback[name] - mapped[name]
                for name in _PIPER_ARM_ORDER
            }
            self._safe_start_grip_offset_m = feedback["gripper"] - grip_m
            self.get_logger().info(
                "Safe-start captured current Piper pose; teleop is now relative"
            )

        adjusted = {
            name: _clamp(
                mapped[name] + self._safe_start_offsets[name],
                _PIPER_LIMITS[name][0],
                _PIPER_LIMITS[name][1],
            )
            for name in _PIPER_ARM_ORDER
        }
        adjusted_grip = _clamp(
            grip_m + self._safe_start_grip_offset_m,
            _GRIPPER_OPEN_M,
            _GRIPPER_PIPER_MAX_M,
        )
        return adjusted, adjusted_grip

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
        raw = dict(zip(msg.name, msg.position))
        mapped = self._map_positions(raw)
        if mapped is None:
            return

        grip_m = self._gripper_from_raw(raw)
        safe_started = self._apply_safe_start(mapped, grip_m)
        if safe_started is None:
            return
        mapped, grip_m = safe_started

        self._last_msg_time = time.monotonic()
        self._watchdog_timer.reset()

        # 3. Low-pass filter
        filtered = self._low_pass(mapped)
        self._filtered = filtered

        # 2. Deadband — skip only if neither arm nor gripper moved enough
        if self._prev is not None:
            arm_delta = self._max_delta(filtered, self._prev)
            grip_delta = (
                abs(grip_m - self._prev_grip_m)
                if self._prev_grip_m is not None
                else 0.0
            )
            if arm_delta < self._deadband and grip_delta < self._grip_deadband_m:
                return

        # 4. Adaptive duration
        dur_ns = self._adaptive_duration_ns(
            self._max_delta(filtered, self._prev) if self._prev is not None else 0.0
        )

        self._prev = dict(filtered)
        self._prev_grip_m = grip_m
        self._last_grip_m = grip_m

        if self._mode == "real":
            self._publish_real(filtered, grip_m)
        else:
            self._publish_sim_arm(filtered, dur_ns)
            self._publish_sim_gripper(grip_m, dur_ns)

    # ── sim mode ───────────────────────────────────────────────────────────────

    def _publish_sim_arm(self, filtered: dict[str, float], dur_ns: int) -> None:
        # ordered: joint1-5 from _JOINT_MAP (already mapped), then joint6
        traj = JointTrajectory()
        traj.joint_names = _PIPER_ARM_ORDER
        pt = JointTrajectoryPoint()
        pt.positions = [filtered[n] for n in _PIPER_ARM_ORDER]
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
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = _PIPER_REAL_ORDER
        msg.position = [filtered.get(n, 0.0) for n in _PIPER_ARM_ORDER] + [grip_m]
        # Piper's driver uses velocity[6] as a global speed percentage.
        msg.velocity = [0.0] * 6 + [float(self._real_speed_pct)]
        self._ctrl_pub.publish(msg)

    # ── watchdog ───────────────────────────────────────────────────────────────

    def _watchdog_cb(self) -> None:
        """
        Fire if no SO-101 message received within watchdog interval.

        Holds Piper at the last known arm and gripper command.
        """
        if self._last_msg_time is None or self._filtered is None:
            return

        self.get_logger().warn(
            "SO-101 signal lost — holding position", throttle_duration_sec=2.0
        )

        grip_m = self._last_grip_m if self._last_grip_m is not None else _GRIPPER_OPEN_M
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
