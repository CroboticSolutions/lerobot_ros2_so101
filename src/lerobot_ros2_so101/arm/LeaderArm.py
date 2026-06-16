from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.motors.motors_bus import MotorCalibration
from lerobot.motors import Motor, MotorNormMode

from pathlib import Path

import draccus


CALIBRATION_PATH = Path(
    "~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/lerobot_leader.json"
).expanduser()


class LeaderArm:
    def __init__(self, port=None, calibration_path: str | None = None):
        self.port = port
        self.calibration_path = (
            Path(calibration_path).expanduser()
            if calibration_path
            else CALIBRATION_PATH
        )
        self.calibration = self._load_calibration()

    def _load_calibration(self) -> dict[str, MotorCalibration]:
        if not self.calibration_path.is_file():
            return {}

        with open(self.calibration_path) as f, draccus.config_type("json"):
            return draccus.load(dict[str, MotorCalibration], f)

    def _save_calibration(self) -> None:
        self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.calibration_path, "w") as f, draccus.config_type("json"):
            draccus.dump(self.calibration, f, indent=4)

    @staticmethod
    def _motors_look_calibrated(calibration: dict[str, MotorCalibration]) -> bool:
        if not calibration:
            return False

        for cal in calibration.values():
            if cal.range_min == 0 and cal.range_max >= 4095:
                return False

        return True

    def _ensure_calibration(self) -> None:
        if self.calibration and self.bus.is_calibrated:
            return

        if not self.calibration:
            motor_calibration = self.bus.read_calibration()
            if self._motors_look_calibrated(motor_calibration):
                self.calibration = motor_calibration
                self.bus.write_calibration(self.calibration)
                self._save_calibration()
                return

        if self.calibration and not self.bus.is_calibrated:
            self.bus.write_calibration(self.calibration)
            return

        if not self.calibration or not self.bus.is_calibrated:
            raise RuntimeError(
                "SO-101 leader calibration file is missing and motors are not calibrated. "
                f"Run calibration, then retry:\n"
                f"  cd {Path(__file__).resolve().parents[4]}\n"
                f"  uv run -- python -m lerobot.calibrate "
                f"--teleop.id=lerobot_leader --teleop.type=so101_leader "
                f"--teleop.port={self.port}\n"
                f"Expected calibration file: {self.calibration_path}"
            )

    def connect(self) -> None:
        self.bus = FeetechMotorsBus(
            port=self.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_0_100),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_0_100),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_0_100),
                "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_0_100),
                "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_0_100),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

        self.bus.connect()
        self._ensure_calibration()

        # SO101Leader.configure()
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def read_present_position(self) -> dict[str, float]:
        # SO101Leader.get_action()
        return self.bus.sync_read("Present_Position", normalize=False)
