#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import numpy as np
from rustypot import Scs0009PyController

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.robots.so_follower import SOFollower

from .config_so_amazing_hand import SO101AmazingHandFollowerConfig

logger = logging.getLogger(__name__)


class SO101AmazingHandFollower(SOFollower):
    """
    SO-101 follower arm where the original gripper (servo #6) has been replaced by an
    AmazingHand dexterous hand running on its own serial bus (e.g. /dev/ttyACM2).

    The leader arm's `gripper.pos` action is mapped to the AmazingHand open/close state
    using hysteresis thresholds.
    """

    config_class = SO101AmazingHandFollowerConfig
    name = "so101_amazing_hand"

    def __init__(self, config: SO101AmazingHandFollowerConfig):
        # Do NOT call SOFollower.__init__ because it creates a 6-motor bus including the
        # original gripper. Instead, initialise the Robot base and build a 5-motor arm bus
        # plus the AmazingHand controller ourselves.
        Robot.__init__(self, config)
        self.config = config

        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

        self.MiddlePos = list(config.hand_middle_pos)
        self.OpenAngles = list(config.hand_open_angles)
        self.CloseAngles = list(config.hand_close_angles)
        self.CloseSpeeds = list(config.hand_close_speeds)
        self.MaxSpeed = 7
        self.hand = Scs0009PyController(
            serial_port=config.hand_port,
            baudrate=1_000_000,
            timeout=0.5,
        )
        self.hand_state = "open"
        self._hand_torque_enabled = False
        # Last gripper command received, used for proportional control and observation.
        self._last_gripper_cmd = self.config.gripper_open_pos

    @property
    def _motors_ft(self) -> dict[str, type]:
        # Keep the 5 arm joints plus a virtual `gripper.pos` so the leader teleop loop
        # (which still emits `gripper.pos`) stays compatible.
        return {f"{motor}.pos": float for motor in self.bus.motors} | {"gripper.pos": float}

    def configure(self) -> None:
        # Configure the SO-101 arm motors (5 motors, no gripper).
        super().configure()

        # Enable torque on all 8 AmazingHand servos.
        logger.info(f"{self} enabling AmazingHand torque on {self.config.hand_port}")
        servo_ids = list(range(1, 9))
        self.hand.sync_write_torque_enable(servo_ids, [1] * 8)
        self._hand_torque_enabled = True

        # Start with an open hand.
        self.open_hand()
        self.hand_state = "open"

    def connect(self, calibrate: bool = True) -> None:
        # Connect/calibrate/configure the arm and cameras via SOFollower logic.
        super().connect(calibrate)
        # `configure()` is already called inside SOFollower.connect(), which in our case
        # also enables hand torque and opens the hand.

    def get_observation(self) -> RobotObservation:
        # Read the 5 arm joint positions.
        obs_dict = super().get_observation()

        # Expose the current hand command as a virtual `gripper.pos` observation.
        if self.config.hand_use_proportional_control:
            obs_dict["gripper.pos"] = float(self._last_gripper_cmd)
        else:
            # Binary mode: 100 = closed, 0 = open.
            obs_dict["gripper.pos"] = 100.0 if self.hand_state == "closed" else 0.0
        return obs_dict

    def send_action(self, action: RobotAction) -> RobotAction:
        # Split arm joints from the virtual gripper command.
        arm_action = {k: v for k, v in action.items() if k != "gripper.pos"}
        gripper_cmd = action.get("gripper.pos", None)

        # Send arm commands through the normal SOFollower path (handles safety limits).
        # Avoid calling sync_write with an empty dict, which would raise StopIteration.
        sent_arm_action = super().send_action(arm_action) if arm_action else {}

        # Map the leader gripper position to AmazingHand open/close.
        if gripper_cmd is not None:
            self._update_hand_from_gripper(gripper_cmd)

        # Return the action that was actually sent.
        sent_action = dict(sent_arm_action)
        if self.config.hand_use_proportional_control:
            sent_action["gripper.pos"] = float(self._last_gripper_cmd)
        else:
            sent_action["gripper.pos"] = 100.0 if self.hand_state == "closed" else 0.0
        return sent_action

    def disconnect(self):
        if not self.is_connected:
            return

        # Disable AmazingHand torque before closing the serial port.
        if self._hand_torque_enabled:
            try:
                servo_ids = list(range(1, 9))
                self.hand.sync_write_torque_enable(servo_ids, [0] * 8)
            except Exception as e:
                logger.warning(f"{self} failed to disable AmazingHand torque: {e}")
            self._hand_torque_enabled = False

        # Disconnect the arm bus and cameras.
        super().disconnect()

    # -------------------------------------------------------------------------
    # AmazingHand helpers
    # -------------------------------------------------------------------------

    def _update_hand_from_gripper(self, gripper_pos: float) -> None:
        """Map the leader gripper position to AmazingHand open/close or proportional pose."""
        self._last_gripper_cmd = gripper_pos

        if self.config.hand_use_proportional_control:
            ratio = self._gripper_ratio(gripper_pos)
            self._move_hand_proportional(ratio)
            self.hand_state = "closed" if ratio > 0.5 else "open"
            return

        # Binary mode with hysteresis.
        open_pos = self.config.gripper_open_pos
        close_pos = self.config.gripper_close_pos

        dist_to_open = abs(gripper_pos - open_pos)
        dist_to_close = abs(gripper_pos - close_pos)

        if self.hand_state == "open" and dist_to_close < dist_to_open:
            self.close_hand()
            self.hand_state = "closed"
        elif self.hand_state == "closed" and dist_to_open < dist_to_close:
            self.open_hand()
            self.hand_state = "open"

    def _gripper_ratio(self, gripper_pos: float) -> float:
        """Normalize gripper position to [0, 1] where 1 = fully closed."""
        open_pos = self.config.gripper_open_pos
        close_pos = self.config.gripper_close_pos
        denom = close_pos - open_pos
        if abs(denom) < 1e-6:
            return 0.0
        ratio = (gripper_pos - open_pos) / denom
        return max(0.0, min(1.0, ratio))

    def _move_hand_proportional(self, ratio: float) -> None:
        """Move the hand to a pose interpolated between open and close angles."""
        angles = [
            self.OpenAngles[i] + ratio * (self.CloseAngles[i] - self.OpenAngles[i])
            for i in range(8)
        ]
        # Use open speeds near open pose, close speeds near closed pose.
        speeds = [
            int(self.MaxSpeed + ratio * (self.CloseSpeeds[i] - self.MaxSpeed))
            for i in range(8)
        ]
        self._move_hand(angles, speeds)

    def open_hand(self) -> None:
        """Open all fingers of the AmazingHand."""
        speeds = [self.MaxSpeed] * 8
        self._move_hand(self.OpenAngles, speeds)

    def close_hand(self) -> None:
        """Close all fingers of the AmazingHand."""
        self._move_hand(self.CloseAngles, self.CloseSpeeds)

    def _move_hand(self, angles: list[float], speeds: list[int]) -> None:
        servo_ids = list(range(1, 9))
        positions = [np.deg2rad(self.MiddlePos[i] + angles[i]) for i in range(8)]
        self.hand.sync_write_goal_speed(servo_ids, speeds)
        self.hand.sync_write_goal_position(servo_ids, positions)
