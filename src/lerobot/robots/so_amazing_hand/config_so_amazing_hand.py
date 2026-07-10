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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class SOAmazingHandFollowerConfig:
    """Base configuration class for SO Follower arms with an AmazingHand dexterous hand as end-effector."""

    # Port to connect to the SO-101 arm (without the original gripper servo #6)
    port: str

    # Port to connect to the AmazingHand dexterous hand
    hand_port: str = "/dev/ttyACM2"

    # Hand side: 1 = right hand, 2 = left hand
    hand_side: int = 1

    # Mapping from the leader arm's `gripper.pos` (range [0, 100]) to the AmazingHand
    # open/close state. Adjust these values if your leader gripper is inverted.
    # Default assumes 0 = open and 100 = closed.
    gripper_open_pos: float = 0.0
    gripper_close_pos: float = 100.0

    # Default middle/zero positions for the 8 AmazingHand servos.
    # Edit these values to match your own mechanical calibration.
    hand_middle_pos: list[float] = field(
        default_factory=lambda: [3.0, 0.0, -5.0, -8.0, -2.0, 5.0, -12.0, 0.0]
    )

    # Open/close target angles (in degrees) for the 8 AmazingHand servos.
    # Servos are ordered: [index_1, index_2, middle_1, middle_2, ring_1, ring_2, thumb_1, thumb_2].
    # The last two values control the thumb. By default the thumb closes in the opposite
    # direction from the other fingers, which matches a natural pinch grasp.
    hand_open_angles: list[float] = field(
        default_factory=lambda: [90.0, -90.0, 90.0, -90.0, 90.0, -90.0, 90.0, -90.0]
    )
    hand_close_angles: list[float] = field(
        default_factory=lambda: [-35.0, 35.0, -35.0, 35.0, -35.0, 35.0, -35.0, 35.0]
    )
    # Close speed for each servo. Last two entries are the thumb and use CloseSpeed+1 in the original demo.
    hand_close_speeds: list[int] = field(
        default_factory=lambda: [3, 3, 3, 3, 3, 3, 4, 4]
    )

    # If True, the hand follows the leader gripper proportionally: gripper positions between
    # `gripper_open_pos` and `gripper_close_pos` are linearly interpolated between `hand_open_angles`
    # and `hand_close_angles`. If False, the hand only switches between fully open and fully closed
    # (binary behavior). Default is True for AmazingHand teleoperation.
    hand_use_proportional_control: bool = True

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False


@RobotConfig.register_subclass("so101_amazing_hand")
@dataclass
class SO101AmazingHandFollowerConfig(RobotConfig, SOAmazingHandFollowerConfig):
    pass
