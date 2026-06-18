#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
from functools import cached_property

from lerobot.teleoperators.so_leader import SOLeaderTeleopConfig
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..so_leader import SOLeader
from ..teleoperator import Teleoperator
from .config_bi_xlerobot_leader import BiXLeRobotLeaderConfig

logger = logging.getLogger(__name__)


class BiXLeRobotLeader(Teleoperator):
    """XLeRobot 双臂主臂遥操作器。

    使用两个 SO101 主臂（Feetech sts3215 电机），与 XLeRobot 从臂匹配：
    - 归一化模式：use_degrees=False（RANGE_M100_100），和 XLeRobot 一致
    - Feature 名称：left_arm_/right_arm_ 前缀，和 XLeRobot 一致

    输出 action feature 示例：
        left_arm_shoulder_pan.pos, left_arm_shoulder_lift.pos, ...
        right_arm_shoulder_pan.pos, right_arm_shoulder_lift.pos, ...
    """

    config_class = BiXLeRobotLeaderConfig
    name = "bi_xlerobot_leader"

    def __init__(self, config: BiXLeRobotLeaderConfig):
        super().__init__(config)
        self.config = config

        # 使用 use_degrees=False 创建子臂，与 XLeRobot 从臂归一化模式一致
        left_arm_config = SOLeaderTeleopConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_config.port,
            use_degrees=False,
        )

        right_arm_config = SOLeaderTeleopConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_config.port,
            use_degrees=False,
        )

        self.left_arm = SOLeader(left_arm_config)
        self.right_arm = SOLeader(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        """返回与 XLeRobot action_features 一致的 feature 名称。"""
        left_arm_features = self.left_arm.action_features
        right_arm_features = self.right_arm.action_features

        return {
            # left_shoulder_pan.pos → left_arm_shoulder_pan.pos
            **{f"left_arm_{k}": v for k, v in left_arm_features.items()},
            # right_shoulder_pan.pos → right_arm_shoulder_pan.pos
            **{f"right_arm_{k}": v for k, v in right_arm_features.items()},
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        """读取双臂关节位置，返回与 XLeRobot 一致的 feature 名称。"""
        action_dict = {}

        # 读取左臂，加 left_arm_ 前缀
        left_action = self.left_arm.get_action()
        action_dict.update({f"left_arm_{key}": value for key, value in left_action.items()})

        # 读取右臂，加 right_arm_ 前缀
        right_action = self.right_arm.get_action()
        action_dict.update({f"right_arm_{key}": value for key, value in right_action.items()})

        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO: 实现力反馈
        raise NotImplementedError

    @check_if_not_connected
    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()
