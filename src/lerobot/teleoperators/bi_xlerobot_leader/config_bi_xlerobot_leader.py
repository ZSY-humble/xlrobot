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

from dataclasses import dataclass

from lerobot.teleoperators.so_leader import SOLeaderConfig

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_xlerobot_leader")
@dataclass
class BiXLeRobotLeaderConfig(TeleoperatorConfig):
    """XLeRobot 双臂主臂遥操作配置。

    使用两个 SO101 主臂（Feetech sts3215 电机），
    归一化模式与 XLeRobot 从臂一致（use_degrees=False），
    feature 名称使用 XLeRobot 命名约定（left_arm_/right_arm_ 前缀）。
    """

    left_arm_config: SOLeaderConfig
    right_arm_config: SOLeaderConfig
