#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""act/teleoperate_self.py — XLeRobot 自我遥操作（不录数据）。

🎯 验证模式：人手推动**左臂**（关扭矩） → **右臂**实时跟随（开扭矩） → 头部锁定。

用途：
  1. 第一次接入硬件验证扭矩控制是否正确
  2. 调试镜像表（哪些关节要取反）—— 配合 `python -m act.mirror probe`
  3. Rerun 里看 right_arm action 与 right_arm observation 是否对齐

使用：
    python act/teleoperate_self.py
    python act/teleoperate_self.py --no-display    # 不开 Rerun（headless）
    python act/teleoperate_self.py --hz 30          # 控制频率

退出：Ctrl+C
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# 兜底：允许 `python act/teleoperate_self.py` 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.xlerobot_2wheels import XLerobot2Wheels, XLerobot2WheelsConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from act.config import CONFIG
from act.mirror import (
    NEGATE_JOINTS,
    extract_left_arm,
    extract_right_arm,
    limit_action_relative_to_observation,
    mirror_left_delta_to_right_target,
    mirror_left_to_right,
)


def _build_camera_configs(use_cameras: bool):
    """构造从臂相机配置；不要相机则返回空 dict。"""
    if not use_cameras:
        return {}
    return {
        "top": OpenCVCameraConfig(
            index_or_path=CONFIG.cam_top,
            width=CONFIG.cam_width,
            height=CONFIG.cam_height,
            fps=CONFIG.cam_fps,
        ),
        "right_wrist": OpenCVCameraConfig(
            index_or_path=CONFIG.cam_right_wrist,
            width=CONFIG.cam_width,
            height=CONFIG.cam_height,
            fps=CONFIG.cam_fps,
        ),
    }


def _disconnect_quietly(robot: XLerobot2Wheels) -> None:
    try:
        robot.disconnect()
    except Exception as exc:
        logging.warning(f"断开机器人时忽略异常：{exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="XLeRobot 自我遥操作（左推右跟，不录数据）")
    parser.add_argument("--no-display", action="store_true", help="不启用 Rerun 可视化")
    parser.add_argument("--no-cameras", action="store_true", help="不连相机（更快启动）")
    parser.add_argument("--hz", type=int, default=30, help="控制循环频率（Hz）")
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="使用绝对位置镜像；默认使用 delta 变化量跟随，更适合首次真机验证",
    )
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=None,
        help="单次 action 相对当前位置最大变化；默认 10.0，可用 MAX_RELATIVE_TARGET 覆盖，<=0 关闭",
    )
    args = parser.parse_args()

    init_logging()
    max_relative_target = (
        args.max_relative_target
        if args.max_relative_target is not None
        else CONFIG.max_relative_target
    )
    if max_relative_target <= 0:
        max_relative_target = None

    CONFIG.banner("XLeRobot 自我遥操作（验证模式）")
    print(f"  控制频率   : {args.hz} Hz")
    print(f"  Rerun      : {not args.no_display}")
    print(f"  相机       : {'OFF' if args.no_cameras else 'top + right_wrist'}")
    print(f"  跟随模式   : {'absolute' if args.absolute else 'delta'}")
    print(f"  镜像取反   : {set(NEGATE_JOINTS)}")
    print()
    print("⚠️  即将关闭左臂扭矩 —— 请先用手扶住左臂！")
    print("   起始位建议：双臂立直、夹爪朝下、远离桌面")
    print()
    input("准备好按回车开始 ... ")

    # 1. 构造 XLeRobot
    xlerobot_cfg = XLerobot2WheelsConfig(
        port1=CONFIG.follower_port1,
        port2=CONFIG.follower_port2,
        id=CONFIG.follower_id,
        cameras=_build_camera_configs(use_cameras=not args.no_cameras),
        max_relative_target=None,
    )
    robot = XLerobot2Wheels(xlerobot_cfg)

    # 2. 连接（内部会启用所有扭矩；之后再单独关左臂）
    try:
        robot.connect()
    except Exception:
        _disconnect_quietly(robot)
        raise

    # 3. ⭐ 仅关左臂 6 个电机的扭矩；头部 + 右臂保持
    robot.bus1.disable_torque(robot.left_arm_motors)
    logging.info(f"✅ 左臂扭矩已关闭（{robot.left_arm_motors}），可用手推动")
    logging.info("ℹ️  头部 / 底盘扭矩保持开启，不主动控制（自然伺服锁位）")

    origin_obs = robot.get_observation()
    left_origin = extract_left_arm(origin_obs)
    right_origin = extract_right_arm(origin_obs)
    logging.info("✅ 已记录左右臂启动姿态，delta 模式会按变化量跟随")

    # 5. Rerun
    if not args.no_display:
        init_rerun(session_name="self_teleop")

    # 6. 主循环
    period = 1.0 / args.hz
    print("\n🚀 开始遥操作。Ctrl+C 退出。\n")
    try:
        while True:
            t0 = time.perf_counter()

            obs = robot.get_observation()
            left_pos = extract_left_arm(obs)
            if args.absolute:
                right_target = mirror_left_to_right(left_pos)
            else:
                right_target = mirror_left_delta_to_right_target(
                    left_current=left_pos,
                    left_origin=left_origin,
                    right_origin=right_origin,
                )
            action_to_send = limit_action_relative_to_observation(
                right_target,
                obs,
                max_relative_target,
            )

            # 仅控右臂；左臂 / 头部 / 底盘都不传 → send_action 不会动它们
            sent_action = robot.send_action(action_to_send)

            if not args.no_display:
                # log obs + sent_action 到 Rerun
                log_rerun_data(observation=obs, action=sent_action)

            elapsed = time.perf_counter() - t0
            precise_sleep(max(period - elapsed, 0.0))
    except KeyboardInterrupt:
        print("\n🛑 收到 Ctrl+C，断开机器人 ...")
    finally:
        _disconnect_quietly(robot)
        print("✅ 已安全断开。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
