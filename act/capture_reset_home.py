#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""采集并保存 XLeRobot 右臂与头部固定复位姿态。

用法：
  1. 先通过遥操作或手动流程把右臂、头部相机放到希望回到的初始姿态
  2. 运行本脚本保存当前右臂 6 关节位置 + 头部 2 关节位置
  3. record_self_teleop.py 会自动读取这个 JSON 作为 reset home

示例：
  python act/capture_reset_home.py
  python act/capture_reset_home.py --output act/config/reset_home.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# 兜底：允许 `python act/capture_reset_home.py` 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lerobot.robots.xlerobot_2wheels import XLerobot2Wheels, XLerobot2WheelsConfig
from lerobot.utils.utils import init_logging

from act.config import CONFIG
from act.mirror import extract_head, extract_right_arm


def _build_robot() -> XLerobot2Wheels:
    cfg = XLerobot2WheelsConfig(
        port1=CONFIG.follower_port1,
        port2=CONFIG.follower_port2,
        id=CONFIG.follower_id,
        cameras={},
        max_relative_target=None,
    )
    return XLerobot2Wheels(cfg)


def _disconnect_quietly(robot: XLerobot2Wheels) -> None:
    try:
        robot.disconnect()
    except Exception as exc:
        logging.warning(f"断开机器人时忽略异常：{exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="保存 XLeRobot 右臂 + 头部固定 reset home")
    parser.add_argument("--output", type=Path, default=CONFIG.reset_home_path, help="输出 JSON 路径")
    args = parser.parse_args()

    init_logging()
    CONFIG.banner("采集右臂 + 头部 reset home")
    print("请确认右臂、头部相机已经处在每段采集希望回到的初始姿态。")
    print("本脚本只读取当前位置，不会关闭扭矩，也不会主动移动电机。")
    print()
    input("确认姿态正确后按回车读取并保存 ... ")

    robot = _build_robot()
    try:
        robot.connect()
        obs = robot.get_observation()
        right_home = extract_right_arm(obs)
        head_home = extract_head(obs)
    finally:
        _disconnect_quietly(robot)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "robot_id": CONFIG.follower_id,
        "robot_type": "xlerobot_2wheels",
        "right_home": {key: float(value) for key, value in right_home.items()},
        "head_home": {key: float(value) for key, value in head_home.items()},
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n✅ 已保存右臂 + 头部 reset home：")
    print(f"   {args.output}")
    print("  right_home:")
    for key, value in payload["right_home"].items():
        print(f"    {key}: {value:.3f}")
    print("  head_home:")
    for key, value in payload["head_home"].items():
        print(f"    {key}: {value:.3f}")
    print("\n采集脚本会默认读取该文件：")
    print("   python act/record_self_teleop.py --num-episodes 3 --task \"...\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
