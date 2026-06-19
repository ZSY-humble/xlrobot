#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试 XLeRobot 右臂固定 reset home。

用途：
  读取 act/config/reset_home.json，让右臂慢速回到该姿态。
  本脚本不连相机、不录数据，只用于确认固定复位姿态是否正确。

示例：
  python act/test_reset_home.py
  python act/test_reset_home.py --dry-run
  python act/test_reset_home.py --tolerance 2.0 --timeout 12
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# 兜底：允许 `python act/test_reset_home.py` 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lerobot.robots.xlerobot_2wheels import XLerobot2Wheels, XLerobot2WheelsConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

from act.config import CONFIG
from act.mirror import extract_right_arm, limit_action_relative_to_observation
from act.record_self_teleop import ACTION_NAMES


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


def _load_home(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"reset home 文件不存在：{path}，请先运行 python act/capture_reset_home.py")

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_home = payload.get("right_home", payload)
    missing = [name for name in ACTION_NAMES if name not in raw_home]
    if missing:
        raise ValueError(f"reset home 文件缺少字段：{missing}，文件：{path}")
    return {name: float(raw_home[name]) for name in ACTION_NAMES}


def _print_error(obs: dict, home: dict[str, float]) -> float:
    print("\n关节误差：")
    max_error = 0.0
    for name in ACTION_NAMES:
        current = float(obs[name])
        target = float(home[name])
        error = target - current
        max_error = max(max_error, abs(error))
        print(f"  {name:32s} current={current:9.3f}  target={target:9.3f}  error={error:9.3f}")
    print(f"  max_abs_error = {max_error:.3f}")
    return max_error


def main() -> int:
    parser = argparse.ArgumentParser(description="测试右臂固定 reset home")
    parser.add_argument("--home", type=Path, default=CONFIG.reset_home_path, help="reset home JSON 路径")
    parser.add_argument("--hz", type=int, default=CONFIG.cam_fps, help="控制频率")
    parser.add_argument("--timeout", type=float, default=12.0, help="最多复位秒数")
    parser.add_argument("--tolerance", type=float, default=2.0, help="最大误差小于该值认为到位")
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=CONFIG.max_relative_target,
        help="单步最大相对变化；默认 10.0，<=0 表示直接发送目标",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印当前误差，不发送动作")
    args = parser.parse_args()

    init_logging()
    home = _load_home(args.home)

    CONFIG.banner("测试右臂 reset home")
    print(f"  home 文件   : {args.home}")
    print(f"  控制频率    : {args.hz} Hz")
    print(f"  timeout     : {args.timeout}s")
    print(f"  tolerance   : {args.tolerance}")
    print(f"  dry-run     : {args.dry_run}")
    print()
    print("⚠️  右臂将移动到固定 reset home，请确认路径无遮挡、不会撞桌子/相机/自己。")
    input("确认安全后按回车继续 ... ")

    robot = _build_robot()
    try:
        robot.connect()
        obs = robot.get_observation()
        print("\n当前位置与 home 对比：")
        error = _print_error(obs, home)

        if args.dry_run:
            return 0

        if error <= args.tolerance:
            print("\n✅ 当前已经在 reset home 附近，无需移动。")
            return 0

        period = 1.0 / max(args.hz, 1)
        step_limit = None if args.max_relative_target <= 0 else args.max_relative_target
        t_start = time.perf_counter()

        while True:
            t0 = time.perf_counter()
            obs = robot.get_observation()
            error = max(abs(float(obs[name]) - float(home[name])) for name in ACTION_NAMES)
            if error <= args.tolerance:
                break

            action = (
                home
                if step_limit is None
                else limit_action_relative_to_observation(home, obs, step_limit)
            )
            robot.send_action(action)

            if (time.perf_counter() - t_start) >= args.timeout:
                logging.warning("复位超时：当前最大误差 %.3f", error)
                break

            elapsed = time.perf_counter() - t0
            precise_sleep(max(period - elapsed, 0.0))

        obs = robot.get_observation()
        final_error = _print_error(obs, home)
        if final_error <= args.tolerance:
            print("\n✅ reset home 测试通过：右臂已到达固定复位姿态。")
            return 0

        print("\n⚠️  reset home 未完全到位，请检查 home 姿态、限幅、机械阻挡或标定。")
        return 2
    finally:
        _disconnect_quietly(robot)
        print("✅ 已安全断开。")


if __name__ == "__main__":
    sys.exit(main())
