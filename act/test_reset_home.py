#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试 XLeRobot 右臂 + 头部固定 reset home。

用途：
  读取 act/config/reset_home.json，让右臂和头部相机慢速回到该姿态。
  头部两个电机按有限范围关节位置控制，不再走 360 度连续轴速度控制。
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
from act.head_control import (
    choose_continuous_head_direction,
    continuous_head_errors_raw,
    head_position_targets_without_continuous,
    has_continuous_head_target,
    prepare_continuous_head_velocity,
    step_continuous_head_toward_home,
    stop_continuous_head_velocity,
)
from act.mirror import limit_action_relative_to_observation
from act.record_self_teleop import ACTION_NAMES, HEAD_NAMES


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


def _load_home(path: Path) -> tuple[dict[str, float], dict[str, float] | None]:
    if not path.exists():
        raise FileNotFoundError(f"reset home 文件不存在：{path}，请先运行 python act/capture_reset_home.py")

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_home = payload.get("right_home", payload)
    missing = [name for name in ACTION_NAMES if name not in raw_home]
    if missing:
        raise ValueError(f"reset home 文件缺少字段：{missing}，文件：{path}")
    right_home = {name: float(raw_home[name]) for name in ACTION_NAMES}

    raw_head_home = payload.get("head_home")
    head_home = None
    if raw_head_home is not None:
        missing_head = [name for name in HEAD_NAMES if name not in raw_head_home]
        if missing_head:
            raise ValueError(f"reset home 文件缺少头部字段：{missing_head}，文件：{path}")
        head_home = {name: float(raw_head_home[name]) for name in HEAD_NAMES}
    return right_home, head_home


def _print_error(obs: dict, right_home: dict[str, float], head_home: dict[str, float] | None) -> float:
    print("\n右臂关节误差：")
    max_error = 0.0
    for name in ACTION_NAMES:
        current = float(obs[name])
        target = float(right_home[name])
        error = target - current
        max_error = max(max_error, abs(error))
        print(f"  {name:32s} current={current:9.3f}  target={target:9.3f}  error={error:9.3f}")
    print(f"  right_max_abs_error = {max_error:.3f}")

    if head_home is None:
        print("\n头部误差：reset_home 未包含 head_home，跳过。")
        return max_error

    print("\n头部关节误差：")
    position_targets = head_position_targets_without_continuous(head_home) or {}
    for name in position_targets:
        current = float(obs[name])
        target = float(position_targets[name])
        error = target - current
        max_error = max(max_error, abs(error))
        print(f"  {name:32s} current={current:9.3f}  target={target:9.3f}  error={error:9.3f}")
    print(f"  position_max_abs_error = {max_error:.3f}")
    return max_error


def _max_continuous_head_raw_error(robot, head_home: dict[str, float] | None) -> int:
    if head_home is None:
        return 0
    raw_errors = continuous_head_errors_raw(robot, head_home)
    return max((abs(value) for value in raw_errors.values()), default=0)


def _print_continuous_head_raw_error(robot, head_home: dict[str, float] | None) -> int:
    if head_home is None:
        return 0
    raw_errors = continuous_head_errors_raw(robot, head_home)
    for motor, raw_error in raw_errors.items():
        print(f"  {motor}.raw_shortest_error = {raw_error:+d}")
    if raw_errors:
        print("  注：连续头部关节用 raw 最短误差判断是否到位，避免绕大圈。")
    return max((abs(value) for value in raw_errors.values()), default=0)


def _max_home_error(obs: dict, right_home: dict[str, float], head_home: dict[str, float] | None) -> float:
    right_error = max(abs(float(obs[name]) - float(right_home[name])) for name in ACTION_NAMES)
    if head_home is None:
        return right_error
    position_targets = head_position_targets_without_continuous(head_home)
    if position_targets is None:
        return right_error
    head_error = max(abs(float(obs[name]) - float(position_targets[name])) for name in position_targets)
    return max(right_error, head_error)


def _max_right_error(obs: dict, right_home: dict[str, float]) -> float:
    return max(abs(float(obs[name]) - float(right_home[name])) for name in ACTION_NAMES)


def _merge_home_action(
    right_home: dict[str, float],
    head_home: dict[str, float] | None,
) -> dict[str, float]:
    action = dict(right_home)
    if head_home is not None:
        action.update(head_home)
    return action


def main() -> int:
    parser = argparse.ArgumentParser(description="测试右臂 + 头部固定 reset home")
    parser.add_argument("--home", type=Path, default=CONFIG.reset_home_path, help="reset home JSON 路径")
    parser.add_argument("--hz", type=int, default=CONFIG.cam_fps, help="控制频率")
    parser.add_argument("--timeout", type=float, default=12.0, help="最多复位秒数")
    parser.add_argument("--tolerance", type=float, default=0.5, help="最大误差小于该值认为到位")
    head_group = parser.add_mutually_exclusive_group()
    head_group.add_argument(
        "--control-head-home",
        dest="control_head_home",
        action="store_true",
        default=True,
        help="同时控制头部回 head_home；默认开启",
    )
    head_group.add_argument(
        "--no-control-head-home",
        dest="control_head_home",
        action="store_false",
        help="只控制右臂，头部只打印误差",
    )
    parser.add_argument("--head-continuous-speed", type=int, default=80, help=argparse.SUPPRESS)
    parser.add_argument("--head-continuous-tolerance-raw", type=int, default=16, help=argparse.SUPPRESS)
    parser.add_argument(
        "--head-continuous-direction",
        choices=("auto", "-1", "1"),
        default="auto",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=CONFIG.max_relative_target,
        help="单步最大相对变化；默认 15.0，<=0 表示直接发送目标",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印当前误差，不发送动作")
    args = parser.parse_args()

    init_logging()
    right_home, head_home = _load_home(args.home)
    if args.control_head_home and head_home is None:
        raise RuntimeError(
            "当前启用了头部 home 控制，但 reset_home 文件没有 head_home。"
            "请先摆好右臂和头部相机，再运行：python act/capture_reset_home.py"
        )

    CONFIG.banner("测试右臂 + 头部 reset home")
    print(f"  home 文件   : {args.home}")
    print(f"  控制频率    : {args.hz} Hz")
    print(f"  timeout     : {args.timeout}s")
    print(f"  tolerance   : {args.tolerance}")
    print(f"  控制头部    : {args.control_head_home}")
    if args.control_head_home:
        print("  头部控制    : 有限范围位置控制（head_motor_1/2 都按标定范围限位）")
    print(f"  dry-run     : {args.dry_run}")
    print()
    if args.dry_run:
        print("⚠️  dry-run 只打印误差，不会移动右臂或头部。")
    elif args.control_head_home:
        print("⚠️  右臂和头部将移动到固定 reset home，请确认头部标定正常、路径无遮挡。")
    else:
        print("⚠️  只控制右臂回 reset home；头部只打印误差，不会主动移动。")
    input("确认安全后按回车继续 ... ")

    robot = _build_robot()
    try:
        robot.connect()
        obs = robot.get_observation()
        print("\n当前位置与 home 对比：")
        error = _print_error(obs, right_home, head_home)
        raw_error = _print_continuous_head_raw_error(robot, head_home)

        if args.dry_run:
            return 0

        continuous_head_enabled = args.control_head_home and has_continuous_head_target(head_home)
        continuous_ok = (not continuous_head_enabled) or raw_error <= args.head_continuous_tolerance_raw
        if error <= args.tolerance and continuous_ok:
            print("\n✅ 当前已经在 reset home 附近，无需移动。")
            return 0

        period = 1.0 / max(args.hz, 1)
        step_limit = None if args.max_relative_target <= 0 else args.max_relative_target
        t_start = time.perf_counter()
        head_direction = -1
        if continuous_head_enabled:
            prepare_continuous_head_velocity(robot, head_home)
            if args.head_continuous_direction == "auto":
                head_direction = choose_continuous_head_direction(
                    robot,
                    head_home,
                    speed_raw=args.head_continuous_speed,
                    tolerance_raw=args.head_continuous_tolerance_raw,
                )
            else:
                head_direction = int(args.head_continuous_direction)
            logging.info("head_motor_1 本次使用方向符号：%+d", head_direction)

        try:
            while True:
                t0 = time.perf_counter()
                obs = robot.get_observation()
                error = (
                    _max_home_error(obs, right_home, head_home)
                    if args.control_head_home
                    else _max_right_error(obs, right_home)
                )
                raw_errors = {}
                if continuous_head_enabled:
                    raw_errors = step_continuous_head_toward_home(
                        robot,
                        head_home,
                        speed_raw=args.head_continuous_speed,
                        tolerance_raw=args.head_continuous_tolerance_raw,
                        direction_sign=head_direction,
                    )
                raw_error = max((abs(v) for v in raw_errors.values()), default=0)
                continuous_ok = (not continuous_head_enabled) or raw_error <= args.head_continuous_tolerance_raw
                if error <= args.tolerance and continuous_ok:
                    break

                right_action = (
                    right_home
                    if step_limit is None
                    else limit_action_relative_to_observation(right_home, obs, step_limit)
                )
                action = _merge_home_action(
                    right_action,
                    head_position_targets_without_continuous(head_home) if args.control_head_home else None,
                )
                robot.send_action(action)

                if (time.perf_counter() - t_start) >= args.timeout:
                    logging.warning("复位超时：当前最大误差 %.3f", error)
                    break

                elapsed = time.perf_counter() - t0
                precise_sleep(max(period - elapsed, 0.0))
        finally:
            if continuous_head_enabled:
                stop_continuous_head_velocity(robot, head_home)

        obs = robot.get_observation()
        final_error = _print_error(obs, right_home, head_home)
        final_raw_error = _print_continuous_head_raw_error(robot, head_home)
        pass_error = (
            final_error
            if args.control_head_home
            else _max_right_error(obs, right_home)
        )
        final_continuous_ok = (
            not continuous_head_enabled
            or final_raw_error <= args.head_continuous_tolerance_raw
        )
        if pass_error <= args.tolerance and final_continuous_ok:
            if args.control_head_home:
                print("\n✅ reset home 测试通过：右臂和头部已到达固定复位姿态。")
            else:
                print("\n✅ reset home 测试通过：右臂已到达固定复位姿态；头部未控制。")
            return 0

        print("\n⚠️  reset home 未完全到位，请检查 home 姿态、限幅、机械阻挡或标定。")
        return 2
    finally:
        _disconnect_quietly(robot)
        print("✅ 已安全断开。")


if __name__ == "__main__":
    sys.exit(main())
