#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""act/calibrate_xlerobot.py — XLeRobot ACT 左右臂标定。

🎯 默认只标定 ACT 需要的左右两条机械臂。

头部和底盘不参与 ACT 自我遥操作采集；本脚本会保留已有头部/底盘标定，
或读取电机当前参数补齐标定文件，避免 LeRobot 连接时反复触发整机标定。

使用：
    python act/calibrate_xlerobot.py
    python act/calibrate_xlerobot.py --head-only
    python act/calibrate_xlerobot.py --full-body

按终端提示，把左臂和右臂分别推到中位与完整活动范围即可。
标定结果保存到：
    ~/.cache/huggingface/lerobot/calibration/robots/xlerobot_2wheels/<follower_id>.json
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# 兜底：允许 `python act/calibrate_xlerobot.py` 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lerobot.motors import MotorCalibration
from lerobot.motors.feetech import OperatingMode
from lerobot.robots.xlerobot_2wheels import XLerobot2Wheels, XLerobot2WheelsConfig
from lerobot.utils.utils import enter_pressed

from act.config import CONFIG

ENCODER_MAX = 4095
DEFAULT_EDGE_GUARD = 128
DEFAULT_CENTER_WINDOW = 2200
DISPLAY_PERIOD_S = 0.2


def _default_calibration(motor_id: int) -> MotorCalibration:
    """给非 ACT 电机兜底用的全量程标定。"""
    return MotorCalibration(
        id=motor_id,
        drive_mode=0,
        homing_offset=0,
        range_min=0,
        range_max=4095,
    )


def _record_ranges_limited(
    bus,
    motors: list[str],
    *,
    edge_guard: int,
    center_window: int,
) -> tuple[dict[str, int], dict[str, int]]:
    """记录非连续关节范围，只接受围绕中位的连续编码窗口。"""
    start_positions = {
        name: int(value)
        for name, value in bus.sync_read("Present_Position", motors, normalize=False).items()
    }
    range_mins = start_positions.copy()
    range_maxes = start_positions.copy()
    ignored_counts = dict.fromkeys(motors, 0)
    low_limits = {
        name: max(edge_guard, pos - center_window) for name, pos in start_positions.items()
    }
    high_limits = {
        name: min(ENCODER_MAX - edge_guard, pos + center_window)
        for name, pos in start_positions.items()
    }
    print(
        "有效范围记录已开启：POS 后带 * 表示该样本被忽略；"
        f"edge_guard={edge_guard}, center_window=±{center_window}"
    )
    print("推完所有非连续关节后按回车停止记录。Ctrl+C 取消本次标定。")
    print("\033[2J\033[H", end="")

    while True:
        positions = {
            name: int(value)
            for name, value in bus.sync_read("Present_Position", motors, normalize=False).items()
        }

        for name, pos in positions.items():
            if pos < low_limits[name] or pos > high_limits[name]:
                ignored_counts[name] += 1
                continue
            range_mins[name] = min(range_mins[name], pos)
            range_maxes[name] = max(range_maxes[name], pos)

        print("\033[H", end="")
        print("---------------------------------------------------------------")
        print(f"{'NAME':<28} | {'MIN':>6} | {'POS':>6} | {'MAX':>6} | {'IGN':>5}")
        for name in motors:
            pos = positions[name]
            pos_text = f"{pos}"
            if pos < low_limits[name] or pos > high_limits[name]:
                pos_text = f"{pos}*"
            print(
                f"{name:<28} | {range_mins[name]:>6} | "
                f"{pos_text:>6} | {range_maxes[name]:>6} | {ignored_counts[name]:>5}"
            )
        print("* = 样本已忽略；按回车停止记录。Ctrl+C 取消。")
        print(" " * 80)

        if enter_pressed():
            break
        time.sleep(DISPLAY_PERIOD_S)

    same_min_max = [name for name in motors if range_mins[name] == range_maxes[name]]
    if same_min_max:
        raise RuntimeError(
            "以下关节没有记录到有效活动范围，请重新标定并推动它们："
            f"{same_min_max}"
        )

    return range_mins, range_maxes


def _merge_existing_or_current_calibration(robot: XLerobot2Wheels) -> dict[str, MotorCalibration]:
    """生成完整标定字典，优先已有文件，其次读取当前电机参数。"""
    calibration: dict[str, MotorCalibration] = dict(robot.calibration)

    for bus in (robot.bus1, robot.bus2):
        try:
            current = bus.read_calibration()
        except Exception as exc:  # noqa: BLE001 - 标定工具需要继续给出可用兜底
            print(f"⚠️  读取 {bus.port} 当前标定失败，将对非臂电机使用默认值：{exc}")
            current = {}

        for name, motor in bus.motors.items():
            if name not in calibration:
                calibration[name] = current.get(name, _default_calibration(motor.id))

    return calibration


def _calibrate_arm(
    *,
    title: str,
    bus,
    motors: list[str],
    edge_guard: int,
    center_window: int,
    full_turn_motors: list[str] | None = None,
) -> dict[str, MotorCalibration]:
    """标定一条机械臂，返回该臂 6 个电机的标定结果。"""
    print("\n" + "=" * 60)
    print(f"🎯 {title}")
    print("=" * 60)
    print("⚠️  即将关闭这条机械臂扭矩，请用手扶住机械臂。")
    input("准备好后按回车继续 ... ")

    bus.disable_torque(motors)
    for name in motors:
        bus.write("Operating_Mode", name, OperatingMode.POSITION.value)

    input("把这条机械臂移动到中位姿态，然后按回车记录中位 ... ")
    homing_offsets = bus.set_half_turn_homings(motors)

    if full_turn_motors is None:
        full_turn_motors = [name for name in motors if name.endswith("_wrist_roll")]
    range_motors = [name for name in motors if name not in full_turn_motors]

    if full_turn_motors:
        print(
            "现在依次缓慢推动这条机械臂的非连续旋转关节，覆盖后续任务会用到的完整范围。\n"
            f"连续旋转关节 {full_turn_motors} 不参与 min/max 记录，会直接设为 0..4095。\n"
            "不要硬顶机械限位；全部推完后按回车停止记录。"
        )
    else:
        print(
            "现在依次缓慢推动这些关节，覆盖后续任务会用到的完整范围。\n"
            "这些关节都会记录有限 min/max；不要硬顶机械限位。\n"
            "全部推完后按回车停止记录。"
        )
    range_mins, range_maxes = _record_ranges_limited(
        bus,
        range_motors,
        edge_guard=edge_guard,
        center_window=center_window,
    )
    for name in full_turn_motors:
        range_mins[name] = 0
        range_maxes[name] = ENCODER_MAX

    calibration: dict[str, MotorCalibration] = {}
    for name in motors:
        calibration[name] = MotorCalibration(
            id=bus.motors[name].id,
            drive_mode=0,
            homing_offset=int(homing_offsets[name]),
            range_min=int(range_mins[name]),
            range_max=int(range_maxes[name]),
        )

    bus.write_calibration(calibration, num_retry=5)
    print(f"✅ {title} 标定完成。")
    return calibration


def _calibrate_arms_only(*, edge_guard: int, center_window: int) -> int:
    """只标定左右臂；头部和两轮底盘不做人工标定。"""
    CONFIG.banner("XLeRobot ACT 左右臂标定")
    print("本流程只标定：")
    print("  1. 左臂 6 个电机")
    print("  2. 右臂 6 个电机")
    print()
    print("不会要求你手动标定头部和底盘。")
    print("头部/底盘字段会从已有标定或电机当前参数补齐到 JSON。")
    print(f"非连续关节记录范围会忽略靠近 0/4095 的样本：edge_guard={edge_guard}")
    print(f"非连续关节只记录中位附近连续窗口：center_window=±{center_window}")
    print()

    cfg = XLerobot2WheelsConfig(
        port1=CONFIG.follower_port1,
        port2=CONFIG.follower_port2,
        id=CONFIG.follower_id,
        cameras={},
    )
    robot = XLerobot2Wheels(cfg)

    try:
        robot.bus1.connect()
        robot.bus2.connect()

        calibration = _merge_existing_or_current_calibration(robot)
        calibration.update(
            _calibrate_arm(
                title="左臂标定（bus1 / ID 1-6）",
                bus=robot.bus1,
                motors=robot.left_arm_motors,
                edge_guard=edge_guard,
                center_window=center_window,
            )
        )
        calibration.update(
            _calibrate_arm(
                title="右臂标定（bus2 / ID 1-6）",
                bus=robot.bus2,
                motors=robot.right_arm_motors,
                edge_guard=edge_guard,
                center_window=center_window,
            )
        )

        robot.calibration = calibration
        robot._save_calibration()
        print("\n✅ ACT 左右臂标定完成。")
        print(f"   标定文件: {robot.calibration_fpath}")
        print("   下一步建议先跑：python act/teleoperate_self.py --no-cameras --no-display")
        return 0
    finally:
        for bus in (robot.bus1, robot.bus2):
            if bus.is_connected:
                bus.disconnect(disable_torque=False)


def _calibrate_head_only(*, edge_guard: int, center_window: int) -> int:
    """只标定头部两个电机，保留左右臂和底盘已有标定。"""
    CONFIG.banner("XLeRobot 头部相机单独标定")
    print("本流程只标定：")
    print("  1. head_motor_1 / ID 7")
    print("  2. head_motor_2 / ID 8")
    print()
    print("不会标定左右臂和底盘；它们的标定会从已有文件或当前电机参数保留。")
    print("头部两个电机都会按有限范围关节标定，不再把 head_motor_1 当 360 度连续轴。")
    print(f"头部范围记录会忽略靠近 0/4095 的样本：edge_guard={edge_guard}")
    print(f"头部只记录中位附近连续窗口：center_window=±{center_window}")
    print()

    cfg = XLerobot2WheelsConfig(
        port1=CONFIG.follower_port1,
        port2=CONFIG.follower_port2,
        id=CONFIG.follower_id,
        cameras={},
    )
    robot = XLerobot2Wheels(cfg)

    try:
        robot.bus1.connect()
        robot.bus2.connect()

        calibration = _merge_existing_or_current_calibration(robot)
        calibration.update(
            _calibrate_arm(
                title="头部相机标定（bus1 / ID 7-8）",
                bus=robot.bus1,
                motors=robot.head_motors,
                edge_guard=edge_guard,
                center_window=center_window,
                full_turn_motors=[],
            )
        )

        robot.calibration = calibration
        robot._save_calibration()
        print("\n✅ 头部相机标定完成。")
        print(f"   标定文件: {robot.calibration_fpath}")
        print("   下一步：先手动摆好右臂和头部视角，再运行 python act/capture_reset_home.py")
        print("   然后测试：python act/test_reset_home.py --dry-run")
        print("   确认误差正常后，再测试：python act/test_reset_home.py")
        return 0
    finally:
        for bus in (robot.bus1, robot.bus2):
            if bus.is_connected:
                bus.disconnect(disable_torque=False)


def _full_body_calibration() -> int:
    """调用 LeRobot 官方整机标定，包含头部和两轮底盘。"""
    CONFIG.banner("XLeRobot 两轮版整机标定")
    print("流程：\n"
          "  1. LeRobot 官方整机标定\n"
          "  2. 会覆盖左右臂、头部、两轮底盘对应的标定文件\n")

    cmd = [
        "lerobot-calibrate",
        "--robot.type=xlerobot_2wheels",
        f"--robot.port1={CONFIG.follower_port1}",
        f"--robot.port2={CONFIG.follower_port2}",
        f"--robot.id={CONFIG.follower_id}",
    ]
    print("$", " ".join(cmd), "\n")
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="XLeRobot ACT 左右臂标定")
    parser.add_argument(
        "--head-only",
        action="store_true",
        help="只标定头部两个电机（head_motor_1/2），保留左右臂和底盘标定",
    )
    parser.add_argument(
        "--full-body",
        action="store_true",
        help="调用 LeRobot 官方整机标定（左右臂 + 头部 + 两轮底盘）",
    )
    parser.add_argument(
        "--edge-guard",
        type=int,
        default=DEFAULT_EDGE_GUARD,
        help="非连续关节忽略 0/4095 两端多少编码值，默认 128",
    )
    parser.add_argument(
        "--center-window",
        type=int,
        default=DEFAULT_CENTER_WINDOW,
        help=f"非连续关节围绕中位允许记录的半窗口，默认 {DEFAULT_CENTER_WINDOW}",
    )
    args = parser.parse_args()

    try:
        if args.head_only and args.full_body:
            raise ValueError("--head-only 和 --full-body 不能同时使用")
        if args.head_only:
            return _calibrate_head_only(
                edge_guard=args.edge_guard,
                center_window=args.center_window,
            )
        if args.full_body:
            return _full_body_calibration()
        return _calibrate_arms_only(
            edge_guard=args.edge_guard,
            center_window=args.center_window,
        )
    except KeyboardInterrupt:
        print("\n🛑 已取消本次标定，未保存新的标定文件。")
        return 130


if __name__ == "__main__":
    sys.exit(main())
