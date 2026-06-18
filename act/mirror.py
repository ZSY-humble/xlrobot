#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""act/mirror.py — 左臂 → 右臂 关节角镜像映射。

🎯 自我遥操作的核心：人手推动左臂 → 读出左臂 6 个关节角 → 镜像变换 → 喂给右臂作目标位置。

由于 XLeRobot 左右臂物理上**镜像安装**，部分关节（典型为 shoulder_pan / wrist_roll）
方向相反，必须取反才能让右臂跟随得正确。

镜像表通过 **`NEGATE_JOINTS`** 一处管理；首次接入硬件后用：

    python -m act.mirror probe

逐关节交互式实测哪些关节要取反，并按提示更新本文件。

也可以通过环境变量临时覆盖（不改文件）：

    NEGATE_JOINTS="shoulder_pan,wrist_roll,gripper" python act/teleoperate_self.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# 兜底：允许 `python act/mirror.py probe` 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# 镜像表
# ============================================================

# 左→右 关节名映射（一一对应，6 对）
LEFT_TO_RIGHT_NAMES: dict[str, str] = {
    "left_arm_shoulder_pan.pos":   "right_arm_shoulder_pan.pos",
    "left_arm_shoulder_lift.pos":  "right_arm_shoulder_lift.pos",
    "left_arm_elbow_flex.pos":     "right_arm_elbow_flex.pos",
    "left_arm_wrist_flex.pos":     "right_arm_wrist_flex.pos",
    "left_arm_wrist_roll.pos":     "right_arm_wrist_roll.pos",
    "left_arm_gripper.pos":        "right_arm_gripper.pos",
}

# 短关节名（去掉 left_arm_ 前缀和 .pos 后缀）
JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# 已在当前 XLeRobot 自我遥操作链路上实测：delta 跟随时不需要额外取反。
DEFAULT_NEGATE_JOINTS: frozenset[str] = frozenset()


def _load_negate_joints() -> frozenset[str]:
    """从环境变量 NEGATE_JOINTS 读取；为空则用默认。"""
    raw = os.environ.get("NEGATE_JOINTS")
    if raw is None:
        return DEFAULT_NEGATE_JOINTS
    return frozenset(j.strip() for j in raw.split(",") if j.strip())


NEGATE_JOINTS = _load_negate_joints()


# ============================================================
# 镜像函数
# ============================================================

def mirror_left_to_right(
    left_pos: dict[str, float],
    negate_joints: frozenset[str] | None = None,
) -> dict[str, float]:
    """把左臂关节位置映射成右臂目标位置。

    Args:
        left_pos: 含 `left_arm_*.pos` 6 个键的 dict（其它键忽略）。
        negate_joints: 取反的短关节名集合；默认用模块级 NEGATE_JOINTS。

    Returns:
        含 `right_arm_*.pos` 6 个键的 dict。
    """
    negate = NEGATE_JOINTS if negate_joints is None else negate_joints
    out: dict[str, float] = {}
    for left_key, right_key in LEFT_TO_RIGHT_NAMES.items():
        if left_key not in left_pos:
            raise KeyError(f"镜像输入缺少 {left_key}（实际 keys: {sorted(left_pos.keys())[:3]}...）")
        v = float(left_pos[left_key])
        joint = left_key.removeprefix("left_arm_").removesuffix(".pos")
        if joint in negate:
            v = -v
        out[right_key] = v
    return out


def mirror_left_delta_to_right_target(
    *,
    left_current: dict[str, float],
    left_origin: dict[str, float],
    right_origin: dict[str, float],
    negate_joints: frozenset[str] | None = None,
) -> dict[str, float]:
    """把左臂相对启动姿态的变化量映射到右臂启动姿态上。

    这比绝对位置跟随更适合第一次真机验证：右臂不会因为左右臂初始姿态
    不完全一致而突然追一个远处目标。
    """
    negate = NEGATE_JOINTS if negate_joints is None else negate_joints
    out: dict[str, float] = {}
    for left_key, right_key in LEFT_TO_RIGHT_NAMES.items():
        if left_key not in left_current or left_key not in left_origin:
            raise KeyError(f"镜像 delta 输入缺少 {left_key}")
        if right_key not in right_origin:
            raise KeyError(f"右臂起点缺少 {right_key}")

        delta = float(left_current[left_key]) - float(left_origin[left_key])
        joint = left_key.removeprefix("left_arm_").removesuffix(".pos")
        if joint in negate:
            delta = -delta
        out[right_key] = float(right_origin[right_key]) + delta
    return out


def limit_action_relative_to_observation(
    action: dict[str, float],
    observation: dict[str, float],
    max_relative_target: float | None,
) -> dict[str, float]:
    """把 action 限制在当前 observation 附近，避免单帧目标跳变过大。

    这是 ACT 层的安全限幅，避免触发底层 xlerobot_2wheels 当前
    max_relative_target 的 key 命名问题。
    """
    if max_relative_target is None:
        return action

    cap = float(max_relative_target)
    limited: dict[str, float] = {}
    for key, target in action.items():
        if key not in observation:
            limited[key] = float(target)
            continue
        current = float(observation[key])
        target = float(target)
        limited[key] = min(max(target, current - cap), current + cap)
    return limited


def extract_left_arm(obs_or_action: dict[str, float]) -> dict[str, float]:
    """从 robot 的 observation/action dict 中筛出 6 个 left_arm_*.pos 键。"""
    return {k: float(v) for k, v in obs_or_action.items()
            if k.startswith("left_arm_") and k.endswith(".pos")}


def extract_right_arm(obs_or_action: dict[str, float]) -> dict[str, float]:
    """同上，筛 right_arm。"""
    return {k: float(v) for k, v in obs_or_action.items()
            if k.startswith("right_arm_") and k.endswith(".pos")}


def extract_head(obs: dict[str, float]) -> dict[str, float]:
    """筛 head_motor_1.pos / head_motor_2.pos。"""
    return {k: float(v) for k, v in obs.items()
            if k.startswith("head_motor_") and k.endswith(".pos")}


# ============================================================
# Probe 子命令：交互式判断哪些关节需要取反
# ============================================================

def _probe(loop_hz: int = 30) -> int:
    """连机器人，关左臂扭矩，逐关节让用户推动并观察右臂跟随方向。

    流程：
      1. 连 XLeRobot 两轮版驱动
      2. 关左臂扭矩
      3. 锁右臂初始位
      4. 逐个关节：提示用户推一下左臂某一关节，观察右臂相应关节是否同向
      5. 用户按 'y' 表示同向（不取反），'n' 表示反向（要取反）
      6. 收集结果输出新的 NEGATE_JOINTS 建议
    """
    from lerobot.robots.xlerobot_2wheels import XLerobot2Wheels, XLerobot2WheelsConfig

    from act.config import CONFIG

    print("=" * 60)
    print("🔍 镜像表 probe —— 逐关节实测哪些关节要取反")
    print("=" * 60)
    print("⚠️  操作前请先用手扶住左臂，启动后会立刻关掉左臂扭矩。")
    print()
    input("准备好了按回车继续 ... ")

    cfg = XLerobot2WheelsConfig(
        port1=CONFIG.follower_port1,
        port2=CONFIG.follower_port2,
        id=CONFIG.follower_id,
        cameras={},  # probe 不用相机
        max_relative_target=None,
    )
    robot = XLerobot2Wheels(cfg)
    try:
        robot.connect()  # 内部会启用所有扭矩
    except Exception:
        try:
            robot.disconnect()
        except Exception:
            pass
        raise

    # 仅关左臂 6 电机的扭矩；头部 + 右臂保持
    robot.bus1.disable_torque(robot.left_arm_motors)
    print("✅ 左臂扭矩已关闭，可以用手推动了。\n")

    # 锁定右臂在当前位置
    obs = robot.get_observation()
    right_lock = extract_right_arm(obs)
    head_lock = extract_head(obs)

    needs_negate: list[str] = []
    period = 1.0 / loop_hz

    try:
        for idx, joint in enumerate(JOINT_NAMES, 1):
            print(f"--- [{idx}/{len(JOINT_NAMES)}] 测试关节: {joint} ---")
            print(f"请缓慢推动左臂 **{joint}** 关节往一个方向 5 秒，观察右臂 {joint} 是否同向：")
            print("   （右臂会被锁住不跟随，但你能看到 Goal_Position 的方向）")

            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 5.0:
                obs = robot.get_observation()
                left_pos = extract_left_arm(obs)
                # 仅本关节做镜像试探（其它关节锁住），看右臂的 Goal 方向
                test_target = dict(right_lock)
                test_target[f"right_arm_{joint}.pos"] = left_pos[f"left_arm_{joint}.pos"]
                action_to_send = limit_action_relative_to_observation(
                    {**test_target, **head_lock},
                    obs,
                    CONFIG.max_relative_target if CONFIG.max_relative_target > 0 else None,
                )
                robot.send_action(action_to_send)
                time.sleep(period)

            ans = ""
            while ans not in {"y", "n", "s"}:
                ans = input("右臂方向是否与左臂相同？[y=同向 / n=反向 / s=跳过]: ").strip().lower()

            if ans == "n":
                needs_negate.append(joint)
                print(f"   → 标记 {joint} 需要取反\n")
            elif ans == "y":
                print(f"   → {joint} 不取反\n")
            else:
                print(f"   → 跳过 {joint}\n")
    finally:
        # 让左臂回到合理位置后再上扭矩；这里直接断开（解析器自身会清理）
        try:
            robot.disconnect()
        except Exception:
            pass

    print("=" * 60)
    print("✅ Probe 完成。建议的 NEGATE_JOINTS：")
    print(f"   {set(needs_negate)}")
    print()
    print("应用方式（任选其一）：")
    print(f"   1) 临时（环境变量）: export NEGATE_JOINTS='{','.join(needs_negate)}'")
    print(f"   2) 永久（改本文件 DEFAULT_NEGATE_JOINTS）")
    print("=" * 60)
    return 0


def _show() -> int:
    """打印当前镜像表设置。"""
    print("当前镜像表配置：")
    print(f"  关节顺序        : {list(JOINT_NAMES)}")
    print(f"  DEFAULT_NEGATE  : {set(DEFAULT_NEGATE_JOINTS)}")
    print(f"  环境变量 NEGATE : {os.environ.get('NEGATE_JOINTS', '(未设置)')}")
    print(f"  实际生效        : {set(NEGATE_JOINTS)}")
    print()
    print("镜像示例（左 → 右）：")
    sample = {f"left_arm_{j}.pos": 1.0 for j in JOINT_NAMES}
    out = mirror_left_to_right(sample)
    for k, v in out.items():
        print(f"  {k:<32s} = {v:+.1f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="左→右臂镜像表工具")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe", help="连机器人，逐关节交互式判断是否取反")
    sub.add_parser("show", help="打印当前镜像表配置")
    args = parser.parse_args()

    if args.cmd == "probe":
        return _probe()
    elif args.cmd == "show":
        return _show()
    return 1


if __name__ == "__main__":
    sys.exit(main())
