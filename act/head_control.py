#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""头部连续旋转关节辅助函数。

当前 ACT 流程会把 head_motor_1/head_motor_2 都标定成有限范围关节，
因此默认不启用连续轴速度控制。这里保留辅助函数只是为了兼容旧参数。
"""

from __future__ import annotations

import logging
import time

from lerobot.motors.feetech import OperatingMode

ENCODER_SIZE = 4096
HALF_ENCODER = ENCODER_SIZE // 2
CONTINUOUS_HEAD_MOTORS = frozenset()


def shortest_raw_delta(current_raw: int, target_raw: int) -> int:
    """返回从 current 到 target 的环形最短 raw delta，范围 [-2048, 2047]。"""
    return int((target_raw - current_raw + HALF_ENCODER) % ENCODER_SIZE - HALF_ENCODER)


def head_target_to_raw(bus, motor: str, target_pos: float) -> int:
    """把 head_motor_*.pos 的归一化目标转成 raw 编码器值。"""
    motor_id = bus.motors[motor].id
    return int(bus._unnormalize({motor_id: float(target_pos)})[motor_id]) % ENCODER_SIZE


def continuous_head_errors_raw(robot, head_home: dict[str, float]) -> dict[str, int]:
    """返回连续头部关节的最短 raw 误差。"""
    motors = [name for name in CONTINUOUS_HEAD_MOTORS if f"{name}.pos" in head_home]
    if not motors:
        return {}

    current = robot.bus1.sync_read("Present_Position", motors, normalize=False)
    errors: dict[str, int] = {}
    for motor in motors:
        target_raw = head_target_to_raw(robot.bus1, motor, head_home[f"{motor}.pos"])
        errors[motor] = shortest_raw_delta(int(current[motor]), target_raw)
    return errors


def has_continuous_head_target(head_home: dict[str, float] | None) -> bool:
    if head_home is None:
        return False
    return any(f"{motor}.pos" in head_home for motor in CONTINUOUS_HEAD_MOTORS)


def prepare_continuous_head_velocity(robot, head_home: dict[str, float] | None) -> None:
    """把连续头部关节切到速度模式，准备按最短方向闭环回 home。"""
    if not has_continuous_head_target(head_home):
        return
    for motor in CONTINUOUS_HEAD_MOTORS:
        if f"{motor}.pos" not in head_home:
            continue
        robot.bus1.write("Goal_Velocity", motor, 0, normalize=False, num_retry=2)
        robot.bus1.write("Operating_Mode", motor, OperatingMode.VELOCITY.value, num_retry=2)


def stop_continuous_head_velocity(robot, head_home: dict[str, float] | None) -> None:
    """停止连续头部关节，并切回位置模式锁住当前位置。"""
    if not has_continuous_head_target(head_home):
        return
    for motor in CONTINUOUS_HEAD_MOTORS:
        if f"{motor}.pos" not in head_home:
            continue
        robot.bus1.write("Goal_Velocity", motor, 0, normalize=False, num_retry=2)
        current_raw = int(robot.bus1.read("Present_Position", motor, normalize=False, num_retry=2))
        robot.bus1.write("Operating_Mode", motor, OperatingMode.POSITION.value, num_retry=2)
        robot.bus1.write("Goal_Position", motor, current_raw, normalize=False, num_retry=2)


def _max_abs_raw_error(errors: dict[str, int]) -> int:
    return max((abs(value) for value in errors.values()), default=0)


def choose_continuous_head_direction(
    robot,
    head_home: dict[str, float] | None,
    *,
    speed_raw: int,
    tolerance_raw: int,
    preferred_direction: int = -1,
    probe_seconds: float = 0.18,
) -> int:
    """低速试探连续头部关节方向，返回能让 raw 最短误差变小的方向符号。

    有些 XLeRobot 头部连续轴的速度正负方向与 raw 编码器误差方向相反。
    这里不假设方向，先用很短的低速脉冲分别试 `preferred` 和反向，
    选择误差更小的一侧，避免 head_motor_1 回 home 时绕反方向。
    """
    if not has_continuous_head_target(head_home):
        return -1 if preferred_direction < 0 else 1

    preferred = -1 if preferred_direction < 0 else 1
    speed = max(1, min(int(abs(speed_raw)), 24))
    probe_seconds = max(0.05, min(float(probe_seconds), 0.5))

    start_errors = continuous_head_errors_raw(robot, head_home or {})
    start_score = _max_abs_raw_error(start_errors)
    if start_score <= tolerance_raw:
        return preferred

    candidates = [preferred, -preferred]
    best_direction = preferred
    best_score: int | None = None

    for direction in candidates:
        step_continuous_head_toward_home(
            robot,
            head_home,
            speed_raw=speed,
            tolerance_raw=tolerance_raw,
            direction_sign=direction,
        )
        time.sleep(probe_seconds)
        for motor in CONTINUOUS_HEAD_MOTORS:
            if f"{motor}.pos" in (head_home or {}):
                robot.bus1.write("Goal_Velocity", motor, 0, normalize=False, num_retry=1)
        time.sleep(0.04)

        score = _max_abs_raw_error(continuous_head_errors_raw(robot, head_home or {}))
        logging.info(
            "head_motor_1 方向试探：direction=%+d, raw_abs_error %d -> %d",
            direction,
            start_score,
            score,
        )
        if best_score is None or score < best_score:
            best_direction = direction
            best_score = score

    logging.info(
        "✅ head_motor_1 自动选择方向符号：%+d（起始 raw_abs_error=%d，试探后最小=%d）",
        best_direction,
        start_score,
        best_score if best_score is not None else start_score,
    )
    return best_direction


def step_continuous_head_toward_home(
    robot,
    head_home: dict[str, float] | None,
    *,
    speed_raw: int,
    tolerance_raw: int,
    direction_sign: int = -1,
) -> dict[str, int]:
    """按 raw 编码器最短方向推进连续头部关节，返回当前最短 raw 误差。"""
    if not has_continuous_head_target(head_home):
        return {}

    errors = continuous_head_errors_raw(robot, head_home or {})
    speed = max(1, int(abs(speed_raw)))
    sign = 1 if direction_sign >= 0 else -1
    for motor, delta in errors.items():
        if abs(delta) <= tolerance_raw:
            velocity = 0
        else:
            velocity = sign * (speed if delta > 0 else -speed)
        robot.bus1.write("Goal_Velocity", motor, velocity, normalize=False, num_retry=1)
    return errors


def head_position_targets_without_continuous(head_home: dict[str, float] | None) -> dict[str, float] | None:
    """过滤掉连续头部关节，只保留可直接用位置模式安全下发的头部目标。"""
    if head_home is None:
        return None
    filtered = {
        key: value
        for key, value in head_home.items()
        if key.removesuffix(".pos") not in CONTINUOUS_HEAD_MOTORS
    }
    return filtered or None
