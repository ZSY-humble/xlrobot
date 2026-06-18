#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""识别 XLeRobot 的两条串口总线。

XLeRobot 两轮底盘驱动约定：
  port1 = bus1 = 左臂 6 个电机 + 头部 2 个电机，ID 应为 1..8
  port2 = bus2 = 右臂 6 个电机 + 底盘 2 个轮子，ID 应为 1..6,9,10

本脚本只做串口打开 + broadcast ping，不写电机参数，不会让电机运动。

示例：
    python act/check_xlerobot_ports.py
    python act/check_xlerobot_ports.py --ports /dev/ttyACM0 /dev/ttyACM1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 兜底：允许 `python act/check_xlerobot_ports.py` 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lerobot.motors.feetech import FeetechMotorsBus


def _default_ports() -> list[str]:
    ports = sorted(Path("/dev").glob("ttyACM*")) + sorted(Path("/dev").glob("ttyUSB*"))
    return [str(p) for p in ports]


def _ping_port(port: str) -> dict[int, int] | None:
    """返回端口上的电机 ID -> model number；失败返回 None。"""
    bus = FeetechMotorsBus(port=port, motors={})
    try:
        bus.connect(handshake=False)
        return bus.broadcast_ping(num_retry=2, raise_on_error=False)
    except Exception as exc:  # noqa: BLE001 - 诊断脚本需要完整兜底
        print(f"⚠️  {port}: 打开或 ping 失败：{exc}")
        return None
    finally:
        try:
            bus.disconnect(disable_torque=False)
        except Exception:
            pass


def _role_from_ids(ids: list[int]) -> str:
    if ids == list(range(1, 9)):
        return "bus1 / port1：左臂 + 头部"
    if ids == [1, 2, 3, 4, 5, 6, 9, 10]:
        return "bus2 / port2：右臂 + 两轮底盘"
    if len(ids) == 8 and set(range(1, 9)).issubset(ids):
        return "疑似 bus1 / port1：左臂 + 头部"
    if set([1, 2, 3, 4, 5, 6, 9, 10]).issubset(ids):
        return "疑似 bus2 / port2：右臂 + 两轮底盘"
    return "未知：ID 数量或编号不符合 XLeRobot 默认布线"


def main() -> int:
    parser = argparse.ArgumentParser(description="识别 XLeRobot /dev/ttyACM* 总线角色")
    parser.add_argument("--ports", nargs="+", default=None, help="要检查的串口列表")
    args = parser.parse_args()

    ports = args.ports if args.ports else _default_ports()
    if not ports:
        print("❌ 没找到 /dev/ttyACM* 或 /dev/ttyUSB*。")
        return 1

    print("\n🎯 XLeRobot 串口总线识别")
    print("   只读 ping，不会让电机运动。")
    print(f"   检查端口: {', '.join(ports)}\n")

    found: dict[str, list[int]] = {}
    for port in ports:
        id_model = _ping_port(port)
        if not id_model:
            print(f"❌ {port}: 未发现电机")
            continue

        ids = sorted(id_model)
        found[port] = ids
        print(f"✅ {port}: 发现 ID {ids}")
        print(f"   判断: {_role_from_ids(ids)}")
        print(f"   model: {id_model}\n")

    port1 = next((p for p, ids in found.items() if ids == list(range(1, 9))), None)
    port2 = next((p for p, ids in found.items() if ids == [1, 2, 3, 4, 5, 6, 9, 10]), None)
    if port1 and port2:
        print("✅ 可以这样设置：")
        print(f"   export FOLLOWER_PORT1={port1}")
        print(f"   export FOLLOWER_PORT2={port2}")
        return 0

    print("⚠️  没有同时识别出标准 bus1(1..8) 和 bus2(1..6,9,10)。")
    print("   请检查两根 USB 串口线、电源、舵机 ID、是否有程序占用串口。")
    return 2


if __name__ == "__main__":
    sys.exit(main())
