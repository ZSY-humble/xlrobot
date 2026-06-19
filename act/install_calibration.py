#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""安装仓库内置 XLeRobot 标定文件到 LeRobot 默认 cache。

LeRobot 连接机器人时默认从：
  ~/.cache/huggingface/lerobot/calibration/robots/xlerobot_2wheels/<robot_id>.json
读取标定文件。

本脚本把仓库里的 act/config/calibration/xlerobot_main.json 复制到该位置，
方便新电脑 clone 仓库后直接复用这台 XLeRobot 的标定。

注意：标定值强依赖具体硬件、舵机 ID、装配方向和机械限位。换机器人后应重新标定。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# 兜底：允许 `python act/install_calibration.py` 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lerobot.utils.constants import HF_LEROBOT_HOME  # noqa: E402

from act.config import CONFIG  # noqa: E402


DEFAULT_SOURCE = Path("act/config/calibration/xlerobot_main.json")
ROBOT_TYPE = "xlerobot_2wheels"


def _target_path(robot_id: str) -> Path:
    return HF_LEROBOT_HOME / "calibration" / "robots" / ROBOT_TYPE / f"{robot_id}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="安装 XLeRobot 仓库内置标定文件")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="仓库内标定 JSON 路径")
    parser.add_argument("--robot-id", type=str, default=CONFIG.follower_id, help="目标机器人 id")
    parser.add_argument("--force", action="store_true", help="目标标定文件已存在时允许覆盖")
    args = parser.parse_args()

    source = args.source
    target = _target_path(args.robot_id)

    if not source.exists():
        raise FileNotFoundError(f"仓库标定文件不存在：{source}")

    if target.exists() and not args.force:
        print("⚠️  目标标定文件已存在，未覆盖：")
        print(f"   {target}")
        print()
        print("如果确认要使用仓库内标定覆盖本机 cache，请运行：")
        print(f"   python act/install_calibration.py --force --robot-id {args.robot_id}")
        return 2

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

    print("✅ 已安装 XLeRobot 标定文件：")
    print(f"   source: {source}")
    print(f"   target: {target}")
    print()
    print("下一步可直接连接机器人，或先 dry-run 检查 reset home：")
    print("   python act/test_reset_home.py --dry-run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
