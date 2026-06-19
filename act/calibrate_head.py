#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""act/calibrate_head.py — XLeRobot 头部相机两个电机单独标定。

这个脚本只标定 head_motor_1 / head_motor_2，不重新标定左右臂和底盘。
两个头部电机都会按有限范围关节记录 min/max，用来限制相机转动范围。

使用：
    python act/calibrate_head.py

标定后建议：
    python act/capture_reset_home.py
    python act/test_reset_home.py --dry-run
    python act/test_reset_home.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 兜底：允许 `python act/calibrate_head.py` 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from act.calibrate_xlerobot import (  # noqa: E402
    DEFAULT_CENTER_WINDOW,
    DEFAULT_EDGE_GUARD,
    _calibrate_head_only,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="XLeRobot 头部相机单独标定")
    parser.add_argument(
        "--edge-guard",
        type=int,
        default=DEFAULT_EDGE_GUARD,
        help="忽略 0/4095 两端多少编码值，默认 128",
    )
    parser.add_argument(
        "--center-window",
        type=int,
        default=DEFAULT_CENTER_WINDOW,
        help="围绕中位允许记录的半窗口，默认 2200",
    )
    args = parser.parse_args()

    return _calibrate_head_only(
        edge_guard=args.edge_guard,
        center_window=args.center_window,
    )


if __name__ == "__main__":
    sys.exit(main())
