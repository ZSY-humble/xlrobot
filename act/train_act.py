#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""act/train_act.py — 用自我遥操作采集的双臂数据集训练 ACT 策略。

数据集 schema（采集端 act/record_self_teleop.py 已写死）：
  observation.state              float32 (6,)  仅右臂 6 关节
  observation.images.top         video         桌面相机
  observation.images.right_wrist video         右腕相机
  action                         float32 (6,)  右臂 6 关节目标位置

> 头部 / 底盘**都不录、都不下发**。

使用：
    python act/train_act.py
    python act/train_act.py --steps=200000 --batch-size=16
    python act/train_act.py --resume                    # 断点续训
    python act/train_act.py --wandb                     # 开 wandb 日志

输出：outputs/train/<job_name>/checkpoints/last/pretrained_model
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# 兜底：允许 `python act/train_act.py` 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from act.config import CONFIG


def main() -> int:
    parser = argparse.ArgumentParser(description="XLeRobot 自我遥操作 ACT 训练")
    parser.add_argument("--steps", type=int, default=None, help="训练总步数")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu / mps")
    parser.add_argument("--resume", action="store_true", help="断点续训")
    parser.add_argument("--wandb", action="store_true", help="启用 wandb 日志")
    parser.add_argument("--streaming", action="store_true", help="从 Hub 流式读取数据集，减少本地缓存")
    args = parser.parse_args()

    steps = args.steps or CONFIG.steps
    batch_size = args.batch_size or CONFIG.batch_size
    device = args.device or CONFIG.train_device
    wandb_enable = "true" if (args.wandb or CONFIG.wandb_enable) else "false"

    CONFIG.banner("启动 ACT 训练 🚀")
    print(f"  数据集     : {CONFIG.repo_id}")
    print(f"  数据路径   : {CONFIG.dataset_path}")
    print(f"  输出目录   : {CONFIG.output_dir}")
    print(f"  steps      : {steps}")
    print(f"  batch_size : {batch_size}")
    print(f"  device     : {device}")
    print(f"  wandb      : {wandb_enable}")
    print(f"  streaming  : {args.streaming}")
    print(f"  续训       : {args.resume}")
    print()

    cmd = [
        "lerobot-train",
        "--policy.type=act",
        f"--dataset.repo_id={CONFIG.repo_id}",
        f"--dataset.root={CONFIG.dataset_path}",
        f"--output_dir={CONFIG.output_dir}",
        f"--job_name={CONFIG.job_name}",
        f"--policy.device={device}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        f"--save_freq={CONFIG.save_freq}",
        f"--log_freq={CONFIG.log_freq}",
        f"--wandb.enable={wandb_enable}",
        "--policy.push_to_hub=false",
    ]
    if args.streaming:
        cmd.append("--dataset.streaming=true")
    if args.resume:
        cmd.append("--resume=true")

    print("$", " ".join(cmd), "\n")
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
