#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""act/eval_act.py — 训好的 ACT 策略真机推理（仅控右臂）。

🎯 推理模式：
  - 模型 policy 输出 6 维 right_arm action
  - XLeRobot.send_action() 按 key 前缀过滤：仅写 right_arm，左臂 / 头部 / 底盘不动
  - 因此推理时**不需要再推左臂**

走 lerobot-record CLI（已注册 xlerobot_2wheels 与 ACT policy 解析），
自动处理 preprocessor / postprocessor / action chunk 队列等推理细节。

使用：
    python act/eval_act.py
    python act/eval_act.py --policy=/abs/path/to/checkpoints/last/pretrained_model
    python act/eval_act.py --num-episodes=20 --episode-time=30

⌨️ 三键（用于按段统计成功率）：
    →  本段成功
    ←  本段失败 / 重试
    Esc 停止
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# 兜底
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from act.config import CONFIG


def _cameras_arg() -> str:
    h, w, fps = CONFIG.cam_height, CONFIG.cam_width, CONFIG.cam_fps
    return (
        "{"
        f'top:         {{"type": "opencv", "index_or_path": "{CONFIG.cam_top}",         "width": {w}, "height": {h}, "fps": {fps}}}, '
        f'right_wrist: {{"type": "opencv", "index_or_path": "{CONFIG.cam_right_wrist}", "width": {w}, "height": {h}, "fps": {fps}}}'
        "}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="XLeRobot 自我遥操作 ACT 真机推理（仅右臂）")
    parser.add_argument("--policy", type=str, default=None, help="checkpoint 路径")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--episode-time", type=int, default=30)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--eval-name", type=str, default=None, help="评估数据集后缀名")
    args = parser.parse_args()

    policy_path = Path(args.policy) if args.policy else CONFIG.policy_path
    if not policy_path.exists():
        print(f"❌ 找不到 checkpoint：{policy_path}")
        print("   请确认训练已完成，或用 --policy=/abs/path/to/pretrained_model 指定")
        return 1

    device = args.device or CONFIG.train_device
    task_desc = args.task or CONFIG.task_desc
    eval_dataset_name = args.eval_name or f"{CONFIG.dataset_name}_eval_run"
    eval_repo_id = f"{CONFIG.hf_user}/{eval_dataset_name}" if CONFIG.hf_user else f"local/{eval_dataset_name}"

    CONFIG.banner("ACT 真机推理 🤖（仅控右臂）")
    print(f"  checkpoint  : {policy_path}")
    print(f"  评估数据集  : {eval_repo_id}")
    print(f"  评估计划    : {args.num_episodes} × {args.episode_time}s")
    print(f"  task        : {task_desc}")
    print(f"  device      : {device}")
    print()
    print("⚠️  推理时**不需要推左臂**。左臂保持不动（不接收 policy action）。")
    print("   桌面物体摆到训练时的相似位置，按提示开始。")
    print("⌨️  → 成功 / ← 失败 / Esc 停止")
    print()

    cmd = [
        "lerobot-record",
        "--robot.type=xlerobot_2wheels",
        f"--robot.port1={CONFIG.follower_port1}",
        f"--robot.port2={CONFIG.follower_port2}",
        f"--robot.id={CONFIG.follower_id}",
        f"--robot.cameras={_cameras_arg()}",
        f"--policy.path={policy_path}",
        f"--policy.device={device}",
        "--display_data=true",
        f"--dataset.repo_id={eval_repo_id}",
        f"--dataset.num_episodes={args.num_episodes}",
        f"--dataset.episode_time_s={args.episode_time}",
        f"--dataset.fps={CONFIG.cam_fps}",
        f"--dataset.single_task={task_desc}",
        "--dataset.streaming_encoding=true",
        "--dataset.push_to_hub=false",
    ]
    print("$", __import__("shlex").join(cmd), "\n")
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
