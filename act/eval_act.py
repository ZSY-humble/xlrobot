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
import os
import subprocess
import sys
from pathlib import Path

# 兜底
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from act.config import CONFIG


def _quiet_rerun_env() -> dict[str, str]:
    """给 Rerun/wgpu 子进程降噪，避免真机确认界面被 warning 刷屏。"""
    env = os.environ.copy()
    env["RUST_LOG"] = "error"
    return env


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
    parser.add_argument("--yes", action="store_true", help="跳过开始前安全确认")
    parser.add_argument(
        "--preview-cameras",
        action="store_true",
        help="执行前先打开 top/right_wrist 相机预览；按 q/Esc 关闭预览后再确认启动",
    )
    parser.add_argument(
        "--preview-rerun",
        action="store_true",
        help="在同一个 lerobot-record 进程里先打开 Rerun 看相机，按 Enter 后再执行",
    )
    parser.add_argument(
        "--preview-seconds",
        type=float,
        default=0.0,
        help="Rerun 预览最长时长；<=0 表示确认前一直显示",
    )
    parser.add_argument(
        "--max-relative-target",
        type=int,
        default=int(CONFIG.max_relative_target),
        help="每次下发给单个关节的最大相对变化；越小越慢，<=0 表示关闭限幅",
    )
    args = parser.parse_args()

    policy_path = Path(args.policy) if args.policy else CONFIG.policy_path
    if not policy_path.exists():
        print(f"❌ 找不到 checkpoint：{policy_path}")
        print("   请确认训练已完成，或用 --policy=/abs/path/to/pretrained_model 指定")
        return 1

    device = args.device or CONFIG.train_device
    task_desc = args.task or CONFIG.task_desc
    checkpoint_name = policy_path.parent.name if policy_path.name == "pretrained_model" else policy_path.name
    eval_dataset_name = args.eval_name or f"eval_{CONFIG.dataset_name}_{checkpoint_name}"
    if not eval_dataset_name.startswith("eval_"):
        eval_dataset_name = f"eval_{eval_dataset_name}"
    eval_repo_id = f"{CONFIG.hf_user}/{eval_dataset_name}" if CONFIG.hf_user else f"local/{eval_dataset_name}"
    eval_root = CONFIG.dataset_root / eval_repo_id

    CONFIG.banner("ACT 真机推理 🤖（仅控右臂）")
    print(f"  checkpoint  : {policy_path}")
    print(f"  评估数据集  : {eval_repo_id}")
    print(f"  评估路径    : {eval_root}")
    print(f"  评估计划    : {args.num_episodes} × {args.episode_time}s")
    print(f"  task        : {task_desc}")
    print(f"  device      : {device}")
    print(f"  单步限幅    : {args.max_relative_target}")
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
        f"--dataset.root={eval_root}",
        f"--dataset.num_episodes={args.num_episodes}",
        f"--dataset.episode_time_s={args.episode_time}",
        f"--dataset.fps={CONFIG.cam_fps}",
        f"--dataset.single_task={task_desc}",
        "--dataset.streaming_encoding=true",
        "--dataset.push_to_hub=false",
    ]
    if args.max_relative_target > 0:
        cmd.insert(6, f"--robot.max_relative_target={args.max_relative_target}")
    if args.preview_rerun and not args.yes:
        cmd.extend(
            [
                "--wait_before_start=true",
                "--wait_preview_fps=10",
            ]
        )
    print("$", __import__("shlex").join(cmd), "\n")

    if args.preview_cameras:
        preview_cmd = [
            sys.executable,
            "act/check_cameras.py",
            "--devices",
            CONFIG.cam_top,
            CONFIG.cam_right_wrist,
            "--width",
            str(CONFIG.cam_width),
            "--height",
            str(CONFIG.cam_height),
            "--fps",
            str(CONFIG.cam_fps),
        ]
        print("🔍 先打开相机预览；确认画面后按 q 或 Esc 关闭预览窗口。")
        print("$", __import__("shlex").join(preview_cmd), "\n")
        preview_ret = subprocess.call(preview_cmd, env=_quiet_rerun_env())
        if preview_ret != 0:
            print(f"❌ 相机预览失败，返回码：{preview_ret}。未启动 policy 执行。")
            return preview_ret
        print()

    if not args.yes and not args.preview_rerun:
        print("⚠️  即将启动 ACT 真机执行，右臂会移动。")
        print("   请确认：右臂周围安全、桌面已清空或物体在安全位置、手在急停附近。")
        try:
            input("   按 Enter 开始执行；按 Ctrl+C 取消：")
        except KeyboardInterrupt:
            print("\n🛑 已取消，未启动 policy 执行。")
            return 130
        print()
    elif args.preview_rerun and not args.yes:
        print("🔍 将在 lerobot-record 内部先打开 Rerun 预览。")
        print("   同一个进程会停在确认提示；按 Enter 后直接开始 3 秒执行，不会重启。")
        print()

    try:
        return subprocess.call(cmd, env=_quiet_rerun_env())
    except KeyboardInterrupt:
        print("\n🛑 已停止 ACT 真机执行。")
        return 130


if __name__ == "__main__":
    sys.exit(main())
