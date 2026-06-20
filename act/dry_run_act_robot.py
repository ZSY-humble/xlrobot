#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ACT 真机只预测不执行。

用途：
  - 连接 XLeRobot 和相机
  - 读取当前 observation
  - 跑完整 ACT preprocessor -> policy -> postprocessor
  - 打印反归一化后的 6 维 action
  - 不调用 robot.send_action()，因此不会驱动关节
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Rerun 底层会经过 Rust/wgpu，默认可能打印大量显卡后端 warning。
# 真机测试时这些日志会遮挡安全确认和动作输出，默认只保留 error。
os.environ["RUST_LOG"] = "error"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from act.config import CONFIG
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.feature_utils import build_dataset_frame
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots.xlerobot_2wheels.config_xlerobot_2wheels import XLerobot2WheelsConfig
from lerobot.robots.xlerobot_2wheels.xlerobot_2wheels import XLerobot2Wheels
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


DEFAULT_POLICY = (
    "outputs/train/"
    "xlerobot_act_pick_place_clean_le400_act_a100_80k_bs64_chunk60_quantiles/"
    "checkpoints/last/pretrained_model"
)
DEFAULT_ROOT = "dataset/zhoushangyu77/xlerobot_act_pick_place_clean_le400"
DEFAULT_REPO = "zhoushangyu77/xlerobot_act_pick_place_clean_le400"
DEFAULT_CALIBRATION_DIR = (
    Path.home() / ".cache/huggingface/lerobot/calibration/robots/xlerobot_2wheels"
)


def _cameras_config() -> dict[str, OpenCVCameraConfig]:
    return {
        "top": OpenCVCameraConfig(
            index_or_path=CONFIG.cam_top,
            fps=CONFIG.cam_fps,
            width=CONFIG.cam_width,
            height=CONFIG.cam_height,
        ),
        "right_wrist": OpenCVCameraConfig(
            index_or_path=CONFIG.cam_right_wrist,
            fps=CONFIG.cam_fps,
            width=CONFIG.cam_width,
            height=CONFIG.cam_height,
        ),
    }


def _format_action(action: dict[str, float], prev_action: dict[str, float] | None) -> str:
    parts = []
    for name, value in action.items():
        short = name.removeprefix("right_arm_").removesuffix(".pos")
        if prev_action is None or name not in prev_action:
            parts.append(f"{short}={value:8.3f}")
        else:
            delta = value - prev_action[name]
            parts.append(f"{short}={value:8.3f} Δ={delta:7.3f}")
    return " | ".join(parts)


def _state_delta_summary(
    action: dict[str, float],
    obs: dict,
    action_names: list[str],
) -> tuple[float, str, str]:
    """返回 action 到当前右臂真实 state 的最大差值摘要。"""
    deltas = {name: float(action[name]) - float(obs[name]) for name in action_names}
    worst_name = max(deltas, key=lambda name: abs(deltas[name]))
    worst_short = worst_name.removeprefix("right_arm_").removesuffix(".pos")
    compact = " | ".join(
        f"{name.removeprefix('right_arm_').removesuffix('.pos')}={deltas[name]:7.2f}"
        for name in action_names
    )
    return abs(deltas[worst_name]), worst_short, compact


def main() -> int:
    parser = argparse.ArgumentParser(description="真机只预测 ACT action，不执行")
    parser.add_argument("--policy", default=DEFAULT_POLICY, help="pretrained_model 路径")
    parser.add_argument("--dataset-root", default=DEFAULT_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--device", default=CONFIG.train_device)
    parser.add_argument(
        "--seconds",
        type=float,
        default=60.0,
        help="dry-run 时长；<=0 表示一直预览，直到 Ctrl+C 停止",
    )
    parser.add_argument("--fps", type=int, default=5, help="只预测时的打印频率")
    parser.add_argument("--task", default=CONFIG.task_desc)
    parser.add_argument("--warn-delta", type=float, default=8.0)
    parser.add_argument("--display-data", action="store_true", help="打开 Rerun，显示 observation/action")
    parser.add_argument("--display-ip", type=str, default=None, help="连接远端 Rerun server")
    parser.add_argument("--display-port", type=int, default=None, help="连接远端 Rerun server 端口")
    parser.add_argument("--display-compressed-images", action="store_true", help="Rerun 中压缩图像")
    parser.add_argument(
        "--print-actions",
        action="store_true",
        help="在终端逐帧打印 action、Δ 和 to_current；默认不打印，避免遮挡安全确认",
    )
    parser.add_argument(
        "--consume-action-queue",
        action="store_true",
        help="连续消费 ACT 的 n_action_steps 队列；默认每帧重置队列，只看当前观测第一步 action",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
        help="机器人标定目录，目录内应包含 <robot_id>.json",
    )
    args = parser.parse_args()

    policy_path = Path(args.policy)
    if not policy_path.exists():
        print(f"❌ 找不到 policy: {policy_path}")
        return 1

    CONFIG.banner("ACT 真机 dry-run：只预测，不执行")
    print(f"  policy     : {policy_path}")
    dry_run_time = "直到 Ctrl+C" if args.seconds <= 0 else f"{args.seconds}s"
    print(f"  dry-run    : {dry_run_time} @ {args.fps}Hz")
    print(f"  task       : {args.task}")
    print(f"  Rerun      : {args.display_data}")
    print("  ⚠️ 不会调用 robot.send_action()，关节不会收到 policy action\n")

    print("🎯 加载训练数据集 metadata，用于校验 schema 和归一化参数")
    ds = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.dataset_root,
        video_backend=args.video_backend,
    )

    device = get_safe_torch_device(args.device, log=True)
    policy_cfg = PreTrainedConfig.from_pretrained(
        policy_path,
        cli_overrides=[f"--device={device.type}"],
    )
    # checkpoint 会覆盖权重；dry-run 不需要联网下载 torchvision 的初始化权重。
    if hasattr(policy_cfg, "pretrained_backbone_weights"):
        policy_cfg.pretrained_backbone_weights = None
    policy_cfg.pretrained_path = str(policy_path)
    policy = make_policy(policy_cfg, ds_meta=ds.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        dataset_stats=rename_stats(ds.meta.stats, {}),
        preprocessor_overrides={"device_processor": {"device": policy_cfg.device}},
    )
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    robot_cfg = XLerobot2WheelsConfig(
        port1=CONFIG.follower_port1,
        port2=CONFIG.follower_port2,
        id=CONFIG.follower_id,
        calibration_dir=args.calibration_dir,
        cameras=_cameras_config(),
        max_relative_target=int(CONFIG.max_relative_target)
        if CONFIG.max_relative_target > 0
        else None,
    )
    robot = XLerobot2Wheels(robot_cfg)
    if not robot.calibration_fpath.is_file():
        print("❌ 没找到机器人标定文件，已停止 dry-run，避免自动进入标定流程。")
        print(f"   期望路径: {robot.calibration_fpath}")
        print()
        print("如果这是当前这台 XLeRobot，请先安装仓库里的标定文件：")
        print(f"   python act/install_calibration.py --robot-id {CONFIG.follower_id}")
        print()
        print("安装后再重新运行 dry-run。")
        return 2

    prev_action: dict[str, float] | None = None
    action_names = ds.features["action"]["names"]
    print("🎯 action 顺序")
    for i, name in enumerate(action_names):
        print(f"  {i}: {name}")
    print()

    try:
        robot.connect(calibrate=False)
        if args.display_data:
            init_rerun(session_name="act_dry_run", ip=args.display_ip, port=args.display_port)
        print("✅ 机器人和相机已连接，开始只预测。按 Ctrl+C 停止。\n")
        deadline = None if args.seconds <= 0 else time.perf_counter() + args.seconds
        period = 1.0 / max(args.fps, 1)
        tick = 0

        while deadline is None or time.perf_counter() < deadline:
            t0 = time.perf_counter()
            if not args.consume_action_queue:
                policy.reset()
            obs = robot.get_observation()
            observation_frame = build_dataset_frame(ds.features, obs, prefix=OBS_STR)
            action_tensor = predict_action(
                observation=observation_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=args.task,
                robot_type=robot.robot_type,
            )
            action = make_robot_action(action_tensor, ds.features)
            values = np.array([action[name] for name in action_names], dtype=np.float32)
            bad = not np.isfinite(values).all()
            max_delta = 0.0
            if prev_action is not None:
                deltas = [abs(action[name] - prev_action[name]) for name in action_names]
                max_delta = max(deltas)
            max_to_current, worst_joint, to_current = _state_delta_summary(action, obs, action_names)

            status = "❌ NaN/inf" if bad else "✅"
            if max_delta > args.warn_delta:
                status += f" ⚠️ maxΔ={max_delta:.3f}"
            if max_to_current > args.warn_delta:
                status += f" ⚠️ to_state={max_to_current:.3f}({worst_joint})"
            if args.print_actions:
                print(f"[{tick:04d}] {status} {_format_action(action, prev_action)}")
                print(f"       to_current: {to_current}")
            elif bad or max_delta > args.warn_delta or max_to_current > args.warn_delta:
                print(f"[{tick:04d}] {status}")
                print(f"       to_current: {to_current}")
            if args.display_data:
                log_rerun_data(
                    observation=obs,
                    action=action,
                    compress_images=args.display_compressed_images,
                )
            prev_action = action
            tick += 1

            sleep_s = period - (time.perf_counter() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("\n⚠️ 用户停止 dry-run")
    finally:
        if robot.is_connected:
            robot.disconnect()
        print("✅ dry-run 结束，未执行任何 policy action。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
