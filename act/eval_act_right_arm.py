#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""XLeRobot ACT 真机执行：严格仅控右臂 6 维。

这个脚本不走 lerobot-record 的整机 action schema。
流程：
  1. 加载训练数据集 metadata，得到 6 维 right_arm state/action 顺序和归一化参数。
  2. 连接机器人和相机，按训练 schema 从完整 observation 中取右臂 6 维 + 两路图像。
  3. 执行前在 Rerun 中持续显示相机/状态，不发送 action。
  4. 用户按 Enter 后，ACT 输出 6 维 action，只把这 6 个 right_arm_*.pos 发送给机器人。
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Rerun/wgpu 降噪，避免遮挡确认提示。
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
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


DEFAULT_POLICY = (
    "outputs/train/"
    "xlerobot_act_pick_place_clean_le400_act_a100_80k_bs64_chunk60_quantiles/"
    "checkpoints/030000/pretrained_model"
)
DEFAULT_ROOT = "dataset/zhoushangyu77/xlerobot_act_pick_place_clean_le400"
DEFAULT_REPO = "zhoushangyu77/xlerobot_act_pick_place_clean_le400"
DEFAULT_TASK = "pick up the object and place it into the target container"
DEFAULT_CALIBRATION_DIR = (
    Path.home() / ".cache/huggingface/lerobot/calibration/robots/xlerobot_2wheels"
)
REQUIRED_ROBOT_ENV = (
    "FOLLOWER_PORT1",
    "FOLLOWER_PORT2",
    "CAM_TOP",
    "CAM_RIGHT_WRIST",
)
RIGHT_ARM_ACTION_NAMES = [
    "right_arm_shoulder_pan.pos",
    "right_arm_shoulder_lift.pos",
    "right_arm_elbow_flex.pos",
    "right_arm_wrist_flex.pos",
    "right_arm_wrist_roll.pos",
    "right_arm_gripper.pos",
]


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


def _validate_robot_io_config() -> bool:
    missing = [name for name in REQUIRED_ROBOT_ENV if not os.environ.get(name)]
    if missing:
        print("❌ 真机执行前必须显式配置端口和相机，避免设备号变化导致误控。")
        print("   缺少环境变量: " + ", ".join(missing))
        print()
        print("示例：")
        print("   export FOLLOWER_PORT1=/dev/ttyACM1")
        print("   export FOLLOWER_PORT2=/dev/ttyACM0")
        print("   export CAM_TOP=/dev/video2")
        print("   export CAM_RIGHT_WRIST=/dev/video4")
        print()
        print("配置后再运行：python act/eval_act_right_arm.py")
        return False
    return True


def _validate_schema(ds: LeRobotDataset) -> list[str]:
    action_names = list(ds.features["action"]["names"])
    state_names = list(ds.features["observation.state"]["names"])
    if action_names != RIGHT_ARM_ACTION_NAMES:
        raise ValueError(f"action 顺序不是右臂 6 维：{action_names}")
    if state_names != RIGHT_ARM_ACTION_NAMES:
        raise ValueError(f"state 顺序不是右臂 6 维：{state_names}")
    for image_key in ("observation.images.top", "observation.images.right_wrist"):
        if image_key not in ds.features:
            raise ValueError(f"训练数据集缺少图像 key: {image_key}")
    return action_names


def _action_values(action: dict[str, float], action_names: list[str]) -> np.ndarray:
    return np.array([action[name] for name in action_names], dtype=np.float32)


def _max_to_current(
    action: dict[str, float],
    obs: dict,
    action_names: list[str],
) -> tuple[float, str]:
    deltas = {name: float(action[name]) - float(obs[name]) for name in action_names}
    worst = max(deltas, key=lambda name: abs(deltas[name]))
    return abs(deltas[worst]), worst


def _safe_goal_from_present(
    action: dict[str, float],
    present_pos: dict[str, float],
    action_names: list[str],
    max_relative_target: float,
) -> dict[str, float]:
    cap = float(max_relative_target)
    safe_goal_pos = {}
    for name in action_names:
        present = float(present_pos[name])
        delta = float(action[name]) - present
        safe_goal_pos[name] = present + float(np.clip(delta, -cap, cap))
    return safe_goal_pos


def _send_right_arm_only(
    robot: XLerobot2Wheels,
    action: dict[str, float],
    action_names: list[str],
    max_relative_target: float,
) -> dict[str, float]:
    """只写右臂 Goal_Position，不写左臂、头部、底盘。"""
    present_pos = {
        f"{name}.pos": value
        for name, value in robot.bus2.sync_read(
            "Present_Position",
            robot.right_arm_motors,
        ).items()
    }
    safe_goal_pos = _safe_goal_from_present(
        action=action,
        present_pos=present_pos,
        action_names=action_names,
        max_relative_target=max_relative_target,
    )
    robot.bus2.sync_write(
        "Goal_Position",
        {name.replace(".pos", ""): value for name, value in safe_goal_pos.items()},
    )
    return safe_goal_pos


def _max_delta_from_reference(
    values: dict[str, float],
    reference: dict[str, float],
    action_names: list[str],
) -> tuple[float, str]:
    deltas = {name: float(values[name]) - float(reference[name]) for name in action_names}
    worst = max(deltas, key=lambda name: abs(deltas[name]))
    return abs(deltas[worst]), worst


def _format_joint_deltas(values: dict[str, float], reference: dict[str, float], action_names: list[str]) -> str:
    parts = []
    for name in action_names:
        short = name.removeprefix("right_arm_").removesuffix(".pos")
        delta = float(values[name]) - float(reference[name])
        parts.append(f"{short}={delta:+.2f}")
    return " | ".join(parts)


def _predict_right_arm_action(
    *,
    obs: dict,
    ds: LeRobotDataset,
    action_names: list[str],
    policy,
    device,
    preprocessor,
    postprocessor,
    task: str,
    robot_type: str,
) -> dict[str, float]:
    observation_frame = build_dataset_frame(ds.features, obs, prefix=OBS_STR)
    action_tensor = predict_action(
        observation=observation_frame,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=policy.config.use_amp,
        task=task,
        robot_type=robot_type,
    )
    action = make_robot_action(action_tensor, ds.features)
    if list(action) != action_names:
        raise RuntimeError(f"policy 输出 key 异常：{list(action)}")
    values = _action_values(action, action_names)
    if not np.isfinite(values).all():
        raise RuntimeError(f"policy 输出 NaN/inf：{values}")
    return action


def _wait_for_enter_with_rerun(
    *,
    robot: XLerobot2Wheels,
    fps: int,
    compress_images: bool,
) -> None:
    print()
    print("⚠️  当前仅显示 Rerun 相机/状态，不会发送任何 action。")
    print("   确认画面、初始姿态和周围安全后，按 Enter 开始执行。")
    print("   按 Ctrl+C 取消：", end="", flush=True)
    period = 1.0 / max(fps, 1)
    while True:
        t0 = time.perf_counter()
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if readable:
            sys.stdin.readline()
            print("✅ 已确认，开始执行 ACT。")
            return

        obs = robot.get_observation()
        log_rerun_data(observation=obs, action=None, compress_images=compress_images)

        sleep_s = period - (time.perf_counter() - t0)
        if sleep_s > 0:
            precise_sleep(sleep_s)


def _episode_frame_range(ds: LeRobotDataset, episode_index: int) -> tuple[int, int]:
    """兼容不同 LeRobot metadata 版本里的 episode 起止字段。"""
    episode = ds.meta.episodes[episode_index]
    if "from" in episode and "to" in episode:
        return int(episode["from"]), int(episode["to"])
    if "dataset_from_index" in episode and "dataset_to_index" in episode:
        return int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
    if "length" in episode:
        start = 0
        for idx in range(episode_index):
            start += int(ds.meta.episodes[idx]["length"])
        return start, start + int(episode["length"])
    raise KeyError(f"无法从 episode metadata 解析帧范围：{episode}")


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy()
    return value


def _tensor_to_raw_image(image: torch.Tensor) -> np.ndarray:
    """把 LeRobot 数据集里的 CHW float 图像转回推理入口需要的 HWC uint8。"""
    image = image.detach().cpu().clamp(0, 1)
    return (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def _sample_to_offline_observation(sample: dict) -> dict[str, np.ndarray]:
    return {
        "observation.state": _to_numpy(sample["observation.state"]).astype(np.float32),
        "observation.images.top": _tensor_to_raw_image(sample["observation.images.top"]),
        "observation.images.right_wrist": _tensor_to_raw_image(
            sample["observation.images.right_wrist"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ACT 真机执行：严格仅控右臂 6 维")
    parser.add_argument("--policy", default=DEFAULT_POLICY, help="pretrained_model 路径")
    parser.add_argument("--dataset-root", default=DEFAULT_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--device", default=CONFIG.train_device)
    parser.add_argument(
        "--episode-time",
        type=float,
        default=0.0,
        help="执行时长；<=0 表示不限时，直到 Ctrl+C 中止",
    )
    parser.add_argument("--fps", type=int, default=CONFIG.cam_fps)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--max-relative-target", type=float, default=15.0)
    parser.add_argument("--preview-fps", type=int, default=10)
    parser.add_argument("--display-data", action="store_true", default=True)
    parser.add_argument("--display-ip", type=str, default=None)
    parser.add_argument("--display-port", type=int, default=None)
    parser.add_argument("--display-compressed-images", action="store_true")
    parser.add_argument("--yes", action="store_true", help="跳过 Enter 确认，不建议真机首测使用")
    parser.add_argument(
        "--print-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="执行时打印每个右臂关节的 raw/sent 差值，用于排查是否只有某个关节在动",
    )
    parser.add_argument("--print-every", type=int, default=30, help="每隔多少个控制周期打印一次")
    parser.add_argument(
        "--offline-check",
        action="store_true",
        help="只用数据集样本做离线检查，不连接机器人，不发送 action",
    )
    parser.add_argument("--episode-index", type=int, default=0, help="离线检查使用的 episode 下标")
    parser.add_argument("--max-frames", type=int, default=120, help="离线检查最多预测多少帧")
    parser.add_argument(
        "--abort-start-delta",
        type=float,
        default=40.0,
        help="开始前第一步 action 到当前右臂 state 的最大差值超过该值则中止",
    )
    parser.add_argument(
        "--abort-total-delta",
        type=float,
        default=160.0,
        help="执行中任一右臂关节相对启动姿态超过该值则停止继续发送；<=0 表示关闭",
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
    if args.max_relative_target <= 0:
        print("❌ 真机执行必须设置正数 --max-relative-target，首测建议 2。")
        return 1
    if not args.offline_check and not _validate_robot_io_config():
        return 2

    CONFIG.banner("ACT 真机执行：严格仅控右臂 6 维")
    print(f"  policy           : {policy_path}")
    print(f"  dataset metadata : {args.dataset_root}")
    run_time_desc = "不限时，直到 Ctrl+C" if args.episode_time <= 0 else f"{args.episode_time}s"
    print(f"  执行时长         : {run_time_desc} @ {args.fps}Hz")
    print(f"  单步限幅         : {args.max_relative_target}")
    print(f"  task             : {args.task}")
    print("  ⚠️ 不使用 lerobot-record，不创建 16 维整机 action schema\n")

    print("🎯 加载训练数据集 metadata，强制使用训练时 6 维右臂 schema")
    ds = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.dataset_root,
        video_backend=args.video_backend,
    )
    action_names = _validate_schema(ds)
    for i, name in enumerate(action_names):
        print(f"  {i}: {name}")
    print()

    device = get_safe_torch_device(args.device, log=True)
    policy_cfg = PreTrainedConfig.from_pretrained(
        policy_path,
        cli_overrides=[f"--device={device.type}"],
    )
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

    if args.offline_check:
        episodes = ds.meta.episodes
        if args.episode_index < 0 or args.episode_index >= len(episodes):
            print(f"❌ episode-index 越界：{args.episode_index}, 总数 {len(episodes)}")
            return 1
        start, end = _episode_frame_range(ds, args.episode_index)
        if args.max_frames > 0:
            end = min(end, start + args.max_frames)

        print("🎯 离线检查：不连接机器人，不发送 action")
        print(f"  episode    : {args.episode_index}")
        print(f"  frame range: {start}..{end - 1} ({end - start} frames)")

        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        pred_values = []
        target_values = []
        bad_count = 0
        max_step_delta = 0.0
        prev_action = None

        for frame_idx in range(start, end):
            sample = ds[frame_idx]
            observation_frame = _sample_to_offline_observation(sample)
            action_tensor = predict_action(
                observation=observation_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=args.task,
                robot_type="xlerobot_2wheels",
            )
            action = make_robot_action(action_tensor, ds.features)
            if list(action) != action_names:
                print(f"❌ 第 {frame_idx} 帧输出 key 异常：{list(action)}")
                return 2
            values = _action_values(action, action_names)
            if not np.isfinite(values).all():
                bad_count += 1
            if prev_action is not None:
                max_step_delta = max(max_step_delta, float(np.max(np.abs(values - prev_action))))
            prev_action = values
            pred_values.append(values)
            target_values.append(_to_numpy(sample["action"]))

        pred = np.stack(pred_values)
        target = np.stack(target_values)
        mae = np.mean(np.abs(pred - target), axis=0)
        print("✅ 离线检查完成")
        print(f"  NaN/inf frames : {bad_count}")
        print(f"  max step delta : {max_step_delta:.3f}")
        print(f"  MAE avg        : {float(np.mean(mae)):.3f}")
        for name, value in zip(action_names, mae, strict=True):
            short = name.removeprefix("right_arm_").removesuffix(".pos")
            print(f"  {short:14s}: MAE={float(value):.3f}")
        print()
        print("🎯 右臂发送映射检查")
        for name in action_names:
            print(f"  {name} -> bus2 Goal_Position[{name.replace('.pos', '')}]")
        print("  左臂/头部/底盘：不会写入")
        return 0

    robot_cfg = XLerobot2WheelsConfig(
        port1=CONFIG.follower_port1,
        port2=CONFIG.follower_port2,
        id=CONFIG.follower_id,
        calibration_dir=args.calibration_dir,
        cameras=_cameras_config(),
        max_relative_target=None,
    )
    robot = XLerobot2Wheels(robot_cfg)
    if not robot.calibration_fpath.is_file():
        print("❌ 没找到机器人标定文件，已停止执行，避免自动进入标定流程。")
        print(f"   期望路径: {robot.calibration_fpath}")
        return 2

    try:
        robot.connect(calibrate=False)
        if args.display_data:
            init_rerun(session_name="act_right_arm_eval", ip=args.display_ip, port=args.display_port)

        if not args.yes:
            _wait_for_enter_with_rerun(
                robot=robot,
                fps=args.preview_fps,
                compress_images=args.display_compressed_images,
            )
        else:
            print("⚠️ 已跳过 Enter 确认，将直接执行。")

        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

        # 开始前做一次第一步 action 安全检查；不发送。
        obs = robot.get_observation()
        start_state = {name: float(obs[name]) for name in action_names}
        first_action = _predict_right_arm_action(
            obs=obs,
            ds=ds,
            action_names=action_names,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            task=args.task,
            robot_type=robot.robot_type,
        )
        max_start_delta, worst_joint = _max_to_current(first_action, obs, action_names)
        if args.print_actions:
            print("🎯 第一帧 raw action - 当前右臂 state")
            print(f"   {_format_joint_deltas(first_action, obs, action_names)}")
        if max_start_delta > args.abort_start_delta:
            print(
                f"❌ 第一帧 action 离当前右臂过远：{max_start_delta:.2f} "
                f"({worst_joint})，已中止，未发送 action。"
            )
            return 3
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

        print("🚦 开始发送右臂 6 维 action。按 Ctrl+C 可中止。")
        deadline = None if args.episode_time <= 0 else time.perf_counter() + args.episode_time
        period = 1.0 / max(args.fps, 1)
        tick = 0
        sent_count = 0
        prev_sent: dict[str, float] | None = None

        while deadline is None or time.perf_counter() < deadline:
            t0 = time.perf_counter()
            obs = robot.get_observation()
            action = _predict_right_arm_action(
                obs=obs,
                ds=ds,
                action_names=action_names,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                task=args.task,
                robot_type=robot.robot_type,
            )
            candidate_action = _safe_goal_from_present(
                action=action,
                present_pos={name: float(obs[name]) for name in action_names},
                action_names=action_names,
                max_relative_target=args.max_relative_target,
            )
            max_total_delta, total_worst_joint = _max_delta_from_reference(
                candidate_action,
                start_state,
                action_names,
            )
            if args.abort_total_delta > 0 and max_total_delta > args.abort_total_delta:
                print(
                    f"⚠️ 右臂相对启动姿态变化过大：{max_total_delta:.2f} "
                    f"({total_worst_joint})，停止继续发送。"
                )
                break
            sent_action = _send_right_arm_only(
                robot=robot,
                action=action,
                action_names=action_names,
                max_relative_target=args.max_relative_target,
            )
            if args.print_actions and tick % max(args.print_every, 1) == 0:
                print(f"[{tick:04d}] raw-current : {_format_joint_deltas(action, obs, action_names)}")
                print(f"       sent-current: {_format_joint_deltas(sent_action, obs, action_names)}")
            sent_count += 1
            prev_sent = sent_action

            if args.display_data:
                log_rerun_data(
                    observation=obs,
                    action=sent_action,
                    compress_images=args.display_compressed_images,
                )

            tick += 1
            sleep_s = period - (time.perf_counter() - t0)
            if sleep_s > 0:
                precise_sleep(sleep_s)

        print(f"✅ 执行结束：发送 {sent_count} 帧右臂 action。")
        if prev_sent is not None:
            compact = " | ".join(
                f"{name.removeprefix('right_arm_').removesuffix('.pos')}={prev_sent[name]:.2f}"
                for name in action_names
            )
            print(f"   最后一帧已发送 action: {compact}")
    except KeyboardInterrupt:
        print("\n🛑 用户中止，停止发送 policy action。")
        return 130
    finally:
        if robot.is_connected:
            robot.disconnect()
        print("✅ 机器人已断开连接。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
