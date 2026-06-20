#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""离线检查 ACT checkpoint：只在数据集上推理，不连接机器人。

检查目标：
  1. action 是否出现 NaN / inf
  2. 反归一化后的 action 是否落在数据集统计范围附近
  3. 预测 action 曲线是否平滑
  4. state/action 维度和关节顺序是否一致
  5. 导出 CSV，方便后续画曲线人工检查夹爪时机

默认检查整条 episode；快速 smoke test 时再显式传 --max-frames。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.control_utils import predict_action
from lerobot.utils.device_utils import get_safe_torch_device


DEFAULT_POLICY = (
    "outputs/train/"
    "xlerobot_act_pick_place_clean_le400_act_a100_80k_bs64_chunk60_quantiles/"
    "checkpoints/last/pretrained_model"
)
DEFAULT_ROOT = "dataset/zhoushangyu77/xlerobot_act_pick_place_clean_le400"
DEFAULT_REPO = "zhoushangyu77/xlerobot_act_pick_place_clean_le400"

EXPECTED_NAMES = [
    "right_arm_shoulder_pan.pos",
    "right_arm_shoulder_lift.pos",
    "right_arm_elbow_flex.pos",
    "right_arm_wrist_flex.pos",
    "right_arm_wrist_roll.pos",
    "right_arm_gripper.pos",
]


def _short_name(name: str) -> str:
    return name.removeprefix("right_arm_").removesuffix(".pos")


def _tensor_to_raw_image(image: torch.Tensor) -> np.ndarray:
    """把 LeRobot 数据集里的 CHW float 图像转回推理入口需要的 HWC uint8。"""
    image = image.detach().cpu().clamp(0, 1)
    return (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def _sample_to_observation(sample: dict) -> dict[str, np.ndarray]:
    return {
        "observation.state": sample["observation.state"].detach().cpu().numpy().astype(np.float32),
        "observation.images.top": _tensor_to_raw_image(sample["observation.images.top"]),
        "observation.images.right_wrist": _tensor_to_raw_image(
            sample["observation.images.right_wrist"]
        ),
    }


def _episode_indices(ds: LeRobotDataset, episode_index: int, start_frame: int, max_frames: int, stride: int):
    ep = ds.meta.episodes[episode_index]
    start = int(ep["dataset_from_index"]) + start_frame
    stop = int(ep["dataset_to_index"])
    if max_frames > 0:
        stop = min(stop, start + max_frames * stride)
    return list(range(start, stop, stride))


def _as_np_stats(ds: LeRobotDataset, key: str, stat_name: str) -> np.ndarray | None:
    stats = ds.meta.stats.get(key, {})
    if stat_name not in stats:
        return None
    return torch.as_tensor(stats[stat_name]).detach().cpu().numpy().astype(np.float32)


def _print_schema(ds: LeRobotDataset) -> None:
    state_names = ds.features["observation.state"]["names"]
    action_names = ds.features["action"]["names"]
    print("🎯 schema 检查")
    print(f"  state dim : {len(state_names)}")
    print(f"  action dim: {len(action_names)}")
    print("  state 顺序 :")
    for i, name in enumerate(state_names):
        print(f"    {i}: {name}")
    print("  action 顺序:")
    for i, name in enumerate(action_names):
        print(f"    {i}: {name}")

    if state_names != EXPECTED_NAMES or action_names != EXPECTED_NAMES:
        print("\n⚠️  state/action 顺序和预期不完全一致，先不要上真机。")
        raise SystemExit(2)
    print("  ✅ state/action 顺序一致\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="离线检查 ACT checkpoint 输出动作")
    parser.add_argument("--policy", default=DEFAULT_POLICY, help="pretrained_model 路径")
    parser.add_argument("--dataset-root", default=DEFAULT_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 表示检查整条 episode")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--task", default="pick up the object and place it into the target container")
    parser.add_argument("--output-csv", default="outputs/act_offline_check.csv")
    args = parser.parse_args()

    policy_path = Path(args.policy)
    if not policy_path.exists():
        print(f"❌ 找不到 policy: {policy_path}")
        return 1

    print("🎯 加载数据集")
    ds = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.dataset_root,
        video_backend=args.video_backend,
    )
    print(f"  episodes={ds.num_episodes}, frames={ds.num_frames}, fps={ds.fps}")
    _print_schema(ds)

    print("🎯 加载 policy 和 processor")
    device = get_safe_torch_device(args.device, log=True)
    policy_cfg = PreTrainedConfig.from_pretrained(
        policy_path,
        cli_overrides=[f"--device={device.type}"],
    )
    # checkpoint 会覆盖权重；离线检查时不需要再下载 torchvision 的 ImageNet 初始化权重。
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

    indices = _episode_indices(
        ds,
        episode_index=args.episode_index,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        stride=args.stride,
    )
    print(
        "🎯 开始离线推理 "
        f"episode={args.episode_index}, frames={len(indices)}, stride={args.stride}"
    )

    action_names = ds.features["action"]["names"]
    short_names = [_short_name(name) for name in action_names]
    action_min = _as_np_stats(ds, "action", "min")
    action_max = _as_np_stats(ds, "action", "max")
    action_q01 = _as_np_stats(ds, "action", "q01")
    action_q99 = _as_np_stats(ds, "action", "q99")

    rows: list[dict[str, float | int]] = []
    pred_list = []
    target_list = []
    state_list = []

    for n, idx in enumerate(indices):
        sample = ds[idx]
        obs = _sample_to_observation(sample)
        action_tensor = predict_action(
            observation=obs,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=args.task,
            robot_type="xlerobot_2wheels",
        )
        action_dict = make_robot_action(action_tensor, ds.features)
        pred = np.array([action_dict[name] for name in action_names], dtype=np.float32)
        target = sample["action"].detach().cpu().numpy().astype(np.float32)
        state = sample["observation.state"].detach().cpu().numpy().astype(np.float32)

        pred_list.append(pred)
        target_list.append(target)
        state_list.append(state)

        row: dict[str, float | int] = {
            "global_index": int(idx),
            "episode_index": int(sample["episode_index"]),
            "frame_index": int(sample["frame_index"]),
            "timestamp": float(sample["timestamp"]),
        }
        for i, name in enumerate(short_names):
            row[f"pred_{name}"] = float(pred[i])
            row[f"target_{name}"] = float(target[i])
            row[f"state_{name}"] = float(state[i])
            row[f"abs_err_{name}"] = float(abs(pred[i] - target[i]))
        rows.append(row)

        if n == 0 or (n + 1) % 50 == 0 or n == len(indices) - 1:
            print(f"  推理进度: {n + 1}/{len(indices)}")

    pred_arr = np.stack(pred_list)
    target_arr = np.stack(target_list)
    state_arr = np.stack(state_list)

    finite_mask = np.isfinite(pred_arr)
    has_bad = not bool(finite_mask.all())
    step_delta = np.abs(np.diff(pred_arr, axis=0)) if len(pred_arr) > 1 else np.zeros_like(pred_arr)
    abs_err = np.abs(pred_arr - target_arr)

    print("\n🎯 数值检查")
    print(f"  NaN/inf: {'❌ 有' if has_bad else '✅ 无'}")
    print(f"  pred min/max: {pred_arr.min(axis=0).round(3).tolist()} / {pred_arr.max(axis=0).round(3).tolist()}")
    print(f"  target MAE : {abs_err.mean(axis=0).round(3).tolist()}")
    print(f"  step |Δ| mean: {step_delta.mean(axis=0).round(3).tolist()}")
    print(f"  step |Δ| p95 : {np.percentile(step_delta, 95, axis=0).round(3).tolist()}")
    print(f"  step |Δ| max : {step_delta.max(axis=0).round(3).tolist()}")

    if action_min is not None and action_max is not None:
        outside_minmax = ((pred_arr < action_min) | (pred_arr > action_max)).sum(axis=0)
        print("\n🎯 action min/max 范围检查")
        for i, name in enumerate(short_names):
            print(
                f"  {name:14s}: pred=[{pred_arr[:, i].min():8.3f}, {pred_arr[:, i].max():8.3f}] "
                f"data=[{action_min[i]:8.3f}, {action_max[i]:8.3f}] "
                f"outside={int(outside_minmax[i])}"
            )

    if action_q01 is not None and action_q99 is not None:
        outside_q = ((pred_arr < action_q01) | (pred_arr > action_q99)).sum(axis=0)
        print("\n🎯 action q01/q99 主分布检查")
        for i, name in enumerate(short_names):
            ratio = outside_q[i] / max(len(pred_arr), 1)
            print(
                f"  {name:14s}: q=[{action_q01[i]:8.3f}, {action_q99[i]:8.3f}] "
                f"outside={int(outside_q[i])}/{len(pred_arr)} ({ratio:.1%})"
            )

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n💾 CSV 已保存: {out_csv}")

    summary_path = out_csv.with_suffix(".summary.json")
    summary = {
        "policy": str(policy_path),
        "dataset_root": args.dataset_root,
        "episode_index": args.episode_index,
        "num_frames_checked": len(indices),
        "has_nan_or_inf": has_bad,
        "action_names": action_names,
        "pred_min": pred_arr.min(axis=0).tolist(),
        "pred_max": pred_arr.max(axis=0).tolist(),
        "target_mae": abs_err.mean(axis=0).tolist(),
        "step_abs_delta_p95": np.percentile(step_delta, 95, axis=0).tolist(),
        "state_min": state_arr.min(axis=0).tolist(),
        "state_max": state_arr.max(axis=0).tolist(),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"💾 摘要已保存: {summary_path}")

    return 1 if has_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
