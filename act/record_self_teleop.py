#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""act/record_self_teleop.py — XLeRobot 自我遥操作 ACT 数据采集（核心）。

🎯 主流程：
  1. 连 XLeRobot，关左臂扭矩
  2. 主循环：读左臂 → 镜像 → 写右臂；同时录入 right_arm/head/相机/action 到数据集
  3. →/s 保存当前段，←/r 重录，Esc/q 停止

数据集 schema：
  observation.state              float32 (6,)  右臂 6 关节
  observation.images.top         video         桌面相机
  observation.images.right_wrist video         右腕相机
  action                         float32 (6,)  右臂 6 关节目标位置（来自镜像）

> 头部 / 底盘**都不录、都不下发**（仅左臂关扭矩，右臂跟随）。
> 头部电机扭矩保持开启（伺服锁位），不会飘。

使用：
    python act/record_self_teleop.py
    python act/record_self_teleop.py --resume
    python act/record_self_teleop.py --num-episodes=80 --task="..."

⌨️ 热键（必须本地物理终端）：
    → 或 s    本段成功，保存
    ← 或 r    本段失败，丢弃重录
    Esc 或 q  整体停止
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# 兜底：允许 `python act/record_self_teleop.py` 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.robots.xlerobot_2wheels import XLerobot2Wheels, XLerobot2WheelsConfig
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.control_utils import is_headless
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from act.config import CONFIG
from act.mirror import (
    NEGATE_JOINTS,
    extract_head,
    extract_left_arm,
    extract_right_arm,
    limit_action_relative_to_observation,
    mirror_left_delta_to_right_target,
    mirror_left_to_right,
)


# ============================================================
# 数据集 schema
# ============================================================

# state 顺序：只录右臂 6 维（与 action 同维度，最小化 schema）
STATE_NAMES: list[str] = [
    "right_arm_shoulder_pan.pos",
    "right_arm_shoulder_lift.pos",
    "right_arm_elbow_flex.pos",
    "right_arm_wrist_flex.pos",
    "right_arm_wrist_roll.pos",
    "right_arm_gripper.pos",
]

# action 顺序：右臂 6 维
ACTION_NAMES: list[str] = [
    "right_arm_shoulder_pan.pos",
    "right_arm_shoulder_lift.pos",
    "right_arm_elbow_flex.pos",
    "right_arm_wrist_flex.pos",
    "right_arm_wrist_roll.pos",
    "right_arm_gripper.pos",
]

CAMERA_NAMES: list[str] = ["top", "right_wrist"]


def _make_features() -> dict[str, dict]:
    """返回 LeRobotDataset.create 所需的 features 字典。"""
    h, w = CONFIG.cam_height, CONFIG.cam_width
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES),),
            "names": STATE_NAMES,
        },
        "observation.images.top": {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.right_wrist": {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(ACTION_NAMES),),
            "names": ACTION_NAMES,
        },
    }


def _dataset_root() -> Path:
    """返回本地数据集目录，兼容 create 与 resume 写入模式。"""
    return HF_LEROBOT_HOME / CONFIG.repo_id


def _build_robot() -> XLerobot2Wheels:
    cameras = {
        "top": OpenCVCameraConfig(
            index_or_path=CONFIG.cam_top,
            width=CONFIG.cam_width,
            height=CONFIG.cam_height,
            fps=CONFIG.cam_fps,
        ),
        "right_wrist": OpenCVCameraConfig(
            index_or_path=CONFIG.cam_right_wrist,
            width=CONFIG.cam_width,
            height=CONFIG.cam_height,
            fps=CONFIG.cam_fps,
        ),
    }
    cfg = XLerobot2WheelsConfig(
        port1=CONFIG.follower_port1,
        port2=CONFIG.follower_port2,
        id=CONFIG.follower_id,
        cameras=cameras,
        max_relative_target=None,
    )
    return XLerobot2Wheels(cfg)


def _disconnect_quietly(robot: XLerobot2Wheels) -> None:
    try:
        robot.disconnect()
    except Exception as exc:
        logging.warning(f"断开机器人时忽略异常：{exc}")


def _init_record_keyboard_listener():
    """录制热键：方向键和字母键都支持，降低现场键盘兼容风险。"""
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }

    if is_headless():
        logging.warning("Headless 环境下无法监听键盘；请在本地物理终端运行采集。")
        return None, events

    from pynput import keyboard

    def on_press(key):
        try:
            char = getattr(key, "char", None)
            if key == keyboard.Key.right or char == "s":
                print("保存当前 episode：→ / s")
                events["exit_early"] = True
            elif key == keyboard.Key.left or char == "r":
                print("丢弃并重录当前 episode：← / r")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc or char == "q":
                print("停止整体采集：Esc / q")
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as exc:  # noqa: BLE001 - 键盘回调不能让线程崩掉
            print(f"处理按键失败：{exc}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, events


def _build_state_vector(obs: dict) -> np.ndarray:
    """从 robot.get_observation() 输出里抽出 state 向量。"""
    return np.array([float(obs[k]) for k in STATE_NAMES], dtype=np.float32)


def _build_action_vector(right_target: dict[str, float]) -> np.ndarray:
    return np.array([float(right_target[k]) for k in ACTION_NAMES], dtype=np.float32)


def _load_reset_home_from_file(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_home = payload.get("right_home", payload)
    missing = [name for name in ACTION_NAMES if name not in raw_home]
    if missing:
        raise ValueError(f"reset home 文件缺少字段：{missing}，文件：{path}")
    return {name: float(raw_home[name]) for name in ACTION_NAMES}


def _max_abs_right_error(obs: dict, right_target: dict[str, float]) -> float:
    return max(abs(float(obs[k]) - float(right_target[k])) for k in ACTION_NAMES)


def _log_preview_frame(robot: XLerobot2Wheels, action: dict[str, float], display_data: bool) -> None:
    """等待操作者按回车前写一帧预览，避免 Rerun 空白。"""
    if not display_data:
        return
    obs = robot.get_observation()
    log_rerun_data(observation=obs, action=action)


def _move_right_arm_to_home(
    robot: XLerobot2Wheels,
    right_home: dict[str, float],
    fps: int,
    max_relative_target: float | None,
    display_data: bool,
    timeout_s: float = 12.0,
    tolerance: float = 2.0,
) -> None:
    """复位阶段：只把右臂慢速送回固定 home，不写数据集。"""
    period = 1.0 / fps
    step_limit = max_relative_target if max_relative_target is not None else CONFIG.max_relative_target
    if step_limit <= 0:
        step_limit = 10.0

    log_say("Reset: moving right arm to fixed home", play_sounds=False)
    t_start = time.perf_counter()
    last_error = float("inf")

    while True:
        t0 = time.perf_counter()
        obs = robot.get_observation()
        last_error = _max_abs_right_error(obs, right_home)
        if last_error <= tolerance:
            if display_data:
                log_rerun_data(observation=obs, action=right_home)
            break

        action_to_send = limit_action_relative_to_observation(
            right_home,
            obs,
            step_limit,
        )
        sent_action = robot.send_action(action_to_send)
        if display_data:
            log_rerun_data(observation=obs, action=sent_action)

        if (time.perf_counter() - t_start) >= timeout_s:
            logging.warning(
                "右臂复位超时：未完全到达 home，当前最大误差 %.2f，将继续进入人工复位阶段",
                last_error,
            )
            break

        elapsed = time.perf_counter() - t0
        precise_sleep(max(period - elapsed, 0.0))

    logging.info("✅ 右臂 home 复位结束，最大误差 %.2f", last_error)


def _record_episode(
    robot: XLerobot2Wheels,
    dataset: LeRobotDataset,
    events: dict,
    fps: int,
    episode_time_s: int,
    task: str,
    display_data: bool,
    left_origin: dict[str, float],
    right_origin: dict[str, float],
    absolute: bool,
    max_relative_target: float | None,
) -> None:
    """录一段。返回时事件由调用方决定是 save 还是 clear。"""
    period = 1.0 / fps
    t_start = time.perf_counter()

    while True:
        t0 = time.perf_counter()

        # 1. 读 obs（含双臂、头、相机）—— 但只取右臂入 state
        obs = robot.get_observation()

        # 2. 计算右臂目标
        left_pos = extract_left_arm(obs)
        if absolute:
            right_target = mirror_left_to_right(left_pos)
        else:
            right_target = mirror_left_delta_to_right_target(
                left_current=left_pos,
                left_origin=left_origin,
                right_origin=right_origin,
            )

        # 3. 下发：仅右臂；头部 / 底盘都不传，由电机自身扭矩维持
        action_to_send = limit_action_relative_to_observation(
            right_target,
            obs,
            max_relative_target,
        )
        sent_action = robot.send_action(action_to_send)

        # 4. 构造 frame 并入库
        frame = {
            "observation.state": _build_state_vector(obs),
            "observation.images.top": obs["top"],
            "observation.images.right_wrist": obs["right_wrist"],
            "action": _build_action_vector(action_to_send),
            "task": task,
        }
        dataset.add_frame(frame)

        if display_data:
            log_rerun_data(observation=obs, action=sent_action)

        # 5. 终止条件
        if events["exit_early"]:
            events["exit_early"] = False
            break
        if episode_time_s > 0 and (time.perf_counter() - t_start) >= episode_time_s:
            break

        elapsed = time.perf_counter() - t0
        precise_sleep(max(period - elapsed, 0.0))


def main() -> int:
    parser = argparse.ArgumentParser(description="XLeRobot 自我遥操作 ACT 数据采集")
    parser.add_argument("--resume", action="store_true", help="继续追加到已有数据集")
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument(
        "--episode-time",
        type=int,
        default=None,
        help="单段最大秒数；默认 0=手动按 →/s、←/r、Esc/q 结束",
    )
    parser.add_argument("--reset-time", type=int, default=None, help="段间复位秒数；默认 0=手动按回车继续")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="使用绝对位置镜像；默认使用 delta 变化量跟随",
    )
    parser.add_argument("--root", type=Path, default=None, help="本地数据集目录，默认 $HF_LEROBOT_HOME/<repo_id>")
    parser.add_argument(
        "--max-relative-target",
        type=float,
        default=None,
        help="单次 action 相对当前位置最大变化；默认 10.0，可用 MAX_RELATIVE_TARGET 覆盖，<=0 关闭",
    )
    args = parser.parse_args()

    init_logging()

    num_episodes = args.num_episodes or CONFIG.num_episodes
    episode_time_s = CONFIG.episode_time_s if args.episode_time is None else args.episode_time
    reset_time_s = CONFIG.reset_time_s if args.reset_time is None else args.reset_time
    task = args.task or CONFIG.task_desc
    push_to_hub = args.push_to_hub or CONFIG.push_to_hub
    display_data = not args.no_display
    fps = CONFIG.cam_fps
    dataset_root = args.root or _dataset_root()
    max_relative_target = (
        args.max_relative_target
        if args.max_relative_target is not None
        else CONFIG.max_relative_target
    )
    if max_relative_target <= 0:
        max_relative_target = None

    if push_to_hub and not CONFIG.hf_user:
        raise RuntimeError("上传到 HuggingFace Hub 前必须设置 HF_USER；本地采集可不设置。")

    CONFIG.banner("XLeRobot 自我遥操作数据采集 ⭐")
    print(f"  数据集     : {CONFIG.repo_id}")
    episode_desc = "手动结束" if episode_time_s <= 0 else f"最多 {episode_time_s}s"
    reset_desc = "右臂回 home 后手动继续" if reset_time_s <= 0 else f"右臂回 home 后等待 {reset_time_s}s"
    print(f"  采集计划   : {num_episodes} 段，每段{episode_desc}，复位 {reset_desc}")
    print(f"  task       : {task}")
    print(f"  root       : {dataset_root}")
    print(f"  push hub   : {push_to_hub}")
    print(f"  resume     : {args.resume}")
    print(f"  Rerun      : {display_data}")
    print(f"  跟随模式   : {'absolute' if args.absolute else 'delta'}")
    print(f"  镜像取反   : {set(NEGATE_JOINTS)}")
    print()
    print("⌨️  热键（务必本地物理终端）：→ 或 s 保存 / ← 或 r 重录 / Esc 或 q 停止")
    print("⚠️  即将关闭左臂扭矩 —— 请先用手扶住左臂！")
    print()
    input("准备好按回车连接机器人 ... ")

    # 1. 机器人
    robot = _build_robot()
    try:
        robot.connect()
    except Exception:
        _disconnect_quietly(robot)
        raise
    robot.bus1.disable_torque(robot.left_arm_motors)
    logging.info(f"✅ 左臂扭矩关闭：{robot.left_arm_motors}")
    logging.info("ℹ️  头部 / 底盘电机扭矩保持开启，不主动控制（自然伺服锁位）")

    right_reset_home = _load_reset_home_from_file(CONFIG.reset_home_path)
    if right_reset_home is None:
        reset_home_obs = robot.get_observation()
        right_reset_home = extract_right_arm(reset_home_obs)
        logging.warning(
            "未找到固定 reset home 文件 %s；本次退回使用启动时右臂当前姿态。"
            "建议先运行：python act/capture_reset_home.py",
            CONFIG.reset_home_path,
        )
    else:
        logging.info("✅ 已加载固定右臂 reset home：%s", CONFIG.reset_home_path)

    # 2. 数据集
    features = _make_features()
    if args.resume:
        dataset = LeRobotDataset.resume(
            CONFIG.repo_id,
            root=dataset_root,
            streaming_encoding=True,
            encoder_threads=2,
        )
    else:
        dataset = LeRobotDataset.create(
            CONFIG.repo_id,
            fps=fps,
            root=dataset_root,
            robot_type=robot.name,
            features=features,
            use_videos=True,
            streaming_encoding=True,
            encoder_threads=2,
        )

    # 3. 键盘 + Rerun
    listener, events = _init_record_keyboard_listener()
    if display_data:
        init_rerun(session_name="self_teleop_record")

    # 4. 录制循环
    try:
        with VideoEncodingManager(dataset):
            recorded = 0
            _move_right_arm_to_home(
                robot=robot,
                right_home=right_reset_home,
                fps=fps,
                max_relative_target=max_relative_target,
                display_data=display_data,
            )
            _log_preview_frame(robot, right_reset_home, display_data)
            input(f"右臂已回 home，摆好物体后按回车开始第 {dataset.num_episodes} 段 ... ")

            while recorded < num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {dataset.num_episodes}", play_sounds=False)
                origin_obs = robot.get_observation()
                left_origin = extract_left_arm(origin_obs)
                right_origin = extract_right_arm(origin_obs)
                logging.info("✅ 已记录本段左右臂起始姿态，delta 模式会按变化量跟随")

                _record_episode(
                    robot=robot,
                    dataset=dataset,
                    events=events,
                    fps=fps,
                    episode_time_s=episode_time_s,
                    task=task,
                    display_data=display_data,
                    left_origin=left_origin,
                    right_origin=right_origin,
                    absolute=args.absolute,
                    max_relative_target=max_relative_target,
                )

                # 处理段尾事件
                if events["rerecord_episode"]:
                    log_say("Re-record episode", play_sounds=False)
                    events["rerecord_episode"] = False
                    dataset.clear_episode_buffer()
                else:
                    dataset.save_episode()
                    recorded += 1

                # 每段结束后先回固定 home。这个过程不写入数据集。
                if not events["stop_recording"]:
                    _move_right_arm_to_home(
                        robot=robot,
                        right_home=right_reset_home,
                        fps=fps,
                        max_relative_target=max_relative_target,
                        display_data=display_data,
                    )

                # 段间复位：物体/场景由操作者手动摆好，再进入下一段。
                if not events["stop_recording"] and recorded < num_episodes:
                    if reset_time_s > 0:
                        log_say(f"Reset: please reposition objects ({reset_time_s}s)", play_sounds=False)
                        t0 = time.perf_counter()
                        while time.perf_counter() - t0 < reset_time_s and not events["exit_early"]:
                            # reset 期间锁住右臂当前位置（防右臂下垂）；头部不动
                            obs = robot.get_observation()
                            right_lock = extract_right_arm(obs)
                            robot.send_action(right_lock)
                            precise_sleep(1.0 / fps)
                        events["exit_early"] = False
                    else:
                        log_say("Reset: reposition objects, then press ENTER", play_sounds=False)
                        _log_preview_frame(robot, right_reset_home, display_data)
                        input(f"复位好后按回车开始第 {dataset.num_episodes} 段 ... ")

        # 5. 收尾
        dataset.finalize()
        if push_to_hub:
            log_say("Pushing dataset to hub", play_sounds=False)
            dataset.push_to_hub()
        dataset = None

    finally:
        if dataset is not None:
            dataset.finalize()
        if listener is not None:
            listener.stop()
        _disconnect_quietly(robot)
        print("\n✅ 已安全断开。")
        print(f"   数据集存于：{dataset_root}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
