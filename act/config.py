#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""act/config.py — 自我遥操作 ACT 工作流的集中配置。

🎯 角色：XLeRobot 一台机器，左臂当 leader（人手推），右臂当 follower（被记录）。

修改本文件以改默认值；或通过环境变量临时覆盖（每个字段说明里都给了变量名）。

使用：
    from act.config import CONFIG
    CONFIG.banner("启动 XXX")
直接执行：
    python -m act.config
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


@dataclass
class Config:
    """全局配置 —— 仅 2 路 USB（XLeRobot 双总线）。"""

    # ===== XLeRobot 串口（仅 2 路，即整机本身的两根总线）=====
    follower_port1: str = field(default_factory=lambda: _env("FOLLOWER_PORT1", "/dev/ttyACM0"))
    """bus1：左臂 6 电机 + 头部 2 电机；环境变量 FOLLOWER_PORT1。"""

    follower_port2: str = field(default_factory=lambda: _env("FOLLOWER_PORT2", "/dev/ttyACM1"))
    """bus2：右臂 6 电机 + 两轮底盘 2 电机；环境变量 FOLLOWER_PORT2。"""

    follower_id: str = field(default_factory=lambda: _env("FOLLOWER_ID", "xlerobot_main"))
    """XLeRobot 实例 ID（用于标定文件命名）；环境变量 FOLLOWER_ID。"""

    # ===== 相机（都在 XLeRobot 上）=====
    cam_top: str = field(default_factory=lambda: _env("CAM_TOP", "/dev/video0"))
    """桌面外部相机；环境变量 CAM_TOP。录入数据集。"""

    cam_right_wrist: str = field(default_factory=lambda: _env("CAM_RIGHT_WRIST", "/dev/video2"))
    """右腕相机；环境变量 CAM_RIGHT_WRIST。录入数据集。"""

    cam_width: int = int(_env("CAM_W", "640"))
    cam_height: int = int(_env("CAM_H", "480"))
    cam_fps: int = int(_env("CAM_FPS", "30"))

    # ===== 真机安全 =====
    max_relative_target: float = float(_env("MAX_RELATIVE_TARGET", "15.0"))
    """单次发送 action 相对当前位置的最大变化；<=0 表示关闭限幅。"""

    # ===== 数据集 =====
    hf_user: str = field(default_factory=lambda: _env("HF_USER", ""))
    dataset_name: str = field(default_factory=lambda: _env("DATASET_NAME", "xlerobot_act_self_teleop"))
    dataset_root: Path = field(default_factory=lambda: Path(_env("DATASET_ROOT", "dataset")))
    """本地数据集根目录；默认写入项目内 dataset/，环境变量 DATASET_ROOT。"""

    num_episodes: int = int(_env("NUM_EPISODES", "50"))
    episode_time_s: int = int(_env("EPISODE_TIME_S", "0"))
    """单段最大时长；0 表示手动按键结束。"""

    reset_time_s: int = int(_env("RESET_TIME_S", "0"))
    """段间复位倒计时；0 表示手动按回车继续。"""

    reset_home_path: Path = field(
        default_factory=lambda: Path(_env("RESET_HOME_PATH", "act/config/reset_home.json"))
    )
    """右臂固定复位姿态 JSON；环境变量 RESET_HOME_PATH。"""

    task_desc: str = field(
        default_factory=lambda: _env("TASK_DESC", "Pick up the red cube and place it into the box")
    )
    push_to_hub: bool = _env("PUSH_TO_HUB", "false").lower() == "true"

    # ===== 训练 =====
    job_name: str = field(default_factory=lambda: _env("JOB_NAME", "act_xlerobot_self_teleop"))
    train_device: str = field(default_factory=lambda: _env("DEVICE", "cuda"))
    batch_size: int = int(_env("BATCH_SIZE", "8"))
    steps: int = int(_env("STEPS", "100000"))
    save_freq: int = int(_env("SAVE_FREQ", "10000"))
    log_freq: int = int(_env("LOG_FREQ", "100"))
    wandb_enable: bool = _env("WANDB_ENABLE", "false").lower() == "true"

    # ===== 派生 =====
    @property
    def repo_id(self) -> str:
        if not self.hf_user:
            return f"local/{self.dataset_name}"
        return f"{self.hf_user}/{self.dataset_name}"

    @property
    def output_dir(self) -> Path:
        return Path("outputs/train") / self.job_name

    @property
    def dataset_path(self) -> Path:
        return self.dataset_root / self.repo_id

    @property
    def policy_path(self) -> Path:
        custom = os.environ.get("POLICY_PATH")
        if custom:
            return Path(custom)
        return self.output_dir / "checkpoints" / "last" / "pretrained_model"

    def banner(self, title: str) -> None:
        """打印关键配置，便于现场确认。"""
        print(f"\n{'=' * 60}")
        print(f"🎯 {title}")
        print(f"{'=' * 60}")
        print(f"  XLeRobot port1  : {self.follower_port1}  (左臂 + 头部)")
        print(f"  XLeRobot port2  : {self.follower_port2}  (右臂 + 两轮底盘)")
        print(f"  XLeRobot id     : {self.follower_id}")
        print(f"  相机 top        : {self.cam_top}")
        print(f"  相机 right_wrist: {self.cam_right_wrist}")
        print(f"  分辨率/帧率     : {self.cam_width}x{self.cam_height} @ {self.cam_fps}fps")
        print(f"{'=' * 60}\n")


CONFIG = Config()


if __name__ == "__main__":
    CONFIG.banner("当前配置 —— 自我遥操作 ACT 工作流")
    episode_desc = "手动结束" if CONFIG.episode_time_s <= 0 else f"最多 {CONFIG.episode_time_s}s"
    reset_desc = "右臂回 home 后手动继续" if CONFIG.reset_time_s <= 0 else f"右臂回 home 后等待 {CONFIG.reset_time_s}s"
    print(f"  数据集 repo_id  : {CONFIG.repo_id}")
    print(f"  数据集根目录    : {CONFIG.dataset_root}")
    print(f"  采集计划        : {CONFIG.num_episodes} 段，每段{episode_desc}，复位 {reset_desc}")
    print(f"  右臂 home 文件  : {CONFIG.reset_home_path}")
    print(f"  任务描述        : {CONFIG.task_desc}")
    print(f"  训练 steps      : {CONFIG.steps}, batch_size={CONFIG.batch_size}, device={CONFIG.train_device}")
    print(f"  输出目录        : {CONFIG.output_dir}")
    print()
    print("💡 角色：左臂关扭矩（人手推）→ 右臂跟随（被记录）→ 头部 / 底盘不控不录")
    print("💡 数据集 schema：state=6 (右臂6), action=6 (右臂6), 相机 top + right_wrist")
    print()
