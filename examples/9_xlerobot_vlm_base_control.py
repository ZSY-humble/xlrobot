#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM 大模型控制 xlerobot 底盘移动 — ManiSkill 仿真版

两种运行模式：

  模式1 - 交互模式（默认）：
    先弹出仿真窗口看环境，默认使用 ROS teleop 风格键盘控制。
    按住 w/a/s/d 移动，松开自动停止；按 t 可输入自然语言目标。

    python examples/9_xlerobot_vlm_base_control.py

  模式2 - 自动模式（--auto）：
    给定目标后 VLM 自动循环决策，无需手动干预。

    python examples/9_xlerobot_vlm_base_control.py --auto --goal "前进到红色盒子旁"

  可用指令（交互模式）：
    前进 | 后退 | 左转 | 右转 | 停止 | 前进左转 | 前进右转 | vlm | quit
    左转90度 / 右转180度 / 转一圈试试 / 目标：走到桌子旁

  运行示例：
    # 交互模式（能看到画面）
    export OPENAI_API_KEY="你的key"
    python examples/9_xlerobot_vlm_base_control.py \
        --env-id "ReplicaCAD_SceneManipulation-v1" \
        --render-mode "human" \
        --shader "rt-fast"

    # 自动模式（VLM 自己跑）
    python examples/9_xlerobot_vlm_base_control.py --auto --goal "向前移动避开障碍物"

    # 关代理
    ALL_PROXY="" all_proxy="" python examples/9_xlerobot_vlm_base_control.py
"""

import argparse
import base64
import io
import json
import os
import queue
import re
import select
import subprocess
import sys
import termios
import threading
import time
import tomllib
import tempfile
import tty
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import mani_skill  # noqa: F401
import numpy as np
import cv2
import torch
from PIL import Image


# =========================
# 默认配置
# =========================

CODEX_HOME = Path.home() / ".codex"
CODEX_CONFIG_PATH = CODEX_HOME / "config.toml"
CODEX_AUTH_PATH = CODEX_HOME / "auth.json"


def load_codex_model_defaults() -> Dict[str, str]:
    """从本机 Codex 配置读取模型、base_url 和 wire_api。"""
    if not CODEX_CONFIG_PATH.exists():
        return {}

    try:
        with CODEX_CONFIG_PATH.open("rb") as f:
            config = tomllib.load(f)
    except Exception:
        return {}

    model = str(config.get("model", "")).strip()
    provider = str(config.get("model_provider", "")).strip()
    providers = config.get("model_providers", {})
    provider_config = {}
    if isinstance(providers, dict) and provider:
        provider_config = providers.get(provider, {}) or {}

    defaults = {
        "model": model,
        "base_url": str(provider_config.get("base_url", "")).strip(),
        "wire_api": str(provider_config.get("wire_api", "")).strip(),
    }
    return {key: value for key, value in defaults.items() if value}


def load_codex_api_key() -> str:
    """从本机 Codex auth.json 读取 API key，不在日志中打印。"""
    if not CODEX_AUTH_PATH.exists():
        return ""

    try:
        with CODEX_AUTH_PATH.open("r", encoding="utf-8") as f:
            auth = json.load(f)
    except Exception:
        return ""

    return str(auth.get("OPENAI_API_KEY", "")).strip()


CODEX_DEFAULTS = load_codex_model_defaults()

DEFAULT_VLM_MODEL = os.environ.get(
    "VLM_MODEL",
    CODEX_DEFAULTS.get("model", "gpt-5.5"),
)
DEFAULT_OPENAI_BASE_URL = os.environ.get(
    "OPENAI_BASE_URL",
    CODEX_DEFAULTS.get("base_url", "https://api.cdn-krill-ai.com/codex/v1"),
)
DEFAULT_OPENAI_API_KEY = (
    os.environ.get("OPENAI_API_KEY", "").strip() or load_codex_api_key()
)
_CONFIG_WIRE_API = CODEX_DEFAULTS.get("wire_api", "responses")
DEFAULT_WIRE_API = os.environ.get(
    "OPENAI_WIRE_API",
    "codex-cli"
    if "api.cdn-krill-ai.com/codex" in DEFAULT_OPENAI_BASE_URL
    else _CONFIG_WIRE_API,
)

DEFAULT_LINEAR_SPEED = 0.4
DEFAULT_ANGULAR_SPEED = 0.25
DEFAULT_VLM_QUERY_INTERVAL = 3.0
DEFAULT_MAX_STEPS = 300
DEFAULT_ENV_ID = "ReplicaCAD_SceneManipulation-v1"
DEFAULT_SHADER = "rt-fast"
DEFAULT_INTERACTIVE_RATE = 30.0
DEFAULT_OBSTACLE_DISTANCE = 0.35
DEFAULT_EMERGENCY_DISTANCE = 0.20
DEFAULT_SAFETY_LOG_INTERVAL = 1.0
DEFAULT_TELEOP_HOLD_TIMEOUT = 0.60


# =========================
# 指令集合
# =========================

COMMANDS: List[str] = [
    "前进",
    "后退",
    "左转",
    "右转",
    "停止",
    "前进左转",
    "前进右转",
    "后退左转",
    "后退右转",
]

# 交互模式下的快捷键
SHORTCUT_MAP: Dict[str, str] = {
    "w": "前进",
    "s": "后退",
    "a": "左转",
    "d": "右转",
    "q": "前进左转",
    "e": "前进右转",
    "z": "后退左转",
    "c": "后退右转",
    "x": "停止",
    "v": "vlm",       # 让 VLM 看一眼并决策
    "p": "vlm",       # 同上
    "": "停止",        # 直接回车 = 停止
}

STOP_LONG_GOAL_INPUTS = {
    "clear",
    "cancel",
    "manual",
    "手动",
    "取消",
    "清除",
    "取消目标",
    "清除目标",
}

# 带这些前缀的自然语言，会进入“长期目标 / 持续 VLM 决策”模式。
# 这样可以避免“转一圈试试”这类一次性测试语句被误当成长期导航目标。
STRIP_LONG_GOAL_PREFIXES = (
    "目标:",
    "目标：",
    "长期目标:",
    "长期目标：",
    "goal:",
    "auto:",
)

KEEP_LONG_GOAL_PREFIXES = (
    "导航到",
    "去到",
    "走到",
    "移动到",
)

CONTAINS_LONG_GOAL_KEYWORDS = (
    "导航到",
    "去到",
    "走到",
    "移动到",
    "过去",
    "靠近",
)

EN_COMMAND_MAP: Dict[str, str] = {
    "forward left": "前进左转",
    "forward right": "前进右转",
    "backward left": "后退左转",
    "backward right": "后退右转",
    "turn left": "左转",
    "turn right": "右转",
    "go forward": "前进",
    "go backward": "后退",
    "go left": "左转",
    "go right": "右转",
    "move forward": "前进",
    "move backward": "后退",
    "forward": "前进",
    "backward": "后退",
    "left": "左转",
    "right": "右转",
    "stop": "停止",
    "stay": "停止",
    "wait": "停止",
    "idle": "停止",
}


def make_command_to_action(
    linear_speed: float,
    angular_speed: float,
) -> Dict[str, list]:
    """构造 指令 → 底盘动作 的映射。"""
    # ManiSkill / XLeRobot 约定：base action = [forward_vel, yaw_vel]。
    # yaw_vel 正值为左转，负值为右转，和真机 teleop 的 theta.vel 保持一致。
    return {
        "前进": [linear_speed, 0.0],
        "后退": [-linear_speed, 0.0],
        "左转": [0.0, angular_speed],
        "右转": [0.0, -angular_speed],
        "停止": [0.0, 0.0],
        "前进左转": [linear_speed, angular_speed],
        "前进右转": [linear_speed, -angular_speed],
        "后退左转": [-linear_speed, angular_speed],
        "后退右转": [-linear_speed, -angular_speed],
    }


# =========================
# 工具函数
# =========================

def to_numpy(x: Any) -> np.ndarray:
    """torch.Tensor / numpy / list → numpy.ndarray"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def normalize_rgb_array(image_array: Any) -> np.ndarray:
    """将 ManiSkill 相机 RGB 转成 HWC uint8 RGB 图像。"""
    img = to_numpy(image_array)

    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]

    if img.ndim == 3 and img.shape[0] in [1, 3, 4] and img.shape[-1] not in [1, 3, 4]:
        img = np.transpose(img, (1, 2, 0))

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]

    if img.ndim == 3 and img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    img = np.nan_to_num(img)

    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def image_to_base64(image_array: Any, max_size: int = 512) -> str:
    """将图像转成 base64 JPEG。"""
    img = normalize_rgb_array(image_array)
    pil_img = Image.fromarray(img)

    if max_size > 0:
        pil_img.thumbnail((max_size, max_size))

    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def save_rgb_debug_image(image_array: Any, path: Path, max_size: int = 512) -> None:
    """保存 VLM 请求使用的 RGB 图像，便于复盘模型看到的画面。"""
    img = normalize_rgb_array(image_array)
    pil_img = Image.fromarray(img)

    if max_size > 0:
        pil_img.thumbnail((max_size, max_size))

    path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(path, format="JPEG", quality=85)


def normalize_depth_array(depth_array: Any) -> Optional[np.ndarray]:
    """将 ManiSkill 深度图转成 HW float 数组。"""
    if depth_array is None:
        return None

    depth = to_numpy(depth_array)

    if depth.ndim == 4 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim == 3 and depth.shape[0] == 1 and depth.shape[-1] != 1:
        depth = depth[0]
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    depth = np.asarray(depth, dtype=np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    return depth


def get_camera_rgb(obs: Dict[str, Any], camera_name: str = "base_camera") -> Any:
    """从 obs 中取 RGB 图像。"""
    return obs["sensor_data"][camera_name]["rgb"]


def get_camera_depth(
    obs: Dict[str, Any], camera_name: str = "base_camera"
) -> Optional[Any]:
    """从 obs 中取 depth 图像。"""
    try:
        return obs["sensor_data"][camera_name].get("depth", None)
    except Exception:
        return None


def get_sensor_camera_names(obs: Dict[str, Any]) -> List[str]:
    """返回当前观测里可用的相机名。"""
    sensor_data = obs.get("sensor_data", {})
    if isinstance(sensor_data, dict):
        return list(sensor_data.keys())
    return []


def resolve_camera_name(obs: Dict[str, Any], requested: str) -> str:
    """根据 obs 自动选择 VLM 相机，避免默认拿到只看地面的相机。"""
    if requested and requested != "auto":
        return requested

    names = get_sensor_camera_names(obs)
    preferred = [
        "fetch_head",
        "head_camera",
        "base_camera",
        "fetch_right_arm_camera",
        "fetch_left_arm_camera",
    ]
    for name in preferred:
        if name in names:
            return name

    if names:
        return names[0]

    return requested or "auto"


# =========================
# 深度图安全检测
# =========================

def estimate_front_distance(
    depth_array: Any,
    center_ratio: float = 0.45,
) -> Optional[float]:
    """估计图像中心区域的前方距离。"""
    depth = normalize_depth_array(depth_array)
    if depth is None or depth.ndim != 2:
        return None

    h, w = depth.shape
    crop_h = int(h * center_ratio)
    crop_w = int(w * center_ratio)

    y1 = max(0, h // 2 - crop_h // 2)
    y2 = min(h, h // 2 + crop_h // 2)
    x1 = max(0, w // 2 - crop_w // 2)
    x2 = min(w, w // 2 + crop_w // 2)

    center = depth[y1:y2, x1:x2]
    valid = center[np.isfinite(center) & (center > 0)]

    if valid.size < 20:
        return None

    median_depth = float(np.median(valid))

    # 深度图单位兼容：
    #   ManiSkill 默认输出米制深度，室内场景通常 0.1~5.0m
    #   部分深度图（如 RealSense raw）输出毫米，中位数会 > 100
    #   阈值 100 避免误判：米制下室内距离极少超过 100m
    if median_depth > 100.0:
        median_depth = median_depth / 1000.0

    return median_depth


def apply_depth_safety_filter(
    base_action: list,
    front_distance: Optional[float],
    obstacle_distance: float,
    emergency_distance: float,
) -> Tuple[list, str]:
    """根据前方距离修正底盘动作。"""
    action = [float(base_action[0]), float(base_action[1])]

    if front_distance is None:
        return action, "无深度修正"

    if front_distance < emergency_distance:
        return [0.0, 0.0], f"紧急停止：前方 {front_distance:.2f}m"

    if front_distance < obstacle_distance and action[0] > 0:
        if abs(action[1]) > 1e-6:
            action[0] = 0.0
            return action, f"安全修正：前方 {front_distance:.2f}m，仅保留转向"
        else:
            return [0.0, 0.0], f"安全修正：前方 {front_distance:.2f}m，禁止前进"

    return action, f"前方 {front_distance:.2f}m，动作允许"


# =========================
# VLM 控制器
# =========================

class VLMNavigationController:
    """VLM 导航控制器。每次调用时把 goal、last_command 等状态放进 prompt。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        wire_api: str,
        temperature: float = 0.1,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY 为空，请先设置环境变量。")

        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        self.model = model
        self.wire_api = wire_api
        self.temperature = temperature

    @staticmethod
    def build_system_prompt() -> str:
        allowed = " | ".join(COMMANDS)
        return f"""你是一个机器人第一人称视觉导航控制器。
你会收到机器人当前相机图像、任务目标和简短状态记忆。

你的任务：
根据当前图像和任务目标，选择机器人下一步底盘动作。

只能输出以下指令之一：
{allowed}

控制规则：
1. 前方通畅，且目标在前方或需要继续接近目标 → 前进
2. 目标偏左，或需要向左调整方向 → 左转 / 前进左转
3. 目标偏右，或需要向右调整方向 → 右转 / 前进右转
4. 前方有障碍物 → 优先左转或右转，不要直接前进
5. 已经靠近目标或任务完成 → 停止
6. 图像不清楚、无法判断、存在危险 → 停止

输出要求：
只输出一个指令词。不要解释。不要输出标点。不要输出 JSON。
"""

    @staticmethod
    def build_task_prompt(
        goal: str,
        last_command: str,
        elapsed_time: float,
        front_distance: Optional[float],
        step: int,
    ) -> str:
        if front_distance is None:
            depth_text = "未获得有效前方深度"
        else:
            depth_text = f"前方中心区域估计距离约 {front_distance:.2f} 米"

        allowed = " | ".join(COMMANDS)

        return f"""任务目标：
{goal}

程序记忆：
- 当前仿真步数：{step}
- 距离任务开始：{elapsed_time:.1f} 秒
- 上一次动作：{last_command}
- 深度安全检测：{depth_text}

请根据当前图像和以上任务目标，选择下一步动作。
只能从以下指令中选择一个：
{allowed}

只输出一个指令词。
"""

    def query_responses(self, image_b64: str, task_prompt: str) -> str:
        """使用 Responses API 调用 VLM。"""
        system_prompt = self.build_system_prompt()

        request_kwargs = dict(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": system_prompt},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": task_prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{image_b64}",
                        },
                    ],
                },
            ],
            max_output_tokens=20,
        )

        try:
            response = self.client.responses.create(
                **request_kwargs,
                temperature=self.temperature,
            )
        except Exception as e:
            if "temperature" in str(e).lower():
                response = self.client.responses.create(**request_kwargs)
            else:
                raise

        content = getattr(response, "output_text", None)
        if content:
            return str(content).strip()

        output = getattr(response, "output", None) or []
        chunks: List[str] = []
        for item in output:
            for part in getattr(item, "content", []) or []:
                text = getattr(part, "text", None)
                if text:
                    chunks.append(str(text))

        if chunks:
            return "".join(chunks).strip()

        return "停止"

    def query_chat_completions(self, image_b64: str, task_prompt: str) -> str:
        """使用 Chat Completions API 调用 VLM。"""
        system_prompt = self.build_system_prompt()

        request_kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": task_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                            },
                        },
                    ],
                },
            ],
            max_tokens=20,
        )

        try:
            response = self.client.chat.completions.create(
                **request_kwargs,
                temperature=self.temperature,
            )
        except Exception as e:
            if "temperature" in str(e).lower():
                response = self.client.chat.completions.create(**request_kwargs)
            else:
                raise

        content = response.choices[0].message.content
        if content is None:
            return "停止"

        return content.strip()

    def query_codex_cli(self, image_b64: str, task_prompt: str) -> str:
        """使用本机 Codex CLI 调用 VLM，复用 Codex 自身鉴权链路。"""
        system_prompt = self.build_system_prompt()
        prompt = (
            f"{system_prompt}\n\n"
            f"{task_prompt}\n\n"
            "再次强调：最终只输出一个指令词，不要解释。"
        )

        image_bytes = base64.b64decode(image_b64)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as image_file:
            image_file.write(image_bytes)
            image_file.flush()

            result = subprocess.run(
                [
                    "codex",
                    "exec",
                    "-i",
                    image_file.name,
                    "-",
                ],
                input=prompt,
                text=True,
                capture_output=True,
                timeout=90,
                check=False,
            )

        output = "\n".join(part for part in [result.stdout, result.stderr] if part)

        if result.returncode != 0:
            raise RuntimeError(output.strip() or f"codex exec 退出码 {result.returncode}")

        response_text = self.extract_codex_cli_response(output)
        return response_text or "停止"

    @staticmethod
    def extract_codex_cli_response(output: str) -> str:
        """从 codex exec 输出中提取模型回答段，避开回显的 prompt。"""
        lines = [line.strip() for line in output.splitlines()]

        for i, line in enumerate(lines):
            if line == "codex":
                collected: List[str] = []
                for next_line in lines[i + 1:]:
                    if not next_line:
                        continue
                    if next_line == "tokens used":
                        break
                    if next_line.startswith("202") and " WARN " in next_line:
                        continue
                    if next_line.startswith("OpenAI Codex"):
                        continue
                    if next_line.startswith("--------"):
                        continue
                    collected.append(next_line)

                if collected:
                    return "\n".join(collected).strip()

        for line in reversed(lines):
            if not line:
                continue
            if line == "tokens used" or line.startswith("202"):
                continue
            parsed = parse_command(line)
            if parsed in COMMANDS:
                return parsed

        return ""

    def query(self, image_b64: str, task_prompt: str) -> str:
        """调用 VLM，返回原始文本。"""
        if self.wire_api == "codex-cli":
            return self.query_codex_cli(image_b64, task_prompt)
        if self.wire_api == "chat":
            return self.query_chat_completions(image_b64, task_prompt)
        return self.query_responses(image_b64, task_prompt)


# =========================
# 异步 VLM 调用
# =========================

@dataclass
class VLMAsyncRequest:
    """主线程提交给 VLM 后台线程的只读请求快照。"""

    request_id: int
    generation: int
    goal: str
    image_b64: str
    task_prompt: str
    is_long_goal: bool
    created_step: int
    created_time: float
    last_command: str
    front_distance: Optional[float]
    mode: str
    debug_image_path: Optional[str] = None
    debug_json_path: Optional[str] = None


@dataclass
class VLMAsyncResult:
    """VLM 后台线程返回给主线程的结果。"""

    request_id: int
    generation: int
    goal: str
    is_long_goal: bool
    created_step: int = 0
    created_time: float = 0.0
    finished_time: float = 0.0
    elapsed_s: float = 0.0
    front_distance: Optional[float] = None
    mode: str = ""
    debug_json_path: Optional[str] = None
    raw_response: str = ""
    error: Optional[Exception] = None


class AsyncVLMWorker:
    """单 worker 异步 VLM 调用器，避免 VLM 请求阻塞仿真主循环。"""

    def __init__(self, vlm: VLMNavigationController) -> None:
        self.vlm = vlm
        self.request_queue: "queue.Queue[Optional[VLMAsyncRequest]]" = queue.Queue(
            maxsize=1,
        )
        self.result_queue: "queue.Queue[VLMAsyncResult]" = queue.Queue()
        self.thread = threading.Thread(
            target=self._run,
            name="async-vlm-worker",
            daemon=True,
        )
        self.thread.start()

    def submit(self, request: VLMAsyncRequest) -> bool:
        """提交一个 VLM 请求；队列满时返回 False，避免堆积过期任务。"""
        try:
            self.request_queue.put_nowait(request)
            return True
        except queue.Full:
            return False

    def get_results(self) -> List[VLMAsyncResult]:
        """取出所有已完成结果。"""
        results: List[VLMAsyncResult] = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get_nowait())
        return results

    def clear_pending(self) -> None:
        """清掉尚未开始执行的旧请求；正在运行的请求无法强制中断。"""
        while True:
            try:
                request = self.request_queue.get_nowait()
            except queue.Empty:
                return

            if request is None:
                try:
                    self.request_queue.put_nowait(None)
                except queue.Full:
                    pass
                return

    def stop(self) -> None:
        """通知 worker 退出；正在进行的 VLM 请求会自然结束。"""
        self.clear_pending()
        try:
            self.request_queue.put_nowait(None)
        except queue.Full:
            pass

    def _run(self) -> None:
        while True:
            request = self.request_queue.get()
            if request is None:
                return

            try:
                start_time = time.time()
                print(
                    f"   🚦 VLM#{request.request_id} 后台开始: "
                    f"mode={request.mode}, step={request.created_step}"
                )
                raw_response = self.vlm.query(
                    request.image_b64,
                    request.task_prompt,
                )
                finished_time = time.time()
                print(
                    f"   📨 VLM#{request.request_id} 后台返回: "
                    f"{finished_time - start_time:.2f}s, raw={raw_response!r}"
                )
                result = VLMAsyncResult(
                    request_id=request.request_id,
                    generation=request.generation,
                    goal=request.goal,
                    is_long_goal=request.is_long_goal,
                    created_step=request.created_step,
                    created_time=request.created_time,
                    finished_time=finished_time,
                    elapsed_s=finished_time - start_time,
                    front_distance=request.front_distance,
                    mode=request.mode,
                    debug_json_path=request.debug_json_path,
                    raw_response=raw_response,
                )
            except Exception as e:
                finished_time = time.time()
                result = VLMAsyncResult(
                    request_id=request.request_id,
                    generation=request.generation,
                    goal=request.goal,
                    is_long_goal=request.is_long_goal,
                    created_step=request.created_step,
                    created_time=request.created_time,
                    finished_time=finished_time,
                    elapsed_s=finished_time - request.created_time,
                    front_distance=request.front_distance,
                    mode=request.mode,
                    debug_json_path=request.debug_json_path,
                    error=e,
                )

            self.result_queue.put(result)


# =========================
# 指令解析
# =========================

def parse_command(raw: str) -> str:
    """将 VLM 原始输出解析为标准中文指令（长词优先匹配）。"""
    if raw is None:
        return "停止"

    text = str(raw).lower().strip()

    for ch in ["。", ".", ",", "，", "!", "！", ":", "：", ";", "；", "\"", "'", "`"]:
        text = text.replace(ch, "")

    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())

    for en_cmd in sorted(EN_COMMAND_MAP.keys(), key=len, reverse=True):
        if en_cmd in text:
            return EN_COMMAND_MAP[en_cmd]

    for cmd in sorted(COMMANDS, key=len, reverse=True):
        if cmd in text:
            return cmd

    return "停止"


def parse_macro_input(text: str) -> Optional[str]:
    """把低层自然语言测试指令解析成一次性 macro，不进入长期 VLM 目标。"""
    t = text.strip().lower()
    compact = "".join(t.split())

    # 方向判断：默认右转；显式出现“左/逆时针/counterclockwise”才左转。
    direction = "左转" if any(k in compact for k in ["左", "逆时针", "counterclockwise", "ccw"]) else "右转"

    # 一圈 / 360 度
    if any(k in compact for k in [
        "转一圈",
        "转1圈",
        "旋转一圈",
        "原地转一圈",
        "转弯一圈",
        "转360",
        "旋转360",
        "turnaround",
        "spinaround",
    ]):
        return f"macro_turn:{direction}:circle"

    # 两圈 / 720 度
    if any(k in compact for k in [
        "转两圈",
        "转2圈",
        "旋转两圈",
        "旋转2圈",
        "原地转两圈",
        "原地转2圈",
        "转720",
        "旋转720",
    ]):
        return f"macro_turn:{direction}:double"

    # 半圈 / 180 度
    if any(k in compact for k in [
        "转半圈",
        "旋转半圈",
        "原地转半圈",
        "转180",
        "旋转180",
    ]):
        return f"macro_turn:{direction}:half"

    # 只想“转一下/试试转弯”，给一个较短测试动作，避免进入长期 VLM。
    if any(k in compact for k in [
        "转一下",
        "转弯试试",
        "试试转弯",
        "旋转一下",
    ]):
        return f"macro_turn:{direction}:short"

    return None


def parse_timed_manual_input(text: str) -> Optional[str]:
    """解析 'a 1.2' / '左转2秒' 这类动作+持续时间输入。"""
    raw = text.strip()
    lower_text = raw.lower()

    shortcut_match = re.fullmatch(
        r"([wasdqezc])\s*([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|秒)?",
        lower_text,
    )
    if shortcut_match:
        shortcut, seconds_text = shortcut_match.groups()
        command = SHORTCUT_MAP.get(shortcut)
        if command and command != "vlm":
            return f"timed:{command}:{float(seconds_text):.3f}"

    compact = "".join(raw.split())
    for command in sorted(COMMANDS, key=len, reverse=True):
        match = re.fullmatch(
            rf"{re.escape(command)}([0-9]+(?:\.[0-9]+)?)(?:s|sec|秒)?",
            compact,
            flags=re.IGNORECASE,
        )
        if match:
            return f"timed:{command}:{float(match.group(1)):.3f}"

    spaced_match = re.fullmatch(
        r"(.+?)\s+([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|秒)?",
        raw,
        flags=re.IGNORECASE,
    )
    if spaced_match:
        command_text, seconds_text = spaced_match.groups()
        command = parse_command(command_text)
        if command in COMMANDS:
            return f"timed:{command}:{float(seconds_text):.3f}"

    return None


def parse_turn_angle_input(text: str) -> Optional[str]:
    """解析 '左转90度' / '右转 180deg' 这类角度控制输入。"""
    raw = text.strip()
    compact = "".join(raw.split()).lower()

    match = re.fullmatch(
        r"(左转|右转)([0-9]+(?:\.[0-9]+)?)(?:度|°|deg|degree|degrees)",
        compact,
        flags=re.IGNORECASE,
    )
    if match:
        command, degrees_text = match.groups()
        return f"turn_angle:{command}:{float(degrees_text):.3f}"

    match = re.fullmatch(
        r"(turnleft|left|turnright|right)([0-9]+(?:\.[0-9]+)?)(?:deg|degree|degrees|°)",
        compact,
        flags=re.IGNORECASE,
    )
    if match:
        direction_text, degrees_text = match.groups()
        command = "左转" if "left" in direction_text else "右转"
        return f"turn_angle:{command}:{float(degrees_text):.3f}"

    spaced_match = re.fullmatch(
        r"(左转|右转|turn left|left|turn right|right)\s+"
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:度|°|deg|degree|degrees)",
        raw,
        flags=re.IGNORECASE,
    )
    if spaced_match:
        direction_text, degrees_text = spaced_match.groups()
        command = "左转" if "左" in direction_text or "left" in direction_text.lower() else "右转"
        return f"turn_angle:{command}:{float(degrees_text):.3f}"

    return None


def parse_user_input(user_input: str) -> Optional[str]:
    """
    解析交互模式下用户的终端输入。

    返回：
      "manual:<动作>"：手动持续速度，直到输入停止/新动作
      "timed:<动作>:<秒数>"：动作持续指定时间
      "turn_angle:<左转/右转>:<角度>"：按角速度换算持续时间
      "vlm"：让 VLM 按默认目标决策一次
      "vlm_once:<自然语言>"：让 VLM 按用户语言只决策一次，不进入长期循环
      "vlm_text:<长期目标>"：长期目标，持续低频 VLM 决策
      "macro_turn:<左转/右转>:<circle/half/short>"：一次性转向宏动作
      None：退出
    """
    text = user_input.strip()
    lower_text = text.lower()

    # 退出
    if lower_text in ("quit", "exit", "q!", "退出"):
        return None

    if lower_text in STOP_LONG_GOAL_INPUTS:
        return "clear_goal"

    timed = parse_timed_manual_input(text)
    if timed:
        return timed

    # 明确带“度/deg/°”的输入优先走角速度换算，不被 macro 吃掉。
    turn_angle = parse_turn_angle_input(text)
    if turn_angle:
        return turn_angle

    # macro 优先，避免“转一圈试试”被后面的“右转/左转”或 VLM 长期目标吃掉。
    macro = parse_macro_input(text)
    if macro:
        return macro

    # 快捷键
    if lower_text in SHORTCUT_MAP:
        command = SHORTCUT_MAP[lower_text]
        if command == "vlm":
            return command
        return f"manual:{command}"

    # 直接输入指令
    for cmd in sorted(COMMANDS, key=len, reverse=True):
        if cmd in text:
            return f"manual:{cmd}"

    # 显式 VLM 关键词：只决策一次
    if lower_text in ("vlm", "v", "ai", "大模型"):
        return "vlm"

    # 显式 auto 关键词：按默认目标进入长期 VLM。
    if lower_text in ("auto", "自动", "自动导航"):
        return f"vlm_text:"

    # 目标类语句进入长期目标模式；普通描述句只做一次 VLM 决策。
    for prefix in STRIP_LONG_GOAL_PREFIXES:
        if text.startswith(prefix):
            goal = text[len(prefix):].strip()
            if goal:
                return f"vlm_text:{goal}"
            return f"vlm_text:"

    for prefix in KEEP_LONG_GOAL_PREFIXES:
        if text.startswith(prefix):
            return f"vlm_text:{text}"

    # 允许“你移动到单车那里”这类口语句，不要求目标动词在句首。
    if any(keyword in text for keyword in CONTAINS_LONG_GOAL_KEYWORDS):
        return f"vlm_text:{text}"

    # 普通自然语言默认只让 VLM 看图决策一次，不残留 long_goal。
    if text:
        return f"vlm_once:{text}"

    return "停止"


def macro_turn_degrees(kind: str) -> float:
    """把转向 macro 类型换算成目标角度。"""
    if kind == "double":
        return 720.0
    if kind == "circle":
        return 360.0
    if kind == "half":
        return 180.0
    return 45.0


def format_vlm_error(error: Exception, args: argparse.Namespace) -> str:
    """把 OpenAI 兼容接口错误转成更容易排查的中文提示。"""
    text = str(error)
    lower_text = text.lower()

    if "401" in text or "unauthorized" in lower_text or "invalid token" in lower_text:
        return (
            "鉴权失败：OPENAI_API_KEY 无效、过期，或和 OPENAI_BASE_URL 不匹配。"
            f"当前 API 地址: {args.base_url}，当前模型: {args.model}，"
            f"接口协议: {args.wire_api}。"
            "请重新 export OPENAI_API_KEY，必要时同步检查 OPENAI_BASE_URL / --model。"
        )

    if "404" in text or "not found" in lower_text:
        return (
            f"模型或接口不存在：当前 API 地址 {args.base_url}，"
            f"当前模型 {args.model}，接口协议 {args.wire_api}。"
            "请检查 --model / --wire-api 是否被服务支持。"
        )

    return text


def format_front_distance(front_distance: Optional[float]) -> str:
    """格式化前方距离用于终端 trace。"""
    if front_distance is None:
        return "None"
    return f"{front_distance:.2f}m"


def write_vlm_debug_json(
    path: Optional[str],
    payload: Dict[str, Any],
) -> None:
    """把 VLM 调试信息写入 JSON。"""
    if not path:
        return

    debug_path = Path(path)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    with debug_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# =========================
# 动作构建
# =========================

def resolve_base_action_slice(
    env: gym.Env,
    action_space: Any,
    override_start: Optional[int] = None,
) -> Tuple[int, int, str]:
    """解析底盘 [forward, yaw] 在扁平 action 向量里的位置。"""
    if not hasattr(action_space, "shape") or action_space.shape is None:
        raise ValueError(f"不支持的 action_space: {action_space}")

    action_dim = int(np.prod(action_space.shape))
    if action_dim < 2:
        raise ValueError(f"动作维度过小: shape={action_space.shape}")

    if override_start is not None:
        start = int(override_start)
        end = start + 2
        if start < 0 or end > action_dim:
            raise ValueError(
                f"--base-action-start={start} 越界，action_dim={action_dim}"
            )
        return start, end, "命令行参数"

    unwrapped = getattr(env, "unwrapped", env)
    agent = getattr(unwrapped, "agent", None)
    controller = getattr(agent, "controller", None)
    action_mapping = getattr(controller, "action_mapping", None)

    if isinstance(action_mapping, dict) and "base" in action_mapping:
        start, end = action_mapping["base"]
        start = int(start)
        end = int(end)
        if end - start >= 2 and 0 <= start < end <= action_dim:
            return start, start + 2, "controller.action_mapping['base']"

    return action_dim - 2, action_dim, "回退到最后两维"


def build_full_action(
    base_action: list,
    action_space: Any,
    base_action_slice: Tuple[int, int],
) -> np.ndarray:
    """构建完整动作向量，把底盘 [forward, yaw] 写入指定切片。"""
    if not hasattr(action_space, "shape") or action_space.shape is None:
        raise ValueError(f"不支持的 action_space: {action_space}")

    action = np.zeros(action_space.shape, dtype=np.float32)
    flat = action.reshape(-1)

    if flat.size < 2:
        raise ValueError(f"动作维度过小: shape={action_space.shape}")

    start, end = base_action_slice
    if end - start < 2 or start < 0 or end > flat.size:
        raise ValueError(
            f"底盘动作索引非法: slice=({start}, {end}), action_dim={flat.size}"
        )

    flat[start] = float(base_action[0])
    flat[start + 1] = float(base_action[1])

    return action


def is_done(terminated: Any, truncated: Any) -> bool:
    """兼容 bool / numpy / torch 的 done 判断。"""
    term = bool(np.asarray(to_numpy(terminated)).any())
    trunc = bool(np.asarray(to_numpy(truncated)).any())
    return term or trunc


# =========================
# 渲染保存
# =========================

def save_render_frame(env: Any, step_count: int, save_dir: Path) -> None:
    """保存渲染图，用于调试。"""
    rendered = env.render()
    if rendered is None:
        return

    if isinstance(rendered, torch.Tensor):
        rendered = rendered.detach().cpu().numpy()

    if not isinstance(rendered, np.ndarray):
        return

    img = normalize_rgb_array(rendered)
    save_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(save_dir / f"step_{step_count:04d}.png")


def save_camera_frame(
    obs: Dict[str, Any],
    step_count: int,
    camera_name: str,
    save_dir: Path,
) -> None:
    """保存相机第一人称图像。"""
    rgb = get_camera_rgb(obs, camera_name=camera_name)
    if rgb is None:
        return

    img = normalize_rgb_array(rgb)
    save_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(save_dir / f"camera_{step_count:04d}.png")


def render_human_if_needed(env: gym.Env, args: argparse.Namespace) -> None:
    """human 模式下刷新 ManiSkill Viewer。"""
    if args.render_mode == "human":
        env.render()


# =========================
# 相机实时展示
# =========================

CAMERA_WINDOW_NAME = "XLeRobot Cameras"


def build_camera_display_frame(
    obs: Dict[str, Any],
    display_height: int = 180,
    status_texts: Optional[Dict[str, str]] = None,
) -> Optional[np.ndarray]:
    """从 obs 构建所有相机画面的水平拼接显示帧。

    参数：
        obs: 环境观测字典，包含 sensor_data。
        display_height: 每个相机面板的统一高度（像素）。
        status_texts: 可选的状态文本，键为标签，值为要叠加的文本字符串。
            例如 {"指令": "前进", "前方距离": "0.45m", "目标": "走到桌子旁"}

    返回：
        BGR uint8 numpy 数组，可直接传给 cv2.imshow；
        如果没有相机数据则返回 None。
    """
    sensor_data = obs.get("sensor_data", {})
    if not isinstance(sensor_data, dict) or not sensor_data:
        return None

    camera_names = list(sensor_data.keys())
    panels: List[np.ndarray] = []

    for cam_name in camera_names:
        cam_data = sensor_data[cam_name]
        rgb_raw = cam_data.get("rgb")
        if rgb_raw is None:
            continue

        # RGB → BGR（OpenCV 使用 BGR 通道顺序）
        rgb = normalize_rgb_array(rgb_raw)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # 缩放到统一高度
        h, w = bgr.shape[:2]
        if h != display_height:
            scale = display_height / h
            new_w = max(int(w * scale), 1)
            bgr = cv2.resize(bgr, (new_w, display_height), interpolation=cv2.INTER_AREA)
        else:
            new_w = w

        # 在画面顶部添加相机名称标签条
        label_bar = np.zeros((28, new_w, 3), dtype=np.uint8)
        cv2.putText(
            label_bar,
            cam_name,
            (4, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        panel = np.vstack([label_bar, bgr])
        panels.append(panel)

    if not panels:
        return None

    # 水平拼接所有面板
    display = np.hstack(panels)

    # 在拼接图底部叠加状态文字条
    if status_texts:
        bar_h = 26 + 22 * len(status_texts)
        status_bar = np.zeros((bar_h, display.shape[1], 3), dtype=np.uint8)
        y_offset = 20
        for label, value in status_texts.items():
            text = f"{label}: {value}"
            cv2.putText(
                status_bar,
                text,
                (8, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            y_offset += 22
        display = np.vstack([display, status_bar])

    return display


def update_camera_window(
    obs: Dict[str, Any],
    args: argparse.Namespace,
    status_texts: Optional[Dict[str, str]] = None,
) -> None:
    """刷新相机展示窗口（仅在 --show-cameras 时调用）。"""
    try:
        frame = build_camera_display_frame(
            obs,
            display_height=args.camera_display_height,
            status_texts=status_texts,
        )
        if frame is None:
            return
        cv2.imshow(CAMERA_WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        # 按 Esc 关闭相机窗口（不影响仿真）
        if key == 27:
            cv2.destroyWindow(CAMERA_WINDOW_NAME)
    except cv2.error:
        pass  # headless 环境，静默跳过


def close_camera_window() -> None:
    """安全关闭相机展示窗口（幂等）。"""
    try:
        cv2.destroyWindow(CAMERA_WINDOW_NAME)
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


# =========================
# 参数
# =========================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VLM 低频导航控制 for xlerobot in ManiSkill"
    )

    # 模式
    parser.add_argument(
        "--auto",
        action="store_true",
        help="自动模式：VLM 自动循环决策。不加此参数则为交互模式。",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="请控制机器人向前移动，遇到障碍物时避开，并在接近目标时停止。",
        help="任务目标（自动模式下生效）。",
    )

    # 环境
    parser.add_argument("--env-id", type=str, default=DEFAULT_ENV_ID)
    parser.add_argument("--robot-uid", type=str, default="xlerobot")
    parser.add_argument(
        "--camera-name",
        type=str,
        default="auto",
        help="VLM 使用的相机名。默认 auto，优先选择 fetch_head/head_camera。",
    )
    parser.add_argument("--obs-mode", type=str, default="rgbd")
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        help="渲染模式：human（弹窗看画面）/ rgb_array（无画面）。",
    )
    parser.add_argument(
        "--shader",
        type=str,
        default=DEFAULT_SHADER,
        help="ManiSkill 渲染 shader：default / rt / rt-fast 等。",
    )
    parser.add_argument("--sim-backend", type=str, default="auto")
    parser.add_argument("--render-backend", type=str, default="gpu")
    parser.add_argument(
        "--control-mode",
        type=str,
        default="pd_joint_delta_pos",
        help="ManiSkill 控制模式。",
    )

    # VLM
    parser.add_argument("--model", type=str, default=DEFAULT_VLM_MODEL)
    parser.add_argument("--base-url", type=str, default=DEFAULT_OPENAI_BASE_URL)
    parser.add_argument(
        "--wire-api",
        type=str,
        choices=["codex-cli", "responses", "chat"],
        default=DEFAULT_WIRE_API,
        help="VLM 调用协议。Krill/Codex 默认使用 codex-cli 复用本机 Codex 鉴权。",
    )

    # 速度
    parser.add_argument("--linear-speed", type=float, default=DEFAULT_LINEAR_SPEED)
    parser.add_argument("--angular-speed", type=float, default=DEFAULT_ANGULAR_SPEED)
    parser.add_argument("--vlm-interval", type=float, default=DEFAULT_VLM_QUERY_INTERVAL)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--base-action-start",
        type=int,
        default=None,
        help=(
            "底盘动作在扁平 action 向量中的起始索引。默认自动读取 "
            "ManiSkill controller.action_mapping['base']，失败时回退到最后两维。"
        ),
    )

    # 深度安全
    parser.add_argument("--obstacle-distance", type=float, default=DEFAULT_OBSTACLE_DISTANCE)
    parser.add_argument("--emergency-distance", type=float, default=DEFAULT_EMERGENCY_DISTANCE)
    parser.add_argument("--disable-depth-safety", action="store_true")
    parser.add_argument(
        "--safety-log-interval",
        type=float,
        default=DEFAULT_SAFETY_LOG_INTERVAL,
        help="深度安全日志最小打印间隔，单位秒。",
    )

    # 图片
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--save-vlm-debug",
        action="store_true",
        help="保存每次 VLM 请求的图像、prompt 和结果 JSON。",
    )
    parser.add_argument(
        "--vlm-debug-dir",
        type=str,
        default="outputs/vlm_control/vlm_debug",
        help="VLM 调试文件保存目录。",
    )

    # 渲染保存
    parser.add_argument("--save-render", action="store_true")
    parser.add_argument("--render-every", type=int, default=10)
    parser.add_argument("--render-dir", type=str, default="outputs/vlm_control")

    # 相机画面实时展示
    parser.add_argument(
        "--show-cameras",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "在 OpenCV 窗口中实时展示所有相机的 RGB 画面（默认开启）。"
            "加 --no-show-cameras 关闭。"
        ),
    )
    parser.add_argument(
        "--camera-display-height",
        type=int,
        default=180,
        help="相机预览窗口中每幅画面的统一高度（像素），宽度按原始比例缩放。",
    )

    # 安全
    parser.add_argument("--stop-on-vlm-error", action="store_true")

    # 交互模式：每次指令执行的步数
    parser.add_argument(
        "--steps-per-command",
        type=int,
        default=15,
        help="VLM 单步决策/一次性指令默认执行多少仿真步；手动键为持续速度。",
    )
    parser.add_argument(
        "--turn-circle-steps",
        type=int,
        default=120,
        help="兼容旧参数：当前 '转一圈' 已按角速度自动换算，不再使用固定步数。",
    )
    parser.add_argument(
        "--turn-half-circle-steps",
        type=int,
        default=60,
        help="兼容旧参数：当前 '转半圈' 已按角速度自动换算，不再使用固定步数。",
    )
    parser.add_argument(
        "--turn-short-steps",
        type=int,
        default=20,
        help="兼容旧参数：当前 '转一下/试试转弯' 已按角速度自动换算。",
    )
    parser.add_argument(
        "--interactive-rate",
        type=float,
        default=DEFAULT_INTERACTIVE_RATE,
        help="交互模式下 Viewer/仿真主循环刷新频率，单位 Hz。",
    )
    parser.add_argument(
        "--teleop-mode",
        type=str,
        choices=["hold", "line"],
        default="hold",
        help=(
            "交互键盘模式：hold=按住移动、松开停止；"
            "line=回车输入一整行指令。"
        ),
    )
    parser.add_argument(
        "--teleop-hold-timeout",
        type=float,
        default=DEFAULT_TELEOP_HOLD_TIMEOUT,
        help="hold 模式下多久没收到重复按键就自动停止，单位秒。",
    )
    parser.add_argument(
        "--reset-on-done",
        action="store_true",
        help="交互模式下收到 ManiSkill done 信号时自动 reset。默认忽略任务 done。",
    )

    return parser


# =========================
# 交互模式
# =========================

def start_interactive_input_thread() -> Tuple["queue.Queue[Optional[str]]", threading.Event]:
    """启动后台输入线程，避免主线程阻塞导致 SAPIEN Viewer 无响应。"""
    command_queue: "queue.Queue[Optional[str]]" = queue.Queue()
    command_processed = threading.Event()
    command_processed.set()

    def input_worker() -> None:
        while True:
            command_processed.wait()
            command_processed.clear()

            try:
                user_input = input("\n📍 输入指令: ")
            except (EOFError, KeyboardInterrupt):
                command_queue.put(None)
                return

            command_queue.put(user_input)

            if user_input.strip().lower() in ("quit", "exit", "q!", "退出"):
                return

    thread = threading.Thread(
        target=input_worker,
        name="interactive-input",
        daemon=True,
    )
    thread.start()
    return command_queue, command_processed


def start_hold_teleop_thread(
    hold_timeout: float,
) -> Tuple["queue.Queue[Optional[str]]", threading.Event]:
    """
    启动 ROS teleop 风格键盘线程。

    终端处于 raw 模式时，按键会被立即读取；长按依赖系统键盘重复。
    松开按键后，超过 hold_timeout 没收到重复按键，就自动发送停止。
    """
    command_queue: "queue.Queue[Optional[str]]" = queue.Queue()
    command_processed = threading.Event()
    command_processed.set()
    movement_keys = set("wasdqezc")
    text_mode_keys = {"t", "\r", "\n"}
    timeout = max(float(hold_timeout), 0.05)

    def enqueue_text_line(prompt: str) -> None:
        """临时恢复终端行输入，用于自然语言目标。"""
        try:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, original_attrs)
            line = input(prompt)
        except (EOFError, KeyboardInterrupt):
            command_queue.put(None)
            return
        finally:
            try:
                tty.setcbreak(sys.stdin.fileno())
            except termios.error:
                pass

        command_queue.put(line)

    def teleop_worker() -> None:
        active_key: Optional[str] = None
        last_key_time = 0.0

        try:
            tty.setcbreak(sys.stdin.fileno())
        except termios.error as e:
            print(f"\n⚠️ 无法进入 hold 键盘模式: {e}")
            command_queue.put(None)
            return

        print()
        print("🎮 hold 键盘模式已启用：按住移动，松开自动停止")
        print("   w/s/a/d/q/e/z/c = 移动，x = 停止，v = VLM 一次决策")
        print("   t 或 Enter = 输入自然语言目标/命令，Esc = 退出")

        while True:
            try:
                readable, _, _ = select.select([sys.stdin], [], [], 0.02)
            except (OSError, ValueError):
                command_queue.put(None)
                return

            now = time.time()
            if readable:
                ch = sys.stdin.read(1)

                if ch == "\x03":  # Ctrl+C
                    command_queue.put(None)
                    return
                if ch == "\x1b":  # Esc
                    command_queue.put(None)
                    return

                lower_ch = ch.lower()

                if lower_ch in text_mode_keys:
                    if active_key is not None:
                        command_queue.put("x")
                        active_key = None
                    enqueue_text_line("\n📝 输入语言目标/指令: ")
                    continue

                if lower_ch == "v":
                    if active_key is not None:
                        command_queue.put("x")
                        active_key = None
                    command_queue.put("v")
                    continue

                if lower_ch == "x":
                    active_key = None
                    command_queue.put("x")
                    continue

                if lower_ch in movement_keys:
                    last_key_time = now
                    if lower_ch != active_key:
                        active_key = lower_ch
                        command_queue.put(lower_ch)
                    continue

            if active_key is not None and now - last_key_time >= timeout:
                active_key = None
                command_queue.put("x")

    original_attrs = termios.tcgetattr(sys.stdin.fileno())

    def wrapped_worker() -> None:
        try:
            teleop_worker()
        finally:
            try:
                termios.tcsetattr(
                    sys.stdin.fileno(),
                    termios.TCSADRAIN,
                    original_attrs,
                )
            except termios.error:
                pass

    thread = threading.Thread(
        target=wrapped_worker,
        name="hold-teleop-input",
        daemon=True,
    )
    thread.start()
    return command_queue, command_processed


def run_interactive(
    env: gym.Env,
    args: argparse.Namespace,
    command_to_action: Dict[str, list],
    vlm: Optional[VLMNavigationController],
) -> None:
    """
    交互模式：用户在终端输入指令，机器人执行。

    流程：
      1. 启动仿真，弹出窗口显示画面
      2. 后台线程读取终端指令（支持快捷键）
      3. 主线程持续刷新 Viewer，避免窗口事件循环阻塞
      4. 收到指令后执行指定步数，随后自动停止
      5. 循环直到用户输入 quit
    """
    obs, info = env.reset()
    render_human_if_needed(env, args)
    args.camera_name = resolve_camera_name(obs, args.camera_name)
    base_action_slice, base_action_source = args.base_action_slice
    step_count = 0
    render_dir = Path(args.render_dir)
    task_start_time = time.time()

    print()
    print("🎮 交互模式启动")
    print("─" * 50)
    print(
        f"底盘 action 索引: [{base_action_slice[0]}:{base_action_slice[1]}] "
        f"({base_action_source})"
    )
    print(f"VLM 相机: {args.camera_name}")
    print(f"可用相机: {get_sensor_camera_names(obs)}")
    if args.show_cameras:
        print(f"   相机预览: 开启 (display_height={args.camera_display_height}px)")
        cv2.namedWindow(CAMERA_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    else:
        print("   相机预览: 关闭")
    print("可用指令：")
    if args.teleop_mode == "hold":
        print("  默认 hold 模式：按住 w/s/a/d/q/e/z/c 移动，松开自动停止")
        print("  x=停止，v=VLM 一次决策，t 或 Enter=输入自然语言目标/命令")
    else:
        print("  line 模式：输入 w/s/a/d/q/e/z/c 后回车 = 手动持续速度")
        print("  x/停止 = 停止")
    print("  定时动作：a 1.2 / 左转2秒 / w 0.8")
    print("  角度动作：左转90度 / 右转180度")
    print("  v=让VLM按默认目标看图决策一次")
    print("  也可以直接输入：前进/后退/左转/右转/停止/...")
    print("  一次性 macro：转一圈试试 / 左转一圈 / 转半圈 / 转一下")
    print("  一次性自然语言：例如“往右看一下”，只让 VLM 决策一次")
    print("  长期目标：例如“走到桌子旁/目标：走到桌子旁”，VLM 会持续低频决策")
    print("  x/停止/clear = 停止运动并清除长期目标")
    print("  quit = 退出")
    print("  提示：主循环会持续刷新 Viewer，输入新指令后立即切换动作")
    print("─" * 50)

    teleop_mode = args.teleop_mode
    if teleop_mode == "hold" and not sys.stdin.isatty():
        print("⚠️ 当前 stdin 不是 TTY，hold 模式不可用，自动切换到 line 模式")
        teleop_mode = "line"

    if teleop_mode == "hold":
        command_queue, command_processed = start_hold_teleop_thread(
            hold_timeout=args.teleop_hold_timeout,
        )
    else:
        command_queue, command_processed = start_interactive_input_thread()
    current_command = "停止"
    current_base_action = [0.0, 0.0]
    remaining_steps = 0
    loop_dt = 1.0 / max(float(args.interactive_rate), 1.0)
    last_safety_reason = ""
    last_safety_log_time = 0.0
    current_front_distance: Optional[float] = None
    long_goal: Optional[str] = None
    last_vlm_time = -1e9
    vlm_worker = AsyncVLMWorker(vlm) if vlm is not None else None
    vlm_inflight_request_id: Optional[int] = None
    next_vlm_request_id = 0
    control_generation = 0
    running = True

    def schedule_vlm_request(
        vlm_goal: str,
        is_long_goal: bool,
        label: str,
    ) -> bool:
        """在主线程采集图像快照，并把 VLM 请求提交给后台线程。"""
        nonlocal next_vlm_request_id
        nonlocal vlm_inflight_request_id
        nonlocal last_vlm_time

        if vlm is None or vlm_worker is None:
            print("   ⚠️ 未配置 VLM（缺少 OPENAI_API_KEY）")
            return False

        if vlm_inflight_request_id is not None:
            print("   ⏳ VLM 正在决策中，本次请求已跳过")
            return False

        try:
            rgb = get_camera_rgb(obs, camera_name=args.camera_name)
            image_b64 = image_to_base64(
                rgb,
                max_size=args.image_size,
            )

            created_time = time.time()
            elapsed = created_time - task_start_time
            depth = get_camera_depth(obs, camera_name=args.camera_name)
            front_distance = None
            if not args.disable_depth_safety and depth is not None:
                front_distance = estimate_front_distance(depth)

            task_prompt = vlm.build_task_prompt(
                goal=vlm_goal,
                last_command=current_command,
                elapsed_time=elapsed,
                front_distance=front_distance,
                step=step_count,
            )
        except Exception as e:
            print(f"   ⚠️ VLM 请求快照创建失败: {e}")
            return False

        next_vlm_request_id += 1
        mode = "长期目标" if is_long_goal else "一次决策"
        debug_image_path: Optional[str] = None
        debug_json_path: Optional[str] = None
        if args.save_vlm_debug:
            debug_dir = Path(args.vlm_debug_dir)
            stem = f"vlm_{next_vlm_request_id:04d}_step_{step_count:06d}"
            image_path = debug_dir / f"{stem}.jpg"
            json_path = debug_dir / f"{stem}.json"
            try:
                save_rgb_debug_image(
                    rgb,
                    image_path,
                    max_size=args.image_size,
                )
                debug_image_path = str(image_path)
                debug_json_path = str(json_path)
            except Exception as e:
                print(f"   ⚠️ VLM 调试图像保存失败: {e}")

        request = VLMAsyncRequest(
            request_id=next_vlm_request_id,
            generation=control_generation,
            goal=vlm_goal,
            image_b64=image_b64,
            task_prompt=task_prompt,
            is_long_goal=is_long_goal,
            created_step=step_count,
            created_time=created_time,
            last_command=current_command,
            front_distance=front_distance,
            mode=mode,
            debug_image_path=debug_image_path,
            debug_json_path=debug_json_path,
        )

        if not vlm_worker.submit(request):
            print("   ⏳ VLM 队列繁忙，本次请求已跳过")
            return False

        vlm_inflight_request_id = request.request_id
        last_vlm_time = created_time
        print(label, flush=True)
        print(
            f"   🔎 VLM#{request.request_id} 提交: mode={mode}, "
            f"step={step_count}, last={current_command}, "
            f"front={format_front_distance(front_distance)}, "
            f"goal={vlm_goal}"
        )
        print(
            f"   🧾 VLM#{request.request_id} prompt: "
            f"{len(task_prompt)} chars, image={args.image_size}px, "
            f"wire={args.wire_api}, model={args.model}"
        )
        if debug_json_path:
            write_vlm_debug_json(
                debug_json_path,
                {
                    "request_id": request.request_id,
                    "generation": request.generation,
                    "mode": request.mode,
                    "created_step": request.created_step,
                    "created_time": request.created_time,
                    "goal": request.goal,
                    "last_command": request.last_command,
                    "front_distance": request.front_distance,
                    "model": args.model,
                    "wire_api": args.wire_api,
                    "image_path": debug_image_path,
                    "task_prompt": task_prompt,
                    "status": "submitted",
                },
            )
            print(f"   💾 VLM#{request.request_id} debug: {debug_json_path}")
        return True

    def bump_control_generation() -> None:
        """用户接管或更换目标时，使旧 VLM 结果失效。"""
        nonlocal control_generation
        nonlocal vlm_inflight_request_id

        control_generation += 1
        vlm_inflight_request_id = None
        if vlm_worker is not None:
            vlm_worker.clear_pending()

    try:
        while running:
            loop_start = time.time()

            # 后台输入线程可能一次塞入多条指令，这里按顺序处理。
            while not command_queue.empty():
                user_input = command_queue.get()
                if user_input is None:
                    print("\n🛑 退出")
                    running = False
                    break

                parsed = parse_user_input(user_input)

                if parsed is None:
                    print("🛑 退出")
                    running = False
                    break

                if parsed == "clear_goal":
                    bump_control_generation()
                    long_goal = None
                    last_vlm_time = -1e9
                    current_command = "停止"
                    current_base_action = [0.0, 0.0]
                    remaining_steps = 0
                    print("   🧭 已清除长期目标，切换为手动停止")
                    command_processed.set()
                    continue

                if parsed.startswith("macro_turn:"):
                    bump_control_generation()
                    try:
                        _, direction, kind = parsed.split(":", maxsplit=2)
                    except ValueError:
                        direction, kind = "右转", "short"

                    long_goal = None
                    last_vlm_time = -1e9
                    current_command = direction
                    current_base_action = command_to_action.get(direction, [0.0, 0.0])

                    degrees = macro_turn_degrees(kind)
                    angular_speed = max(abs(float(args.angular_speed)), 1e-6)
                    duration_s = np.deg2rad(degrees) / angular_speed
                    remaining_steps = max(int(duration_s * args.interactive_rate), 1)

                    print(
                        f"   🔁 一次性转向 macro: {current_command}, "
                        f"kind={kind}, {degrees:.0f}° ≈ {duration_s:.2f}s, "
                        f"steps={remaining_steps}"
                    )
                    command_processed.set()
                    continue

                if parsed.startswith("timed:"):
                    bump_control_generation()
                    try:
                        _, command, seconds_text = parsed.split(":", maxsplit=2)
                        duration_s = max(float(seconds_text), 0.0)
                    except ValueError:
                        command = "停止"
                        duration_s = 0.0

                    long_goal = None
                    last_vlm_time = -1e9
                    current_command = command
                    current_base_action = command_to_action.get(command, [0.0, 0.0])
                    remaining_steps = max(int(duration_s * args.interactive_rate), 1)

                    if command == "停止":
                        remaining_steps = 0

                    print(
                        f"   ⏱️ 定时执行: {current_command} "
                        f"{duration_s:.2f}s → {current_base_action}"
                    )
                    command_processed.set()
                    continue

                if parsed.startswith("turn_angle:"):
                    bump_control_generation()
                    try:
                        _, command, degrees_text = parsed.split(":", maxsplit=2)
                        degrees = max(float(degrees_text), 0.0)
                    except ValueError:
                        command = "停止"
                        degrees = 0.0

                    long_goal = None
                    last_vlm_time = -1e9
                    current_command = command
                    current_base_action = command_to_action.get(command, [0.0, 0.0])

                    angular_speed = max(abs(float(args.angular_speed)), 1e-6)
                    duration_s = np.deg2rad(degrees) / angular_speed
                    remaining_steps = max(int(duration_s * args.interactive_rate), 1)

                    if command == "停止" or degrees <= 0.0:
                        current_command = "停止"
                        current_base_action = [0.0, 0.0]
                        remaining_steps = 0

                    print(
                        f"   🎯 角度执行: {current_command} {degrees:.1f}° "
                        f"≈ {duration_s:.2f}s → {current_base_action}"
                    )
                    command_processed.set()
                    continue

                if parsed.startswith("manual:"):
                    bump_control_generation()
                    command = parsed.removeprefix("manual:")
                    long_goal = None
                    last_vlm_time = -1e9
                    current_command = command
                    current_base_action = command_to_action.get(command, [0.0, 0.0])

                    if command == "停止":
                        remaining_steps = 0
                        print("   ⏹️ 手动停止")
                    else:
                        remaining_steps = -1
                        print(
                            f"   ▶ 手动持续: {current_command} "
                            f"→ {current_base_action}，输入 x/停止 结束"
                        )

                    command_processed.set()
                    continue

                if parsed.startswith("vlm_text:"):
                    if vlm is None:
                        print("   ⚠️ 未配置 VLM，无法启动长期语言目标")
                        command_processed.set()
                        continue

                    bump_control_generation()
                    input_goal = parsed.removeprefix("vlm_text:").strip()
                    long_goal = input_goal or args.goal
                    last_vlm_time = -1e9
                    remaining_steps = 0
                    print(f"   🧭 长期目标已更新: {long_goal}")
                    print(
                        f"   🤖 VLM 将每 {args.vlm_interval:.1f}s "
                        "看图决策一次；输入 x/clear 可停止"
                    )
                    command_processed.set()
                    continue

                if parsed == "vlm" or parsed.startswith("vlm_once:"):
                    if vlm is None:
                        print("   ⚠️ 未配置 VLM（缺少 OPENAI_API_KEY）")
                        command_processed.set()
                        continue

                    is_vlm_once = parsed.startswith("vlm_once:")
                    if is_vlm_once:
                        bump_control_generation()
                        vlm_goal = parsed.removeprefix("vlm_once:").strip() or args.goal
                        long_goal = None
                        last_vlm_time = -1e9
                        label = f"   📸 已提交 VLM 一次性理解: {vlm_goal}"
                    else:
                        vlm_goal = long_goal or args.goal
                        label = f"   📸 已提交 VLM 看图决策: {vlm_goal}"

                    schedule_vlm_request(
                        vlm_goal=vlm_goal,
                        is_long_goal=False,
                        label=label,
                    )
                    command_processed.set()
                    continue

            if not running:
                break

            if vlm_worker is not None:
                for vlm_result in vlm_worker.get_results():
                    if vlm_inflight_request_id == vlm_result.request_id:
                        vlm_inflight_request_id = None

                    if vlm_result.generation != control_generation:
                        print(
                            f"   ⏭️ VLM#{vlm_result.request_id} 丢弃过期结果: "
                            f"result_gen={vlm_result.generation}, "
                            f"current_gen={control_generation}"
                        )
                        write_vlm_debug_json(
                            vlm_result.debug_json_path,
                            {
                                "request_id": vlm_result.request_id,
                                "generation": vlm_result.generation,
                                "goal": vlm_result.goal,
                                "mode": vlm_result.mode,
                                "created_step": vlm_result.created_step,
                                "created_time": vlm_result.created_time,
                                "finished_time": vlm_result.finished_time,
                                "elapsed_s": vlm_result.elapsed_s,
                                "front_distance": vlm_result.front_distance,
                                "raw_response": vlm_result.raw_response,
                                "status": "stale",
                            },
                        )
                        continue

                    if vlm_result.error is not None:
                        print(
                            f"   ⚠️ VLM#{vlm_result.request_id} 异步决策失败 "
                            f"({vlm_result.elapsed_s:.2f}s): "
                            f"{format_vlm_error(vlm_result.error, args)}"
                        )
                        write_vlm_debug_json(
                            vlm_result.debug_json_path,
                            {
                                "request_id": vlm_result.request_id,
                                "generation": vlm_result.generation,
                                "goal": vlm_result.goal,
                                "mode": vlm_result.mode,
                                "created_step": vlm_result.created_step,
                                "created_time": vlm_result.created_time,
                                "finished_time": vlm_result.finished_time,
                                "elapsed_s": vlm_result.elapsed_s,
                                "front_distance": vlm_result.front_distance,
                                "error": format_vlm_error(vlm_result.error, args),
                                "status": "error",
                            },
                        )
                        continue

                    parsed_command = parse_command(vlm_result.raw_response)
                    parsed_action = command_to_action.get(parsed_command, [0.0, 0.0])

                    print(
                        f'   🧠 VLM#{vlm_result.request_id} 解析: '
                        f'raw="{vlm_result.raw_response}" '
                        f"→ command={parsed_command} → action={parsed_action}"
                    )

                    current_command = parsed_command
                    current_base_action = parsed_action
                    if vlm_result.is_long_goal:
                        remaining_steps = max(
                            int(args.vlm_interval * args.interactive_rate),
                            int(args.steps_per_command),
                        )
                    else:
                        remaining_steps = max(int(args.steps_per_command), 0)

                    print(
                        f"   ✅ VLM#{vlm_result.request_id} 已应用: "
                        f"mode={vlm_result.mode}, elapsed={vlm_result.elapsed_s:.2f}s, "
                        f"steps={remaining_steps}"
                    )
                    write_vlm_debug_json(
                        vlm_result.debug_json_path,
                        {
                            "request_id": vlm_result.request_id,
                            "generation": vlm_result.generation,
                            "goal": vlm_result.goal,
                            "mode": vlm_result.mode,
                            "created_step": vlm_result.created_step,
                            "created_time": vlm_result.created_time,
                            "finished_time": vlm_result.finished_time,
                            "elapsed_s": vlm_result.elapsed_s,
                            "front_distance": vlm_result.front_distance,
                            "raw_response": vlm_result.raw_response,
                            "parsed_command": parsed_command,
                            "parsed_action": parsed_action,
                            "remaining_steps": remaining_steps,
                            "status": "applied",
                        },
                    )

                    if vlm_result.is_long_goal and current_command == "停止":
                        print("   ✅ VLM 判断目标已完成或应停止，长期目标已清除")
                        long_goal = None
                        bump_control_generation()

            now = time.time()
            if (
                long_goal
                and vlm is not None
                and vlm_inflight_request_id is None
                and now - last_vlm_time >= args.vlm_interval
            ):
                schedule_vlm_request(
                    vlm_goal=long_goal,
                    is_long_goal=True,
                    label=f"   🤖 已提交长期目标 VLM 决策: {long_goal}",
                )

            exec_base_action = current_base_action
            if remaining_steps == 0:
                exec_base_action = [0.0, 0.0]

            if remaining_steps != 0 and not args.disable_depth_safety:
                depth = get_camera_depth(obs, camera_name=args.camera_name)
                front_distance = None
                if depth is not None:
                    front_distance = estimate_front_distance(depth)
                current_front_distance = front_distance

                safe_action, reason = apply_depth_safety_filter(
                    base_action=exec_base_action,
                    front_distance=front_distance,
                    obstacle_distance=args.obstacle_distance,
                    emergency_distance=args.emergency_distance,
                )
                if safe_action != exec_base_action:
                    now = time.time()
                    can_log = (
                        reason != last_safety_reason
                        or now - last_safety_log_time >= args.safety_log_interval
                    )
                    if can_log:
                        print(f"   🛡️ {reason}，执行动作: {safe_action}")
                        last_safety_reason = reason
                        last_safety_log_time = now
                    exec_base_action = safe_action

            full_action = build_full_action(
                base_action=exec_base_action,
                action_space=env.action_space,
                base_action_slice=base_action_slice,
            )
            obs, reward, terminated, truncated, info = env.step(full_action)
            render_human_if_needed(env, args)
            step_count += 1

            if args.show_cameras:
                update_camera_window(
                    obs,
                    args,
                    status_texts={
                        "指令": current_command,
                        "前方距离": format_front_distance(current_front_distance),
                        "长期目标": long_goal or "(无)",
                        "步骤": str(step_count),
                    },
                )

            if remaining_steps > 0:
                remaining_steps -= 1

            if args.save_render and step_count % args.render_every == 0:
                save_render_frame(env, step_count, render_dir)

            if args.reset_on_done and is_done(terminated, truncated):
                print("   🏁 episode 结束，自动 reset")
                obs, info = env.reset()
                render_human_if_needed(env, args)
                current_command = "停止"
                current_base_action = [0.0, 0.0]
                remaining_steps = 0

            sleep_time = loop_dt - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n🛑 用户中断 (Ctrl+C)，正在停止...")

    if vlm_worker is not None:
        vlm_worker.stop()

    if args.show_cameras:
        close_camera_window()

    # 退出前发送停止动作
    stop_action = build_full_action(
        [0.0, 0.0],
        env.action_space,
        base_action_slice=base_action_slice,
    )
    for _ in range(5):
        env.step(stop_action)
        render_human_if_needed(env, args)


# =========================
# 自动模式
# =========================

def run_auto(
    env: gym.Env,
    args: argparse.Namespace,
    command_to_action: Dict[str, list],
    vlm: VLMNavigationController,
) -> None:
    """自动模式：VLM 后台低频决策，主线程持续仿真和渲染。"""
    obs, info = env.reset()
    render_human_if_needed(env, args)
    args.camera_name = resolve_camera_name(obs, args.camera_name)
    base_action_slice, base_action_source = args.base_action_slice

    print(f"🤖 自动模式启动，目标: {args.goal}")
    print(
        f"底盘 action 索引: [{base_action_slice[0]}:{base_action_slice[1]}] "
        f"({base_action_source})"
    )
    print(f"VLM 相机: {args.camera_name}")
    print(f"可用相机: {get_sensor_camera_names(obs)}")
    if args.show_cameras:
        print(f"   相机预览: 开启 (display_height={args.camera_display_height}px)")
        cv2.namedWindow(CAMERA_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    print("─" * 60)

    current_command = "停止"
    current_base_action = [0.0, 0.0]
    task_start_time = time.time()
    last_vlm_time = -1e9
    step_count = 0
    render_dir = Path(args.render_dir)
    vlm_worker = AsyncVLMWorker(vlm)
    vlm_inflight_request_id: Optional[int] = None
    next_vlm_request_id = 0
    control_generation = 0

    def schedule_auto_vlm(
        step: int,
        front_distance: Optional[float],
        reason: str,
    ) -> bool:
        """为自动模式提交一次异步 VLM 请求。"""
        nonlocal next_vlm_request_id
        nonlocal vlm_inflight_request_id
        nonlocal last_vlm_time

        if vlm_inflight_request_id is not None:
            return False

        try:
            rgb = get_camera_rgb(obs, camera_name=args.camera_name)
            image_b64 = image_to_base64(rgb, max_size=args.image_size)
            created_time = time.time()
            elapsed_time = created_time - task_start_time

            task_prompt = vlm.build_task_prompt(
                goal=args.goal,
                last_command=current_command,
                elapsed_time=elapsed_time,
                front_distance=front_distance,
                step=step,
            )
        except Exception as e:
            print(f"[step {step:04d}] ⚠️ VLM 请求快照创建失败: {e}")
            return False

        next_vlm_request_id += 1
        debug_image_path: Optional[str] = None
        debug_json_path: Optional[str] = None
        if args.save_vlm_debug:
            debug_dir = Path(args.vlm_debug_dir)
            stem = f"auto_vlm_{next_vlm_request_id:04d}_step_{step:06d}"
            image_path = debug_dir / f"{stem}.jpg"
            json_path = debug_dir / f"{stem}.json"
            try:
                save_rgb_debug_image(
                    rgb,
                    image_path,
                    max_size=args.image_size,
                )
                debug_image_path = str(image_path)
                debug_json_path = str(json_path)
            except Exception as e:
                print(f"[step {step:04d}] ⚠️ VLM 调试图像保存失败: {e}")

        request = VLMAsyncRequest(
            request_id=next_vlm_request_id,
            generation=control_generation,
            goal=args.goal,
            image_b64=image_b64,
            task_prompt=task_prompt,
            is_long_goal=True,
            created_step=step,
            created_time=created_time,
            last_command=current_command,
            front_distance=front_distance,
            mode=f"自动:{reason}",
            debug_image_path=debug_image_path,
            debug_json_path=debug_json_path,
        )

        if not vlm_worker.submit(request):
            return False

        vlm_inflight_request_id = request.request_id
        last_vlm_time = created_time
        print(f"[step {step:04d}] 📸 已提交 VLM#{request.request_id} ({reason})")
        print(
            f"[step {step:04d}] 🔎 VLM#{request.request_id}: "
            f"last={current_command}, front={format_front_distance(front_distance)}, "
            f"goal={args.goal}"
        )
        if debug_json_path:
            write_vlm_debug_json(
                debug_json_path,
                {
                    "request_id": request.request_id,
                    "generation": request.generation,
                    "mode": request.mode,
                    "created_step": request.created_step,
                    "created_time": request.created_time,
                    "goal": request.goal,
                    "last_command": request.last_command,
                    "front_distance": request.front_distance,
                    "model": args.model,
                    "wire_api": args.wire_api,
                    "image_path": debug_image_path,
                    "task_prompt": task_prompt,
                    "status": "submitted",
                },
            )
        return True

    try:
        for step in range(args.max_steps):
            now = time.time()

            for vlm_result in vlm_worker.get_results():
                if vlm_inflight_request_id == vlm_result.request_id:
                    vlm_inflight_request_id = None

                if vlm_result.generation != control_generation:
                    print(
                        f"[step {step:04d}] ⏭️ VLM#{vlm_result.request_id} "
                        f"丢弃过期结果: result_gen={vlm_result.generation}, "
                        f"current_gen={control_generation}"
                    )
                    write_vlm_debug_json(
                        vlm_result.debug_json_path,
                        {
                            "request_id": vlm_result.request_id,
                            "generation": vlm_result.generation,
                            "goal": vlm_result.goal,
                            "mode": vlm_result.mode,
                            "created_step": vlm_result.created_step,
                            "created_time": vlm_result.created_time,
                            "finished_time": vlm_result.finished_time,
                            "elapsed_s": vlm_result.elapsed_s,
                            "front_distance": vlm_result.front_distance,
                            "raw_response": vlm_result.raw_response,
                            "status": "stale",
                        },
                    )
                    continue

                if vlm_result.error is not None:
                    print(
                        f"[step {step:04d}] ⚠️ VLM#{vlm_result.request_id} "
                        f"异步决策失败 ({vlm_result.elapsed_s:.2f}s): "
                        f"{format_vlm_error(vlm_result.error, args)}"
                    )
                    write_vlm_debug_json(
                        vlm_result.debug_json_path,
                        {
                            "request_id": vlm_result.request_id,
                            "generation": vlm_result.generation,
                            "goal": vlm_result.goal,
                            "mode": vlm_result.mode,
                            "created_step": vlm_result.created_step,
                            "created_time": vlm_result.created_time,
                            "finished_time": vlm_result.finished_time,
                            "elapsed_s": vlm_result.elapsed_s,
                            "front_distance": vlm_result.front_distance,
                            "error": format_vlm_error(vlm_result.error, args),
                            "status": "error",
                        },
                    )
                    if args.stop_on_vlm_error:
                        current_command = "停止"
                        current_base_action = [0.0, 0.0]
                        print("   切换为停止")
                    else:
                        print(f"   保持: {current_command}")
                    continue

                parsed_command = parse_command(vlm_result.raw_response)
                parsed_action = command_to_action.get(
                    parsed_command,
                    [0.0, 0.0],
                )

                print(
                    f'[step {step:04d}] 🧠 VLM#{vlm_result.request_id} 解析: '
                    f'raw="{vlm_result.raw_response}" '
                    f"→ command={parsed_command} → action={parsed_action}"
                )

                current_command = parsed_command
                current_base_action = parsed_action

                print(
                    f"[step {step:04d}] ✅ VLM#{vlm_result.request_id} 已应用: "
                    f"elapsed={vlm_result.elapsed_s:.2f}s"
                )
                write_vlm_debug_json(
                    vlm_result.debug_json_path,
                    {
                        "request_id": vlm_result.request_id,
                        "generation": vlm_result.generation,
                        "goal": vlm_result.goal,
                        "mode": vlm_result.mode,
                        "created_step": vlm_result.created_step,
                        "created_time": vlm_result.created_time,
                        "finished_time": vlm_result.finished_time,
                        "elapsed_s": vlm_result.elapsed_s,
                        "front_distance": vlm_result.front_distance,
                        "raw_response": vlm_result.raw_response,
                        "parsed_command": parsed_command,
                        "parsed_action": parsed_action,
                        "status": "applied",
                    },
                )

            # 深度检测
            front_distance = None
            if not args.disable_depth_safety:
                depth = get_camera_depth(obs, camera_name=args.camera_name)
                if depth is not None:
                    front_distance = estimate_front_distance(depth)

            # 判断是否调用 VLM
            need_vlm = (now - last_vlm_time) >= args.vlm_interval

            if (
                front_distance is not None
                and front_distance < args.obstacle_distance
                and (now - last_vlm_time) >= 1.0
            ):
                need_vlm = True

            if need_vlm and vlm_inflight_request_id is None:
                reason = "近障碍重规划" if (
                    front_distance is not None
                    and front_distance < args.obstacle_distance
                ) else "定时决策"
                schedule_auto_vlm(step, front_distance, reason)

            # 深度安全修正
            exec_base_action = current_base_action

            if not args.disable_depth_safety:
                exec_base_action, safety_reason = apply_depth_safety_filter(
                    base_action=current_base_action,
                    front_distance=front_distance,
                    obstacle_distance=args.obstacle_distance,
                    emergency_distance=args.emergency_distance,
                )

                if exec_base_action != current_base_action:
                    print(
                        f"[step {step:04d}] 🛡️ {safety_reason}，"
                        f"原={current_base_action}，执行={exec_base_action}"
                    )

            # 执行
            full_action = build_full_action(
                base_action=exec_base_action,
                action_space=env.action_space,
                base_action_slice=base_action_slice,
            )

            obs, reward, terminated, truncated, info = env.step(full_action)
            render_human_if_needed(env, args)
            step_count += 1

            if args.show_cameras:
                update_camera_window(
                    obs,
                    args,
                    status_texts={
                        "指令": current_command,
                        "前方距离": format_front_distance(front_distance),
                        "目标": args.goal,
                        "步骤": str(step_count),
                    },
                )

            if args.save_render and step_count % args.render_every == 0:
                save_render_frame(env, step_count, render_dir)

            if is_done(terminated, truncated):
                print("🏁 episode 结束，自动 reset")
                obs, info = env.reset()
                render_human_if_needed(env, args)
                current_command = "停止"
                current_base_action = [0.0, 0.0]
                last_vlm_time = -1e9
                control_generation += 1
                vlm_inflight_request_id = None
                vlm_worker.clear_pending()

    except KeyboardInterrupt:
        print("\n🛑 用户中断 (Ctrl+C)，正在停止...")

    vlm_worker.stop()

    if args.show_cameras:
        close_camera_window()

    stop_action = build_full_action(
        [0.0, 0.0],
        env.action_space,
        base_action_slice=base_action_slice,
    )
    for _ in range(5):
        env.step(stop_action)
        render_human_if_needed(env, args)


# =========================
# 主函数
# =========================

def main() -> None:
    args = build_arg_parser().parse_args()

    api_key = DEFAULT_OPENAI_API_KEY

    # 交互模式下没 Key 也可以用（手动输入指令）
    # 自动模式下必须有 Key
    if args.auto and not api_key:
        print("⚠️  自动模式需要 OPENAI_API_KEY。")
        print()
        print("请先设置：")
        print('  export OPENAI_API_KEY="你的key"')
        return

    # 创建 VLM 控制器（交互模式下没有 Key 就不创建）
    vlm: Optional[VLMNavigationController] = None
    if api_key:
        try:
            vlm = VLMNavigationController(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                wire_api=args.wire_api,
                temperature=0.1,
                timeout=30.0,
            )
        except ValueError:
            pass

    command_to_action = make_command_to_action(
        linear_speed=args.linear_speed,
        angular_speed=args.angular_speed,
    )

    mode_str = "自动" if args.auto else "交互"
    print(f"🤖 VLM 底盘导航控制 — {mode_str}模式")
    print(f"   模型: {args.model}")
    print(f"   API 地址: {args.base_url}")
    print(f"   接口协议: {args.wire_api}")
    print("   VLM 调用: 异步后台线程")
    print(f"   API Key: {'已配置' if api_key else '未配置'}")
    print(f"   环境: {args.env_id}")
    print(f"   渲染模式: {args.render_mode}")
    print(f"   Shader: {args.shader}")
    print(f"   机器人: {args.robot_uid or '环境默认'}")
    if args.auto:
        print(f"   任务目标: {args.goal}")
        print(f"   VLM 调用间隔: {args.vlm_interval:.1f}s")
    print(f"   线速度: {args.linear_speed:.2f}")
    print(f"   角速度: {args.angular_speed:.2f}")
    print(f"   深度安全: {'关闭' if args.disable_depth_safety else '开启'}")
    print(f"   相机预览: {'开启' if args.show_cameras else '关闭'}")
    if not args.disable_depth_safety:
        print(
            f"   安全距离: 避障 {args.obstacle_distance:.2f}m / "
            f"急停 {args.emergency_distance:.2f}m"
        )
    if not args.auto:
        print(f"   VLM/一次性动作步数: {args.steps_per_command}")
        print("   转向 macro: 按角速度自动换算 一圈/两圈/半圈/短转")
        print(f"   交互刷新频率: {args.interactive_rate:.1f} Hz")
        print(f"   键盘模式: {args.teleop_mode}")
        if args.teleop_mode == "hold":
            print(f"   松手停止超时: {args.teleop_hold_timeout:.2f}s")
        print(f"   长期目标 VLM 间隔: {args.vlm_interval:.1f}s")
        print(f"   done 自动 reset: {'开启' if args.reset_on_done else '关闭'}")
    print()

    # 创建仿真环境，渲染参数与 ManiSkill demo_random_action 保持一致。
    env_kwargs = {
        "obs_mode": args.obs_mode,
        "render_mode": args.render_mode,
        "sensor_configs": {"shader_pack": args.shader},
        "human_render_camera_configs": {"shader_pack": args.shader},
        "viewer_camera_configs": {"shader_pack": args.shader},
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
        "enable_shadow": True,
    }
    if args.robot_uid:
        env_kwargs["robot_uids"] = args.robot_uid
    if args.control_mode:
        env_kwargs["control_mode"] = args.control_mode

    env = gym.make(args.env_id, **env_kwargs)

    print("✅ ManiSkill 环境已启动")
    print(f"   env_id: {args.env_id}")
    print(f"   robot: {args.robot_uid or '环境默认'}")
    print(f"   shader: {args.shader}")
    print(f"   action_space: {env.action_space}")

    base_start, base_end, base_source = resolve_base_action_slice(
        env=env,
        action_space=env.action_space,
        override_start=args.base_action_start,
    )
    args.base_action_slice = ((base_start, base_end), base_source)
    print(f"   base_action: action[{base_start}:{base_end}] ({base_source})")

    try:
        if args.auto:
            run_auto(env, args, command_to_action, vlm)
        else:
            run_interactive(env, args, command_to_action, vlm)
    finally:
        if args.show_cameras:
            close_camera_window()
        env.close()
        print("✅ 控制结束")


if __name__ == "__main__":
    main()
