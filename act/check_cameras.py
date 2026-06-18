#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""预览本机 OpenCV/V4L2 相机端口。

用途：
  1. 找出哪些 /dev/video* 真正能出图
  2. 判断哪个是 ACT 数据采集用的 top 相机
  3. 判断哪个是 ACT 数据采集用的 right_wrist 相机

示例：
    python act/check_cameras.py
    python act/check_cameras.py --devices /dev/video0 /dev/video2 /dev/video4
    python act/check_cameras.py --width 640 --height 480 --fps 30

窗口快捷键：
    q / Esc  退出
    s        保存当前所有可用相机画面到 /tmp/xlerobot_camera_check
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# 兜底：允许 `python act/check_cameras.py` 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from act.config import CONFIG


WINDOW_NAME = "XLeRobot camera check"


@dataclass
class CameraSlot:
    device: str
    cap: cv2.VideoCapture
    last_frame: np.ndarray | None = None
    last_ok_time: float = 0.0
    frames: int = 0
    fps: float = 0.0
    fps_t0: float = 0.0


def _existing_video_devices() -> list[str]:
    """按编号顺序返回当前存在的 /dev/video* 设备。"""
    devices = sorted(
        Path("/dev").glob("video*"),
        key=lambda p: int(p.name.replace("video", "")) if p.name[5:].isdigit() else 9999,
    )
    return [str(p) for p in devices if p.name[5:].isdigit()]


def _open_camera(device: str, width: int, height: int, fps: int, mjpg: bool) -> CameraSlot | None:
    """打开单个相机；失败则返回 None。"""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        print(f"⚠️  无法打开: {device}")
        return None

    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # 先读几帧，避开部分 UVC 相机启动时的空帧。
    ok = False
    frame = None
    for _ in range(10):
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        time.sleep(0.03)

    if not ok or frame is None:
        cap.release()
        print(f"⚠️  已打开但读不到画面: {device}")
        return None

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✅ 可用: {device}  实际输出: {real_w}x{real_h} @ {real_fps:.1f}fps")

    now = time.perf_counter()
    return CameraSlot(
        device=device,
        cap=cap,
        last_frame=frame,
        last_ok_time=now,
        frames=1,
        fps_t0=now,
    )


def _put_label(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """在画面左上角绘制端口和状态。"""
    out = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.58
    thickness = 1
    line_h = 24
    pad = 8
    box_h = pad * 2 + line_h * len(lines)
    cv2.rectangle(out, (0, 0), (out.shape[1], box_h), (0, 0, 0), -1)
    for idx, line in enumerate(lines):
        y = pad + 18 + idx * line_h
        cv2.putText(
            out,
            line,
            (pad, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return out


def _blank_tile(width: int, height: int, text: str) -> np.ndarray:
    """生成空白占位图。"""
    tile = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        tile,
        text,
        (20, max(40, height // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return tile


def _make_mosaic(slots: list[CameraSlot], tile_w: int, tile_h: int) -> np.ndarray:
    """把多路相机拼成一个窗口。"""
    if not slots:
        return _blank_tile(tile_w, tile_h, "No camera frames")

    cols = min(3, max(1, math.ceil(math.sqrt(len(slots)))))
    rows = math.ceil(len(slots) / cols)
    tiles: list[np.ndarray] = []
    now = time.perf_counter()

    for slot in slots:
        ok, frame = slot.cap.read()
        if ok and frame is not None:
            slot.last_frame = frame
            slot.last_ok_time = now
            slot.frames += 1

        if now - slot.fps_t0 >= 1.0:
            slot.fps = slot.frames / max(now - slot.fps_t0, 1e-6)
            slot.frames = 0
            slot.fps_t0 = now

        if slot.last_frame is None:
            tile = _blank_tile(tile_w, tile_h, f"{slot.device}: no frame")
        else:
            tile = cv2.resize(slot.last_frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            age_ms = int((now - slot.last_ok_time) * 1000)
            tile = _put_label(
                tile,
                [
                    f"{slot.device}",
                    f"fps={slot.fps:.1f}  age={age_ms}ms",
                ],
            )
        tiles.append(tile)

    while len(tiles) < rows * cols:
        tiles.append(_blank_tile(tile_w, tile_h, ""))

    row_images = []
    for row in range(rows):
        start = row * cols
        row_images.append(np.hstack(tiles[start : start + cols]))
    return np.vstack(row_images)


def _save_frames(slots: list[CameraSlot], save_dir: Path) -> None:
    """保存当前帧，便于远程排查。"""
    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for slot in slots:
        if slot.last_frame is None:
            continue
        name = slot.device.replace("/", "_").strip("_")
        path = save_dir / f"{stamp}_{name}.jpg"
        cv2.imwrite(str(path), slot.last_frame)
        print(f"💾 已保存: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="XLeRobot 相机端口预览工具")
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="指定要测试的设备，例如 /dev/video0 /dev/video2 /dev/video4",
    )
    parser.add_argument("--width", type=int, default=CONFIG.cam_width, help="请求相机宽度")
    parser.add_argument("--height", type=int, default=CONFIG.cam_height, help="请求相机高度")
    parser.add_argument("--fps", type=int, default=CONFIG.cam_fps, help="请求相机帧率")
    parser.add_argument("--tile-width", type=int, default=480, help="预览窗口中每路画面宽度")
    parser.add_argument("--tile-height", type=int, default=360, help="预览窗口中每路画面高度")
    parser.add_argument("--no-mjpg", action="store_true", help="不强制请求 MJPG 格式")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("/tmp/xlerobot_camera_check"),
        help="按 s 保存截图的目录",
    )
    args = parser.parse_args()

    devices = args.devices if args.devices else _existing_video_devices()
    if not devices:
        print("❌ 没找到 /dev/video*，请先确认相机 USB 已连接。")
        return 1

    print("\n🎯 XLeRobot 相机端口预览")
    print(f"   测试设备: {', '.join(devices)}")
    print(f"   请求参数: {args.width}x{args.height} @ {args.fps}fps")
    print("   快捷键  : q/Esc 退出，s 保存当前画面")
    print()

    slots: list[CameraSlot] = []
    for device in devices:
        slot = _open_camera(
            device=device,
            width=args.width,
            height=args.height,
            fps=args.fps,
            mjpg=not args.no_mjpg,
        )
        if slot is not None:
            slots.append(slot)

    if not slots:
        print("\n❌ 没有任何设备能读到画面。")
        print("💡 可检查：USB 线、相机权限、是否被其他程序占用。")
        return 2

    print("\n💡 看窗口里的画面标签，记下：")
    print("   top         = 桌面 / 俯视 / 外部相机")
    print("   right_wrist = 右腕相机")
    print()
    print("例如确认后设置：")
    print("   export CAM_TOP=/dev/videoX")
    print("   export CAM_RIGHT_WRIST=/dev/videoY")
    print()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        while True:
            mosaic = _make_mosaic(slots, args.tile_width, args.tile_height)
            cv2.imshow(WINDOW_NAME, mosaic)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                _save_frames(slots, args.save_dir)
    finally:
        for slot in slots:
            slot.cap.release()
        cv2.destroyAllWindows()

    print("✅ 相机预览已退出。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
