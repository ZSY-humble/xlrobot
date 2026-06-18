# 7️⃣ 故障排查总表

> 出问题先翻这。按出现频率从高到低排序。

---

## 🚨 自我遥操作专属坑（先看这块）

| 现象 | 原因 | 解决 |
|---|---|---|
| 启动后左臂还推不动 | 扭矩没关 | 看终端有没有 `左臂扭矩已关闭`；XLeRobot.connect() 会启用全扭矩，必须**之后**再关 |
| 左臂关扭矩瞬间臂塌下来撞桌 | 重力 + 无 GC | 启动**前**用手扶住左臂；起始位选立直姿态 |
| 右臂某些关节方向反 | 镜像表没配 | `python -m act.mirror probe` 实测；改 `DEFAULT_NEGATE_JOINTS` |
| 头部跟着乱动 | 扭矩误关到头部 | 检查 `disable_torque(left_arm_motors)` 不是 `disable_torque()`（无参=关全部） |
| 录的数据里 action 维度对不上 | schema 不一致 | 必须用 `act/record_self_teleop.py` 录，不要混用旧 lerobot-record 数据 |

---

## 🔌 一、连接 / 端口问题

| 现象 | 原因 | 解决 |
|---|---|---|
| `Could not open port /dev/ttyACMx` | 端口占用 / 不存在 / 权限 | `ls /dev/ttyACM*` 确认；`sudo usermod -aG dialout $USER` 后**重新登录** |
| 端口编号每次开机变 | USB 枚举顺序不固定 | 用 `udev` 起固定别名（见 [00_env_setup.md](00_env_setup.md)） |
| `Permission denied: /dev/ttyACMx` | 用户没在 dialout 组 | 同上 |
| port1 / port2 接反了 | 左右臂跟随不对 | 互换 `--robot.port1` / `port2`（或物理换接） |

---

## 📹 二、相机问题

| 现象 | 原因 | 解决 |
|---|---|---|
| 相机黑屏 | 被占用 | `fuser -v /dev/video*` 看占用，杀掉 |
| `Camera /dev/videoX cannot be opened` | 不存在 / 接错 | `ls /dev/video*` |
| 录到一半某路相机掉 | USB 带宽不够 | 分散到不同 USB 控制器；降分辨率 |
| 视频画面颠倒 | 物理装反 | 加 `"rotation": "ROTATE_180"` 到相机 config |
| 同一摄像头多个 /dev/videoN | V4L2 一个相机会注册多个节点 | 一般 `/dev/video0` 是数据流 |

---

## 🎯 三、标定问题

| 现象 | 原因 | 解决 |
|---|---|---|
| `calibration not found for xlerobot_main` | 没标定 | 走 [01_calibrate.md](01_calibrate.md) |
| 启动时被问 ENTER 还是 c | 在询问是否复用旧标定 | 直接回车 = 用旧；输入 `c` = 重标 |
| 标完跟随方向反 | 极限位推反了 | 重标 |
| 标完行程缩了一半 | 没推到真极限 | 重标 |

---

## ⌨️ 四、键盘按键无响应

| 现象 | 解决 |
|---|---|
| `→` `←` `Esc` 全没反应 | 必须本地物理终端，不能 SSH |
| 偶尔响应 | 必要时 `sudo` 启动 |
| 报 `pynput` 相关错 | `pip install pynput` |
| 在 docker 里没反应 | 容器要挂载 `/dev/input` 并 `--privileged` |

---

## 🎬 五、Rerun 可视化问题

| 现象 | 原因 | 解决 |
|---|---|---|
| `--display_data=true` 没弹窗 | headless / WSL / 纯 SSH | 加 `--display_ip=0.0.0.0 --display_port=9876` 远程查看 |
| Rerun 卡顿 | 图像太多 / 网络慢 | 降相机分辨率 |
| 装不上 rerun-sdk | pip 源问题 | `pip install rerun-sdk` |
| 远程连不上 | 防火墙 | `sudo ufw allow 9876` |

---

## 📦 六、数据集 / 录制问题

| 现象 | 原因 | 解决 |
|---|---|---|
| `Unknown robot type 'xlerobot'` | 包没重装 / 老 lerobot | `pip install -e .` |
| feature 校验失败 | 历史数据集 schema 与新代码不一致 | 用 `lerobot-edit-dataset --operation.type=info` 看真实 features |
| 帧率掉到 < 25 fps | CPU/GPU 瓶颈 | 减相机数 / 降分辨率 / 增 `encoder_threads` |
| 视频编码报错 | ffmpeg 缺编码器 | 改 `vcodec=h264`（更兼容） |
| 数据集存不进 / 磁盘满 | 流式编码也得有空间 | 检查 `~/.cache` 所在分区 |
| `add_frame` 报缺 task | self_teleop 已写明 task | 自查 frame 字典是否含 `"task"` |

---

## 🧠 七、训练问题

| 现象 | 原因 | 解决 |
|---|---|---|
| OOM | batch 太大 | 调小 `--batch_size`，或换 `vision_backbone=resnet18` |
| Loss NaN | 学习率 / 数据问题 | `--optimizer.lr=1e-5`；检查极端值 |
| Loss 震荡不降 | 数据脏 / 模型容量不够 | 清数据 / `dim_model=768` |
| 训练慢 | num_workers 太少 | `--dataloader.num_workers=4` |
| `dataset not found` | repo_id 拼错 / 没下到本地 | 检查 `~/.cache/huggingface/lerobot/` |

---

## 🤖 八、推理 / 部署问题

| 现象 | 原因 | 解决 |
|---|---|---|
| 模型抖动 | 数据脏 / 训练步数不够 | 清数据 / 训到 200k step |
| 总抓空 | 测试时相机角度变了 | 摆回训练时位置 |
| 推理频率不稳 | GPU 抢占 | 关其它 GPU 进程 |
| 加载 checkpoint 报错 | 路径错 | 路径到 `checkpoints/last/pretrained_model` |
| 模型对夹爪没反应 | 训练数据夹爪动作太少 | 重采时多做开合 |
| 左臂在推理时也动了 | 不应该 | 检查 policy 输出维度，应 = 6（仅 right_arm） |

---

## 🩹 九、紧急止损

### 真机失控（动作异常 / 关节快撞）
1. **立刻按急停 / 拔电**
2. 不要先 Ctrl+C（可能延迟生效）

### 终端进程僵尸了
```bash
pkill -9 -f lerobot
pkill -9 -f "python act/"
ps aux | grep -E "lerobot|act/" | grep -v grep
```

### 端口被占用解不开
```bash
fuser -k /dev/ttyACM0   # 强杀占用进程
```

---

## 📞 找不到答案？

1. 看终端完整报错（往上翻所有 traceback）
2. 翻当前 docs：[README.md](README.md)
3. 查源码：
   - XLeRobot：[../../src/lerobot/robots/xlerobot/xlerobot.py](../../src/lerobot/robots/xlerobot/xlerobot.py)
   - 录制脚本：[../../act/record_self_teleop.py](../../act/record_self_teleop.py)
   - 镜像表：[../../act/mirror.py](../../act/mirror.py)
4. 把报错粘给阿宇 👋
