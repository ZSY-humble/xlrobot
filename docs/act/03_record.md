# 3️⃣ 数据采集（核心 ⭐）

> 边人手推左臂、边录右臂动作 + 相机 → 喂给 ACT 训练。

---

## 🎯 一键启动

```bash
python act/record_self_teleop.py
```

可选参数：
```bash
python act/record_self_teleop.py --resume                          # 续采
python act/record_self_teleop.py --num-episodes=80                 # 临时改段数
python act/record_self_teleop.py --task="新的英文任务描述"
python act/record_self_teleop.py --push-to-hub                     # 录完上传 HF
python act/record_self_teleop.py --no-display                      # 不开 Rerun
```

更多参数通过环境变量（见 [act/config.py](../../act/config.py)）：
```bash
NUM_EPISODES=100 EPISODE_TIME_S=25 RESET_TIME_S=12 \
python act/record_self_teleop.py
```

---

## ⌨️ 二、采集时三键（必须背下来）

| 按键 | 作用 | 时机 |
|---|---|---|
| `→` 或 `s` | **结束本段并保存** | 这一段做得不错 |
| `←` 或 `r` | **丢弃本段，重录** | 主臂滑了 / 物体没摆好 / 失败了 |
| `Esc` 或 `q` | **整体停止** | 想中途结束保存已采的 |

⚠️ **必须本地物理终端**（SSH 远程终端收不到键盘事件）。
⚠️ Linux 监听键盘可能要 `sudo` 或加入 `input` 组。

---

## 🎬 三、Rerun 边采边看

启动后自动弹 Rerun 窗口，里面有：
- 🎥 **两路相机实时画面**（`top` + `right_wrist`）
- 📈 **action 曲线**：`action.right_arm_*.pos`（镜像后的目标）
- 📈 **observation 曲线**：`observation.right_arm_*.pos`（右臂实测）

### 边采边自检

| 看哪里 | 应该是什么样 |
|---|---|
| 两路相机 | 都正常显示，无黑屏，无卡顿 |
| action vs observation | 几乎重合 |
| 终端帧率 | 稳定 30 Hz |

发现不对立刻按 `←` 重录，**别凑合**。

---

## 📦 四、数据集 schema（事先了解）

| 字段 | 维度 | 内容 |
|---|---|---|
| `observation.state` | 6 | 仅右臂 6 关节 |
| `observation.images.top` | RGB 480×640 | 桌面相机 |
| `observation.images.right_wrist` | RGB 480×640 | 右腕相机 |
| `action` | 6 | 右臂 6 关节目标位置（镜像后） |

**不录入**：左臂、头部、底盘、左腕相机。

---

## 🎯 五、采集质量黄金法则

### 1. 任务描述清晰
- ✅ "Pick up the red cube on the left and place it into the white box on the right"
- ❌ "do the task"

### 2. 起始姿态规整
每段开始前，把右臂归到固定 home pose。
左臂是"你将开始示范的初始位置"，每段也应大致一致。

### 3. 节奏稳定
- 每段动作时长尽量一致（差异 ≤ 50%）
- 推动主臂的速度别忽快忽慢

### 4. 失败立刻 ←
- 任何不满意立刻按 `←`，**别想着"凑合用"**
- 数据质量 > 数量

### 5. 适度多样化
- 物体位置在合理范围内移动（5–10cm 量级）
- 50 段全摆同一位置 → 模型只会记一个动作

### 6. 操作员安全
- 启动前**用手扶住左臂**（关扭矩瞬间会下垂）
- 桌面留缓冲空间
- 长时间采集中途想休息：`Esc` → 体息 → 重启 `--resume`

---

## 📁 六、采集结果在哪

```
~/.cache/huggingface/lerobot/${HF_USER}/xlerobot_act_self_teleop/
├── meta/                # features / episodes / stats
├── data/chunk-000/      # parquet（关节 / action）
└── videos/chunk-000/    # 两路相机视频（默认 libsvtav1）
```

---

## ⚠️ 七、常见坑

| 现象 | 解决 |
|---|---|
| 启动时 `ENTER 还是 c?` | XLeRobot 问你是否复用旧标定，直接回车 |
| 左臂没法推 | 扭矩没关；看终端有没有 `左臂扭矩已关闭` 日志 |
| 右臂某关节方向反 | 没标定 / `NEGATE_JOINTS` 没配；先跑 [02_teleoperate.md](02_teleoperate.md) probe |
| 三键无响应 | 必须本地终端，可能要 `sudo` |
| Rerun 不弹窗 | `--no-display` 或 headless 环境，加 `--display_ip` 远程查看（参考 lerobot-record） |
| 帧率掉到 < 25 fps | 降相机分辨率 / 关 Rerun 试试 |

---

## ➡️ 录完进入下一步

[04_visualize_dataset.md](04_visualize_dataset.md)：回看每段，删坏样本。
