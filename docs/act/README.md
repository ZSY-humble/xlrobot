# 🤖 XLeRobot 自我遥操作 ACT 工作流

> **核心模式**：**自我遥操作（self-teleoperation）** —— 关掉 XLeRobot **左臂**扭矩，
> 你用手推动左臂模拟示范动作，**右臂**实时跟随并被记录用于训练 ACT 策略。
>
> 🎯 训出来的策略只控制 **右臂**，部署时不再需要左臂被推。
>
> 环境：`conda activate lerobot`

---

## 📐 角色分工（务必先理清）

| 角色 | 物理位置 | 扭矩 | 职责 |
|---|---|---|---|
| **左臂**（leader / 主推） | XLeRobot port1 总线 | ❌ **关闭**（人手才能推动） | 你用手推 → 关节角作为右臂目标 |
| **右臂**（follower / 跟随） | XLeRobot port2 总线 | ✅ 开启 | 实时跟随左臂；其状态 + 动作被录到数据集 |
| **头部** 2 个电机 | port1 总线（与左臂同 bus） | ✅ 开启（伺服锁位） | 不录、不主动控制（保持当前姿态即可） |
| **底盘** 3 个轮子 | port2 总线（与右臂同 bus） | — | 本任务**不用** |

---

## 🎥 相机布局

| 相机 | 位置 | 是否录入数据集 |
|---|---|---|
| `top` | 桌面外部三脚架 | ✅ |
| `right_wrist` | 右臂末端（夹爪附近） | ✅ |
| `left_wrist` | 左臂末端 | ❌ **可装可不装**，不录入（左臂只是动作输入） |

---

## 📦 数据集 schema（关键设计）

| 字段 | 维度 | 内容 |
|---|---|---|
| `observation.state` | **6** | 仅右臂 6 关节（不录头部 / 底盘） |
| `observation.images.top` | RGB | 桌面相机 |
| `observation.images.right_wrist` | RGB | 右腕相机 |
| `action` | **6** | 右臂 6 关节目标位置（= 左臂当前位置经左右镜像变换） |

> 💡 **为什么 action 不录左臂直接的关节角？**
> 因为左右两臂物理上**镜像安装**，左臂的 `shoulder_pan / wrist_roll` 等关节方向与右臂相反。
> 我们用一个独立模块 [`act/mirror.py`](../../act/mirror.py) 做左→右关节角映射，并支持
> `python -m act.mirror probe` 实测哪些关节要取反。

---

## 📂 项目分布

### 📖 文档（你正在看）：`docs/act/`
```
docs/act/
├── README.md                    👈 本文件，工作流总览
├── PROPOSAL_self_teleop.md      方案设计说明（一次性看清思路）
├── 00_env_setup.md              环境与硬件准备
├── 01_calibrate.md              XLeRobot 标定
├── 02_teleoperate.md            自我遥操作验证 + 镜像表调试
├── 03_record.md                 ⭐ 数据采集
├── 04_visualize_dataset.md      回看数据集 + 删坏段
├── 05_train_act.md              ACT 训练
├── 06_replay_and_eval.md        回放 / 真机推理（仅控右臂）
└── 07_troubleshooting.md        故障排查
```

### 🐍 脚本：`act/`
```
act/
├── __init__.py
├── config.py                    集中配置（端口/相机/超参）
├── mirror.py             ⭐    左→右镜像表 + probe 子命令
├── calibrate_xlerobot.py        XLeRobot 整机标定入口
├── teleoperate_self.py          自我遥操作（不录数据）
├── record_self_teleop.py  ⭐   数据采集（自写主循环）
├── train_act.py                 ACT 训练
└── eval_act.py                  真机推理（仅右臂）
```

---

## 🚦 推荐流程（按顺序走）

| 步骤 | 文档 | 一句话 |
|---|---|---|
| 0️⃣ | [00_env_setup.md](00_env_setup.md) | 装环境、定 2 路 USB 端口、配相机 |
| 1️⃣ | [01_calibrate.md](01_calibrate.md) | XLeRobot 整机标定（双臂 + 头部） |
| 2️⃣ | [02_teleoperate.md](02_teleoperate.md) | 验证左推右跟，**调镜像表** |
| 3️⃣ | [03_record.md](03_record.md) | **🌟 正式采集**，Rerun 边采边看 |
| 4️⃣ | [04_visualize_dataset.md](04_visualize_dataset.md) | 回看每段，删坏样本 |
| 5️⃣ | [05_train_act.md](05_train_act.md) | ACT 训练（state=6, action=6） |
| 6️⃣ | [06_replay_and_eval.md](06_replay_and_eval.md) | 真机部署：模型直接控右臂 |
| ❓ | [07_troubleshooting.md](07_troubleshooting.md) | 出问题先翻这个 |

---

## ⌨️ 采集时三键

| 按键 | 作用 |
|---|---|
| `→` | 本段成功 → 保存并进入下一段 |
| `←` | 本段失败 → 丢弃并重录 |
| `Esc` | 整体停止采集 |

---

## ⚠️ 操作员安全须知（务必读）

1. **左臂关扭矩瞬间会因重力下垂** —— 启动前**先用手扶住**左臂再按确认
2. **起始姿态选"重力影响最小"位置**（关节立起来、夹爪朝下）
3. **桌面要留缓冲** —— 防止左臂下垂磕碰物体或自己
4. **头部和底盘电机不要碰** —— 它们扭矩没关，强行推可能报错或损坏齿轮

---

## 🌟 核心特性

- ✅ **真正的自我遥操作**：仅 1 台 XLeRobot，不需要额外主臂硬件
- ✅ **左→右镜像表可独立调** ([act/mirror.py](../../act/mirror.py))，支持逐关节交互式标定
- ✅ **边采集边 Rerun 可视化**：相机 + state + action 曲线
- ✅ **失败一键重录**：`←` 键秒级丢弃当前段
- ✅ **流式视频编码**：长时间采集不爆内存

---

## 🔗 相关源码

- XLeRobot 机器人类：[../../src/lerobot/robots/xlerobot/xlerobot.py](../../src/lerobot/robots/xlerobot/xlerobot.py)
- 标定脚本（lerobot CLI）：[../../src/lerobot/scripts/lerobot_calibrate.py](../../src/lerobot/scripts/lerobot_calibrate.py)
- 训练脚本：[../../src/lerobot/scripts/lerobot_train.py](../../src/lerobot/scripts/lerobot_train.py)
- LeRobotDataset：[../../src/lerobot/datasets/lerobot_dataset.py](../../src/lerobot/datasets/lerobot_dataset.py)

---

## 📝 版本

2026-06-18 · 阿宇 · 自我遥操作专版
