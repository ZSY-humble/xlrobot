# XLeRobot 自我遥操作 ACT 数据采集方案

> **版本**：2026-06-18 · **作者**：阿宇
> **目的**：用 1 台 XLeRobot（双 SO101 臂），通过"一臂主推、另一臂跟随"的自我遥操作方式，
> 采集训练 ACT 单臂策略所需的数据集。

---

## 1. 背景与动机

### 1.1 硬件实际情况
- 用户**只有 1 台 XLeRobot**（自带左右两根 SO101 臂 + 头部 + 底盘）
- **没有额外的独立主臂硬件**（即不存在挂在桌外的两根 SO101 leader）
- 头部 2 个舵机、两轮底盘 2 个电机，但本任务**不用底盘**
- 相机：桌面相机 + 双臂腕部相机

### 1.2 为什么不用现成的双独立主臂方案
之前文档假设 4 路 USB（双独立 SO101 主臂 + XLeRobot 双臂），与实际硬件不符——**用户根本没有那 2 根独立主臂**。所有依赖 `bi_xlerobot_leader` + `lerobot-record` 的命令都不能用。

### 1.3 自我遥操作（self-teleoperation）思路
> 把 XLeRobot 的**左臂**关掉扭矩当 leader（人手推动），**右臂**保持扭矩当 follower（实时跟随并被记录）。
> 训出的策略只控**右臂**，部署时不再需要左臂被推。

这是机器人学习领域的成熟模式（也叫 kinesthetic teleoperation / 拖动示教），适合"没有专用主臂硬件 + 想训单臂策略"的场景。

---

## 2. 角色分工

| 角色 | 物理 | 总线 | 扭矩 | 数据流 |
|---|---|---|---|---|
| **左臂** | XLeRobot 左 SO101 | bus1 (port1) | ❌ 关闭 | 关节角作为右臂目标 |
| **右臂** | XLeRobot 右 SO101 | bus2 (port2) | ✅ 开启 | **状态 + 动作录入数据集** |
| **头部 2 电机** | port1（与左臂同 bus） | bus1 | ✅ 开启 | 不录入数据集（保持不动） |
| **两轮底盘 2 电机** | port2（与右臂同 bus） | bus2 | — | 不使用 |
| **桌面相机** `top` | 外部三脚架 | - | - | **录入** |
| **右腕相机** `right_wrist` | 右臂末端 | - | - | **录入** |
| **左腕相机** `left_wrist` | 左臂末端（可选） | - | - | 不录入 |

---

## 3. 数据集 schema

| 字段 | 维度 | 含义 |
|---|---|---|
| `observation.state` | **6** | 右臂 6 关节 `pos` |
| `observation.images.top` | RGB 480×640 | 桌面相机 |
| `observation.images.right_wrist` | RGB 480×640 | 右腕相机 |
| `action` | **6** | 右臂 6 关节目标位置 = **左臂当前位置经左右镜像变换** |

> 💡 **action 不直接录左臂关节角**。因为左右臂物理上**镜像安装**，
> 部分关节（如 `shoulder_pan`、`wrist_roll`）方向相反，必须经过镜像映射
> 才能成为右臂的合法 `Goal_Position`。

---

## 4. 关键技术决策

### 4.1 为什么不能用 `lerobot-record`
- `lerobot-record` 强制要求 `--teleop` 或 `--policy` 之一
  （[lerobot_record.py:243-244](src/lerobot/scripts/lerobot_record.py#L243-L244)）
- 自我遥操作两者都没有：action 来源是 robot 自己的另一半，不是独立 teleop 设备
- **结论**：自写 record 主循环（参考 [examples/8_vr_teleop_with_dataset_recording.py](examples/8_vr_teleop_with_dataset_recording.py) 的"自写循环"模式）

### 4.2 仅对左臂关扭矩（不影响头部）
- `XLerobot` 类暴露了 `bus1` / `bus2` 两个独立总线和 `left_arm_motors / head_motors / right_arm_motors` 名称列表
- 调用 `robot.bus1.disable_torque(robot.left_arm_motors)` **只松左臂 6 个电机**，头部 2 个电机扭矩保持
- ⚠️ 时序：`robot.connect()` 内部会调 `configure()` 启用全部扭矩，必须**在 connect 之后**再 disable

### 4.3 镜像表独立模块
- 抽出 `act/mirror.py`，包含 `LEFT_TO_RIGHT_NAMES` 字典 + `NEGATE_JOINTS` 集合
- 提供 `python -m act.mirror probe` 子命令：连机器人后逐关节交互式标定哪些关节要取反
- 镜像参数与采集逻辑解耦，方便实验调试

### 4.4 部分动作下发
- `XLerobot.send_action()` 按 key 前缀过滤，只控传入键对应的电机
- 我们传 `{**right_arm_target, **head_hold}`：**右臂跟随 + 头部锁定，不动左臂**
- 左臂被人手自由推动，不会被电机反向拉

---

## 5. 文件分布

### 5.1 文档：`docs/act/`（共 9 个 md，全部重写）
```
README.md                    总览（自我遥操作流程）
00_env_setup.md              2 路 USB + 2 相机
01_calibrate.md              XLeRobot 整机标定（一次完成左右臂+头）
02_teleoperate.md            自我遥操作验证 + 镜像表 probe
03_record.md          ⭐    数据采集
04_visualize_dataset.md      回看 + 删坏段
05_train_act.md              ACT 训练（state=6, action=6）
06_replay_and_eval.md        推理：模型直接控右臂
07_troubleshooting.md        故障排查
```

### 5.2 脚本：`act/`（6 个 .py）
```
__init__.py
config.py                   集中配置（端口、相机、超参）
mirror.py             ⭐    左→右镜像表 + probe 子命令
calibrate_xlerobot.py       XLeRobot 整机标定包装
teleoperate_self.py         自我遥操作（不录数据，调试用）
record_self_teleop.py ⭐   数据采集（自写主循环）
train_act.py                ACT 训练包装
eval_act.py                 真机推理（policy → 仅右臂）
```

### 5.3 上游源码改动
- **保留**：`xlerobot` 在 record/teleoperate/calibrate/replay 4 个脚本里的 import 修复（这是真实的 bug 修复，与方案无关）
- **保留**：`bi_xlerobot_leader` 模块本身（未来若接独立主臂可复用，不影响当前方案）
- **不改**：lerobot-record 源码，自我遥操作走自写脚本

---

## 6. 核心录制脚本结构

```python
# act/record_self_teleop.py 主流程伪代码
robot = XLerobot(config) ; robot.connect()        # 内部 configure() 会启用所有扭矩
robot.bus1.disable_torque(robot.left_arm_motors)  # 仅松左臂 6 个电机

dataset = LeRobotDataset.create(repo_id, fps, features=...)  # 自定义 features
events = init_keyboard_listener()                  # 监听 → ← Esc
init_rerun(...)                                    # 边采边看

for episode in range(num_episodes):
    while not events["exit_early"] and elapsed < episode_time_s:
        obs = robot.get_observation()              # 含左右臂 + 头 + 相机

        left_pos    = {k: v for k, v in obs.items() if k.startswith("left_arm_") and k.endswith(".pos")}
        right_target = mirror_left_to_right(left_pos)
        head_hold   = {k: v for k, v in obs.items() if k.startswith("head_motor_")}

        robot.send_action({**right_target, **head_hold})  # 仅控右臂 + 锁头部

        # 录入数据集（仅右臂 state + 头 state + top + right_wrist + 右臂 action）
        frame = build_frame(obs, action=right_target)
        dataset.add_frame(frame)

        log_rerun_data(obs, right_target)
        precise_sleep(1 / fps)

    if events["rerecord_episode"]: dataset.clear_episode_buffer()
    else:                          dataset.save_episode()

dataset.finalize()
```

预估代码量：**150–200 行**（大量复用 lerobot 工具函数）。

---

## 7. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 左臂关扭矩瞬间下垂 | 撞桌 / 撞人 | 启动前**用手扶住**；起始位选立直姿态；md 反复强调 |
| 镜像表某关节方向反 | 右臂动作错误 | `python -m act.mirror probe` 逐关节实测；md 写明首次必跑 |
| `connect()` 启动全扭矩 | 左臂被锁住推不动 | **先 connect 再 disable_torque**（脚本内置正确顺序） |
| 自写 record 漏掉 lerobot-record 的视频编码 | 数据集体积大 / 易爆内存 | 直接用 `LeRobotDataset.create(use_video=True)` + `VideoEncodingManager` 上下文 |
| state/action 维度对不上策略期望 | 训练或推理报错 | 数据集 features 显式定义；`lerobot-edit-dataset --operation.type=info` 验证 |
| 头部电机被误关扭矩 | 视野漂移 | 严格用 `disable_torque(left_arm_motors)`，名称列表精确 |

---

## 8. 验收标准

完工后必须通过：

- [ ] `python -m act.config` 打印 2 路 USB + 2/3 相机配置
- [ ] 5 个脚本 `--help` 全部正常显示
- [ ] `python -c "from act.record_self_teleop import main; from act.mirror import mirror_left_to_right"` 不报错
- [ ] `docs/act/` 下所有 md grep 不到 `bi_xlerobot_leader / leader_left_port / 4 路 USB / 双主臂` 等旧词
- [ ] README 角色表清晰区分"左推 / 右跟"

---

## 9. 不做的事（明确边界）

- ❌ 不删除 `bi_xlerobot_leader` 模块（保留作未来扩展点）
- ❌ 不实现 `send_feedback`（力反馈，与本任务无关）
- ❌ 不录左臂数据（即使将来训双臂协同也不在本任务范围）
- ❌ 不录底盘数据
- ❌ 不改 `lerobot-record` 源码

---

## 10. 后续扩展（不在本次实施）

- 双臂协同采集：左臂也录入，需要左腕相机参与，action 维度变 12，head 也录
- 头部跟随策略：让头部舵机视觉跟随抓取手（VLM 反馈）
- 力反馈：实现 `send_feedback`，让操作员推动左臂时感受到右臂受阻
- 真机部署成功率自动统计：脚本根据按键统计 `→` / `←` 比例
