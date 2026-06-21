# 6️⃣ 训练后回放与真机部署

> 两个用途：
> - **回放（replay）**：把数据集里某段直接发给右臂跑一遍，验证数据无误。
> - **推理部署（eval）**：用训好的 ACT 策略真机控制右臂。
>
> 🌟 关键：**推理时不需要再推左臂**。模型已经学会"看相机 + 看右臂状态 → 输出右臂动作"。

---

## 🔁 一、回放数据集中某段（不用模型）

```bash
lerobot-replay \
  --robot.type=xlerobot_2wheels \
  --robot.port1=/dev/ttyACM0 \
  --robot.port2=/dev/ttyACM1 \
  --robot.id=xlerobot_main \
  --dataset.repo_id=${HF_USER}/xlerobot_act_self_teleop \
  --dataset.root=dataset/${HF_USER}/xlerobot_act_self_teleop \
  --dataset.episode=0
```

### 用途
- 验证右臂能复现采集时的轨迹
- debug：录的时候没问题但回放抖 → 时序问题
- 不需要主臂、不需要 GPU

> ⚠️ 回放时 `lerobot-replay` 会按数据集里 action（6 维右臂）下发；左臂保持当前位置不动。

---

## 🤖 二、ACT 真机推理（核心部署）

### 一键启动
```bash
python act/eval_act_right_arm.py
```

当前默认参数等价于：

```bash
python act/eval_act_right_arm.py \
  --episode-time=0 \
  --fps=30 \
  --max-relative-target=10 \
  --temporal-ensemble-coeff=0.01 \
  --action-smoothing-alpha=0.7
```

常用参数：

```bash
python act/eval_act_right_arm.py --episode-time=30
python act/eval_act_right_arm.py --policy=/abs/path/to/checkpoints/last/pretrained_model
python act/eval_act_right_arm.py --task="新任务描述（必须与训练一致）"
```

### 关键点
- ✅ **不需要主臂**！action 由模型推理给出
- ✅ 模型只输出 `right_arm_*.pos` 6 维
- ✅ 脚本绕开整机 16 维 action schema，只写 `bus2` 的右臂 6 个 `Goal_Position`
- ✅ **左臂保持当前位置不动**（推理时甚至可以把左臂摆到一个安全姿态再启动）
- ⚠️ `--task` 必须和训练时一致（语言条件 ACT 才需要）
- ⚠️ 按 Enter 前只显示 Rerun 相机 / 状态，不发送 action；按 Enter 后才开始执行

---

## 🎮 三、推理时按键（统计成功率）

| 按键 | 作用 |
|---|---|
| `→` | 这段成功 |
| `←` | 这段失败，重试 |
| `Esc` | 停止 |

录完手算成功率：成功段数 / 总段数。

---

## 📊 四、推理性能调优

### 速度
- 当前默认开启 `--temporal-ensemble-coeff=0.01`，每个控制周期都会重新推理并融合 ACT 未来动作。
- 如果 GPU 压力过大、控制周期掉帧，可以先关闭 Rerun，再做对照：

```bash
python act/eval_act_right_arm.py \
  --episode-time=30 \
  --fps=30 \
  --max-relative-target=10 \
  --temporal-ensemble-coeff=0 \
  --action-smoothing-alpha=0.4
```

### 稳定性
- 推理前**人手把右臂摆到训练数据的 home pose**附近
- 物体位置不要超出训练数据覆盖范围
- **相机外参必须与训练时一致**（位置/角度/分辨率）
- 保持 `--fps=30`，和采集帧率一致
- 保持 `--max-relative-target=10`，和采集限幅一致

---

## ⚠️ 五、推理常见坑

| 现象 | 原因 | 解决 |
|---|---|---|
| 右臂抖动 | 可能是 ACT chunk 边界跳变 / 限幅过大 / 输出缺少时间融合 | 用当前默认：`TE=0.01`、`alpha=0.7`、限幅 10 |
| 每隔约 2 秒顿一下 | 当前模型 `chunk_size=60`，30Hz 下一个 chunk 约 2 秒，边界可能不连续 | 开启 `--temporal-ensemble-coeff=0.01` |
| 开启 TE 后变慢 | temporal ensemble 和低通叠加后滞后 | 提高 `--action-smoothing-alpha` 到 0.85 或 1 |
| 总抓空 | 相机外参变了 | 把相机摆回训练时位置 |
| 模型加载报错 | path 错 | 路径应到 `checkpoints/last/pretrained_model` |
| 推理时左臂在动 | 不应该 —— policy 只输出 right_arm_*.pos | 看模型 features，是不是训练数据 schema 错了 |
| 显存爆 | batch=1 应不会，看相机分辨率 | 推理时确认 `--policy.device=cuda` |
| state 维度不对 | 训练时是 6 维（仅右臂 6 关节） | 确认 dataset features 是新的 self_teleop schema |

---

## 🎯 六、成功率统计

跑完 N 段评估后：
```
成功段数 / N = 成功率
```

经验值：
- < 50% → 多采数据 / 多训 step / 简化任务
- 50–80% → 数据增强（更多物体位置 / 干扰物）
- \> 80% → 可以挑战更难任务

---

## ➡️ 出问题翻

[07_troubleshooting.md](07_troubleshooting.md)
