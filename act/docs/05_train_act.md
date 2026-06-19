# 5️⃣ ACT 训练

> 用刚采集的双臂数据集训练 ACT 策略。
> 模型学的是：**输入相机 + state（6 维 = 右臂6）→ 输出 action（6 维 = 右臂目标）**。

---

## 🎯 一、最简训练命令

```bash
python act/train_act.py
```

或参数覆盖：
```bash
python act/train_act.py --steps=200000 --batch-size=16
python act/train_act.py --resume                       # 断点续训
python act/train_act.py --wandb                        # 启用 wandb
```

或直接用 lerobot CLI（脚本就是这个的薄包装）：
```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=${HF_USER}/xlerobot_act_self_teleop \
  --dataset.root=dataset/${HF_USER}/xlerobot_act_self_teleop \
  --output_dir=outputs/train/act_xlerobot_self_teleop \
  --job_name=act_xlerobot_self_teleop \
  --policy.device=cuda \
  --batch_size=8 \
  --steps=100000 \
  --wandb.enable=false
```

---

## 📋 二、关键参数

| 参数 | 推荐 | 说明 |
|---|---|---|
| `--policy.type=act` | `act` | 用 ACT 策略 |
| `--dataset.repo_id` | 你的数据集 ID | 本地或 HF 都行 |
| `--output_dir` | `outputs/train/xxx` | checkpoint 输出目录 |
| `--job_name` | `xxx` | 本次训练名（日志/wandb 用） |
| `--policy.device` | `cuda` | GPU |
| `--batch_size` | 8（24G）/ 16（48G+） | 看显存 |
| `--steps` | 100000–200000 | ACT 一般 100k 起 |
| `--save_freq` | 10000 | 每多少步存一次 |

### Wandb（强烈建议）
```bash
--wandb.enable=true \
--wandb.project=lerobot_xlerobot \
--wandb.entity=你的wandb用户名
```

---

## 🎛️ 三、ACT 关键超参

```bash
--policy.chunk_size=100 \
--policy.n_action_steps=100 \
--policy.vision_backbone=resnet18 \
--policy.dim_model=512 \
--policy.use_vae=true
```

| 参数 | 默认 | 调整 |
|---|---|---|
| `chunk_size` | 100 | 一次预测多少步动作（30fps × 3.3s） |
| `n_action_steps` | 100 | 实际下发多少步 |
| `vision_backbone` | `resnet18` | 显存紧用 18，宽裕用 50 |
| `dim_model` | 512 | 模型宽度 |
| `use_vae` | true | ACT 论文是开的，建议保持 |

---

## 🖥️ 四、显存 / 速度参考

| GPU | batch_size | 步速（it/s） | 100k step 耗时 |
|---|---|---|---|
| RTX 3090 (24G) | 8 | ~3 | ~9 小时 |
| RTX 4090 (24G) | 8 | ~5 | ~6 小时 |
| A6000 (48G) | 16 | ~6 | ~5 小时 |
| A100 (80G) | 32 | ~10 | ~3 小时 |

> 💡 第一次训练建议先跑 10k step (`--steps=10000`)，看 loss 曲线是否正常下降，再正式开。

---

## 📈 五、训练监控

### Loss 看什么
- `loss` 总损失：稳步下降
- `l1_loss`：动作 L1 误差，应降到 ~0.01
- `kld_loss`（VAE）：先升后降到稳定

### 不收敛的常见原因
1. 数据脏（回 [04_visualize_dataset.md](04_visualize_dataset.md) 删坏段）
2. 数据量太少（< 30 段 ACT 容易欠拟合）
3. 学习率太大（`--optimizer.lr=1e-5`）
4. action 量纲不对（XLeRobot `use_degrees=False`，action 是 ±100 归一化值）

---

## 💾 六、Checkpoint 位置

```
outputs/train/act_xlerobot_self_teleop/
├── checkpoints/
│   ├── 010000/
│   ├── 020000/
│   └── last/         👈 最新
├── train_config.json
└── log/
```

eval 默认从 `checkpoints/last/pretrained_model` 加载。

---

## 🔄 七、断点续训

```bash
python act/train_act.py --resume
```

或显式：
```bash
lerobot-train ... --resume=true --output_dir=outputs/train/act_xlerobot_self_teleop
```

---

## ☁️ 八、推送模型到 HF（可选）

```bash
huggingface-cli upload \
  ${HF_USER}/act_xlerobot_self_teleop \
  outputs/train/act_xlerobot_self_teleop/checkpoints/last \
  --repo-type=model
```

---

## ➡️ 训完进入下一步

[06_replay_and_eval.md](06_replay_and_eval.md)：真机部署，**仅控右臂**。
