# 4️⃣ 数据集回看与编辑

> 录完不是结束。**先把数据看一遍**，剔除坏样本再训练。

---

## 🎬 一、本地可视化（Rerun 模式）

```bash
lerobot-dataset-viz \
  --repo-id=${HF_USER}/xlerobot_act_self_teleop \
  --root=dataset/${HF_USER}/xlerobot_act_self_teleop \
  --episode-index=0
```

> ⚠️ `--episode-index` 必填（一次看一段）。换段就改这个数字。

会弹 Rerun 窗口，可以：
- 同步看两路相机 + action + observation
- 检查时间戳对齐

---

## 🌐 二、Web 版可视化

```bash
lerobot-dataset-viz \
  --repo-id=${HF_USER}/xlerobot_act_self_teleop \
  --root=dataset/${HF_USER}/xlerobot_act_self_teleop \
  --episode-index=0 \
  --mode=html
```

会启动本地 web 服务，浏览器打开提示的 URL。
适合远程机或不想装 Rerun GUI 的场景。

---

## 🔍 三、查看数据集元信息

```bash
lerobot-edit-dataset \
  --repo-id=${HF_USER}/xlerobot_act_self_teleop \
  --root=dataset/${HF_USER}/xlerobot_act_self_teleop \
  --operation.type=info \
  --operation.show_features=true
```

输出应该包含：
- `observation.state` shape (6,)
- `action` shape (6,)
- `observation.images.top` / `observation.images.right_wrist`
- 总 episode 数 / 总帧数 / 任务文本

> 💡 `lerobot-info` 是查 lerobot 版本和环境信息（CUDA / PyTorch），**不是查数据集**，别混淆。

---

## ✂️ 四、删除坏 episode

如果发现某几段录砸了（比如 3、7、12 段）：

```bash
lerobot-edit-dataset \
  --repo-id=${HF_USER}/xlerobot_act_self_teleop \
  --operation.type=delete_episodes \
  --operation.episode_indices='[3,7,12]'
```

> ⚠️ `--operation.episode_indices` 是 list 写法，要带方括号和单引号。

### 其它 `--operation.type`
- `split` — 按比例拆分（用 `--operation.splits`）
- `merge` — 合并多个数据集（用 `--operation.repo_ids`）
- `remove_feature` — 删某个 feature
- `modify_tasks` — 改任务描述
- `info` — 看元信息

更多用法：
```bash
lerobot-edit-dataset --help
```

---

## ✅ 五、训练前自检清单

进训练前确保：

- [ ] 所有 episode 都过了一眼
- [ ] 删了所有失败 / 抓空 / 物体未抓住的段
- [ ] action shape = (6,)，state shape = (6,)
- [ ] 两路视频都正常播放
- [ ] 总帧数 ≈ `num_episodes × episode_time_s × 30 fps`
- [ ] action 数值范围合理（XLeRobot 用 RANGE_M100_100 归一化，约在 ±100 之间）

---

## ☁️ 六、上传到 HuggingFace（可选）

确认数据干净后再传：
```bash
huggingface-cli upload \
  ${HF_USER}/xlerobot_act_self_teleop \
  dataset/${HF_USER}/xlerobot_act_self_teleop \
  --repo-type=dataset
```

或在录制时直接传：
```bash
python act/record_self_teleop.py --push-to-hub
```

---

## ➡️ 数据干净后进入下一步

[05_train_act.md](05_train_act.md)：开训！
