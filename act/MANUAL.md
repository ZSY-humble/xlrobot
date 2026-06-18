# 🎯 XLeRobot 自我遥操作 ACT 使用手册

一句话：这套 `act/` 代码用于 **XLeRobot 左臂手推示教，右臂实时跟随并采集 ACT 数据**，不需要额外 SO101 主臂。

---

## 1. 当前代码是否正确

当前 `act/` 目录的主线是正确的：

- `act/calibrate_xlerobot.py`：ACT 左右臂标定，默认不要求手动标头部/底盘。
- `act/teleoperate_self.py`：自我遥操作验证，左臂关扭矩，人手推动，右臂跟随。
- `act/mirror.py`：左臂到右臂的镜像映射，支持 `probe` 实测哪些关节需要取反。
- `act/record_self_teleop.py`：正式采集，边左推右跟边录数据。
- `act/train_act.py`：训练 ACT。
- `act/eval_act.py`：加载训练好的 ACT 做真机推理，目标是只控制右臂。

已经做过代码级检查：

```bash
conda run -n lerobot python -m py_compile \
  act/record_self_teleop.py \
  act/teleoperate_self.py \
  act/mirror.py \
  act/config.py \
  act/train_act.py
```

也检查过采集 schema：

```text
observation.state: (6,)  右臂 6 关节
action:            (6,)  右臂 6 关节目标
images:            top + right_wrist
```

⚠️ 说明：代码级检查通过不等于硬件一定无误。真实采集前还必须在你的机器上跑一次 `teleoperate_self.py`，确认端口、标定、镜像方向、相机设备号都对。

---

## 2. 工作流总览

物理逻辑：

```text
人手推动 XLeRobot 左臂
        ↓
读取 left_arm_*.pos
        ↓
act/mirror.py 做左右镜像
        ↓
下发 right_arm_*.pos 给右臂
        ↓
录入 right_arm state/action + 相机
        ↓
训练 ACT 策略
```

数据集只训练右臂：

```text
左臂：只作为示教输入，不录进 action
右臂：执行臂，录 state 和 action
头部：保持扭矩锁位，不录
底盘：不用，不录
```

---

## 3. 环境准备

进入项目并激活环境：

```bash
cd /home/zsy/project/lerobot
conda activate lerobot
```

设置 HuggingFace 用户名（可选）：

```bash
export HF_USER=$(huggingface-cli whoami | head -n 1)
echo $HF_USER
```

不设置 `HF_USER` 也可以本地采集，默认数据集 ID 会是：

```text
local/xlerobot_act_self_teleop
```

只有使用 `--push-to-hub` 上传 HuggingFace Hub 时，才必须设置真实 `HF_USER`。

默认配置在 [config.py](/home/zsy/project/lerobot/act/config.py)。

常用环境变量：

```bash
export FOLLOWER_PORT1=/dev/ttyACM0
export FOLLOWER_PORT2=/dev/ttyACM1
export FOLLOWER_ID=xlerobot_main

export CAM_TOP=/dev/video3
export CAM_RIGHT_WRIST=/dev/video4
export CAM_W=640
export CAM_H=480
export CAM_FPS=30

export DATASET_NAME=xlerobot_act_self_teleop
export TASK_DESC="Pick up the red cube and place it into the box"
```

查看当前配置：

```bash
python -m act.config
```

---

## 4. 双臂标定

有，入口是：

```bash
python act/calibrate_xlerobot.py
```

这里的“双臂标定”不是两台外部 SO101 主臂标定，而是 XLeRobot 本体左右臂标定：

```text
port1: 左臂 6 电机 + 头部 2 电机
port2: 右臂 6 电机 + 两轮底盘 2 电机（ID 9、10）
```

默认只要求你手动标定：

```text
left_arm_*  6 个电机
right_arm_* 6 个电机
```

头部和底盘不参与 ACT 自我遥操作采集。脚本会保留已有头部/底盘标定，
或读取电机当前参数补齐 JSON，避免后续连接时反复触发整机标定。

非连续关节的范围记录会被限制在“中位附近的连续编码窗口”内，默认：

```bash
python act/calibrate_xlerobot.py --edge-guard 128 --center-window 1700
```

如果表格里 `POS` 后面出现 `*`，或 `IGN` 计数增加，表示该样本靠近或跨过
`0/4095` 回绕边界，已经被忽略，不会写进 `range_min/range_max`。

如果确实要走 LeRobot 官方整机标定，再显式加 `--full-body`：

```bash
python act/calibrate_xlerobot.py --full-body
```

标定文件保存位置：

```text
~/.cache/huggingface/lerobot/calibration/robots/xlerobot_2wheels/<FOLLOWER_ID>.json
```

默认是：

```text
~/.cache/huggingface/lerobot/calibration/robots/xlerobot_2wheels/xlerobot_main.json
```

首次接入、换过电机、重装结构、关节方向不对时都要重新标定。

---

## 5. 镜像表检查

先看当前取反表：

```bash
python -m act.mirror show
```

当前这台 XLeRobot 已实测确认：delta 跟随时不需要额外取反，默认应为：

```text
DEFAULT_NEGATE_JOINTS = frozenset()
NEGATE_JOINTS 未设置时，实际生效 = set()
```

如果左推右跟方向不对，跑交互 probe：

```bash
python -m act.mirror probe
```

临时覆盖取反表：

```bash
export NEGATE_JOINTS=""
python act/teleoperate_self.py
```

永久修改位置：

```text
act/mirror.py
DEFAULT_NEGATE_JOINTS
```

---

## 6. 自我遥操作验证

正式采集前必须先跑：

```bash
python act/teleoperate_self.py
```

默认行为：

```text
1. 连接 XLeRobot
2. 关闭左臂 6 个电机扭矩
3. 人手推动左臂
4. 程序默认把“左臂相对启动姿态的变化量”镜像成右臂目标
5. 右臂实时跟随
6. Rerun 显示相机、action、observation
```

默认是 delta 变化量跟随：

```text
right_target = right_start + mirror(left_current - left_start)
```

这样不会因为左右臂启动时绝对值不完全一致而突然追远处目标。
如果确实要用绝对位置镜像，显式加 `--absolute`。

如果不想开 Rerun：

```bash
python act/teleoperate_self.py --no-display
```

如果只验证关节、不连相机：

```bash
python act/teleoperate_self.py --no-cameras --no-display --max-relative-target 5.0
```

调控制频率：

```bash
python act/teleoperate_self.py --hz 30
```

安全注意：

- 启动前先用手扶住左臂。
- 脚本会关闭左臂扭矩，左臂可能因重力下垂。
- 右臂、头部保持上电，不要强行掰。
- 第一次真机验证建议使用 `--max-relative-target 5.0`。

---

## 7. 正式采集

基础命令：

```bash
python act/record_self_teleop.py
```

正式采集默认也使用 delta 变化量跟随；采集到的 `action` 是实际下发给右臂的目标。
如果确实要采绝对位置镜像，加 `--absolute`。

常用命令：

```bash
python act/record_self_teleop.py \
  --num-episodes 50 \
  --episode-time 20 \
  --reset-time 10 \
  --task "Pick up the red cube and place it into the box"
```

续采：

```bash
python act/record_self_teleop.py --resume
```

不显示 Rerun：

```bash
python act/record_self_teleop.py --no-display
```

指定本地数据集目录：

```bash
python act/record_self_teleop.py \
  --root ~/.cache/huggingface/lerobot/${HF_USER}/xlerobot_act_self_teleop
```

采完上传 Hub：

```bash
python act/record_self_teleop.py --push-to-hub
```

如果你担心本地磁盘占用，可以先采集并上传到 Hub：

```bash
export HF_USER=$(huggingface-cli whoami | head -n 1)
python act/record_self_teleop.py --push-to-hub
```

注意：实时采集阶段仍然会先在本地写一份数据集，再执行上传；上传成功后可以再清理本地缓存。

采集按键：

| 按键 | 作用 |
|---|---|
| 右箭头 | 当前 episode 成功，提前结束并保存 |
| 左箭头 | 当前 episode 失败，丢弃并重录 |
| Esc | 停止整体采集 |

⚠️ 按键监听需要本地物理终端。SSH/headless 环境下可能收不到箭头键。

---

## 8. 采集时可视化

有。`act/record_self_teleop.py` 默认会启动 Rerun，可边采边看：

```text
observation.images.top
observation.images.right_wrist
observation.state.right_arm_*.pos
action.right_arm_*.pos
```

关闭显示：

```bash
python act/record_self_teleop.py --no-display
```

建议采集时重点看：

- `top` 是否能看到完整桌面和目标物。
- `right_wrist` 是否能看到夹爪和被抓物。
- `action.right_arm_*.pos` 与右臂实际状态趋势是否一致。
- 终端是否有低频、掉帧、相机读取失败提示。

---

## 9. 采完后回看数据集

默认本地路径：

```text
~/.cache/huggingface/lerobot/local/xlerobot_act_self_teleop/
```

用 LeRobot 可视化工具回看：

```bash
lerobot-dataset-viz \
  --repo-id local/xlerobot_act_self_teleop \
  --episode-index 0
```

检查点：

- `action` shape 应该是 `(6,)`
- `observation.state` shape 应该是 `(6,)`
- 视频应该有 `top` 和 `right_wrist`
- 每段任务是否完整、稳定、无明显误操作

---

## 10. 训练 ACT

基础训练：

```bash
python act/train_act.py
```

如果数据集已经上传到 HuggingFace Hub，并且想减少本地缓存，使用流式训练：

```bash
export HF_USER=$(huggingface-cli whoami | head -n 1)
python act/train_act.py --streaming
```

这会在底层给 `lerobot-train` 传入：

```bash
--dataset.streaming=true
```

指定步数和 batch：

```bash
python act/train_act.py --steps 100000 --batch-size 8
```

指定设备：

```bash
python act/train_act.py --device cuda
```

断点续训：

```bash
python act/train_act.py --resume
```

输出目录：

```text
outputs/train/act_xlerobot_self_teleop/
```

最终策略默认在：

```text
outputs/train/act_xlerobot_self_teleop/checkpoints/last/pretrained_model
```

---

## 11. 真机推理

训练完成后：

```bash
python act/eval_act.py
```

指定 policy：

```bash
python act/eval_act.py \
  --policy outputs/train/act_xlerobot_self_teleop/checkpoints/last/pretrained_model
```

指定评估段数：

```bash
python act/eval_act.py --num-episodes 10 --episode-time 30
```

推理时不需要再推动左臂。模型输出右臂 6 维 action，理论上只控制右臂。

⚠️ 这一步建议先短时、低风险测试，因为它涉及模型推理、action 后处理和真机执行。

---

## 12. 推荐现场顺序

第一次跑：

```bash
conda activate lerobot

export FOLLOWER_PORT1=/dev/ttyACM0
export FOLLOWER_PORT2=/dev/ttyACM1
export CAM_TOP=/dev/video3
export CAM_RIGHT_WRIST=/dev/video4

python -m act.config
python act/calibrate_xlerobot.py
python -m act.mirror show
python act/teleoperate_self.py --no-cameras --no-display --max-relative-target 5.0
python act/teleoperate_self.py
python act/record_self_teleop.py --num-episodes 3 --episode-time 10
lerobot-dataset-viz --repo-id local/xlerobot_act_self_teleop --episode-index 0
python act/record_self_teleop.py --num-episodes 50 --episode-time 20
python act/train_act.py
```

日常采集：

```bash
conda activate lerobot
python act/teleoperate_self.py
python act/record_self_teleop.py --resume
```

---

## 13. 常见问题

### 左推右跟方向反了

先跑：

```bash
python -m act.mirror probe
```

再调整：

```bash
export NEGATE_JOINTS=""
```

### 左臂推不动

确认 `teleoperate_self.py` 或 `record_self_teleop.py` 启动后有日志：

```text
左臂扭矩关闭
```

如果没有，检查 `FOLLOWER_PORT1` 是否接对。

### 相机黑屏

检查设备：

```bash
ls /dev/video*
python act/check_cameras.py
```

如果只想看你当前最可能有画面的几个端口：

```bash
python act/check_cameras.py --devices /dev/video3 /dev/video4
```

必要时改：

```bash
export CAM_TOP=/dev/videoX
export CAM_RIGHT_WRIST=/dev/videoY
```

### 续采失败

确保使用的是同一个数据集目录。未设置 `HF_USER` 时：

```bash
python act/record_self_teleop.py --resume \
  --root ~/.cache/huggingface/lerobot/local/xlerobot_act_self_teleop
```

如果你设置了 `HF_USER`，则是：

```bash
python act/record_self_teleop.py --resume \
  --root ~/.cache/huggingface/lerobot/${HF_USER}/xlerobot_act_self_teleop
```

### 训练启动报找不到数据集

确认：

```bash
python -m act.config
ls ~/.cache/huggingface/lerobot/local/xlerobot_act_self_teleop
```

---

## 14. 文件索引

- [config.py](/home/zsy/project/lerobot/act/config.py)：端口、相机、数据集、训练默认参数
- [calibrate_xlerobot.py](/home/zsy/project/lerobot/act/calibrate_xlerobot.py)：ACT 左右臂标定
- [mirror.py](/home/zsy/project/lerobot/act/mirror.py)：左臂到右臂镜像映射
- [teleoperate_self.py](/home/zsy/project/lerobot/act/teleoperate_self.py)：左推右跟验证
- [record_self_teleop.py](/home/zsy/project/lerobot/act/record_self_teleop.py)：正式数据采集
- [train_act.py](/home/zsy/project/lerobot/act/train_act.py)：ACT 训练
- [eval_act.py](/home/zsy/project/lerobot/act/eval_act.py)：ACT 真机推理
