# XLeRobot ACT Self-Teleop

用一台 XLeRobot 做 ACT 数据采集：左臂关扭矩由人手推动，右臂实时镜像跟随并被记录，训练后的策略只控制右臂。

## 采集内容

- `observation.state`: 右臂 6 个关节
- `action`: 右臂 6 个目标关节位置
- `observation.images.top`: 桌面 / 俯视相机
- `observation.images.right_wrist`: 右腕相机

头部相机和底盘不进入数据集。头部两个电机会在每段采集前后回到固定 `head_home`，用于保持主相机视角一致。

## 快速开始

```bash
conda activate lerobot

# 1. 安装仓库内置标定到 LeRobot 默认 cache
# 标定源文件：act/config/calibration/xlerobot_main.json
python act/install_calibration.py --force

# 2. 检查串口和相机，按终端输出确认真实端口
python act/check_xlerobot_ports.py
python act/check_cameras.py

# 3. 根据上一步结果设置当前机器端口
# 下面只是示例，不要照抄；以 check_xlerobot_ports.py / check_cameras.py 的输出为准
export FOLLOWER_PORT1=/dev/ttyACM3
export FOLLOWER_PORT2=/dev/ttyACM2
export CAM_TOP=/dev/video6
export CAM_RIGHT_WRIST=/dev/video4
export CAM_W=640
export CAM_H=480
export CAM_FPS=30

# 4. 可选：检查当前姿态和仓库内置 reset home 的误差，不移动机器人
# reset home 文件：act/config/reset_home.json
python act/test_reset_home.py --dry-run

# 5. 开始采集；脚本会在每段前后自动让右臂和头部回固定 home
export DATASET_NAME=xlerobot_act_pick_place_50
python act/record_self_teleop.py \
  --num-episodes 50 \
  --task "pick up the object and place it into the target container"
```

采集数据默认保存到：

```text
dataset/<HF_USER>/<DATASET_NAME>
```

如果本地数据集已存在，脚本会自动追加 episode。只有明确重录同名数据集时才使用 `--overwrite`。

## 采集热键

| 按键 | 作用 |
| --- | --- |
| `s` 或 `→` | 保存当前 episode |
| `r` 或 `←` | 丢弃当前 episode 并重录 |
| `q` 或 `Esc` | 停止整体采集 |

## 标定与 Home

仓库内置标定文件：

```text
act/config/calibration/xlerobot_main.json
```

安装到本机 cache：

```bash
python act/install_calibration.py --force
```

仓库也内置了当前采集使用的固定复位姿态：

```text
act/config/reset_home.json
```

这个文件包含：

- `right_home`: 每段采集前后右臂回到的固定初始姿态
- `head_home`: 每段采集前后头部相机回到的固定视角

如果只是复现实验，不需要重新标定，也不需要重新采集 home。先安装标定，再用 dry-run 检查误差即可：

```bash
python act/install_calibration.py --force
python act/test_reset_home.py --dry-run
```

如果换了机器人、舵机 ID、装配方向或机械限位，请重新标定：

```bash
python act/calibrate_xlerobot.py
python act/calibrate_head.py
python act/capture_reset_home.py
```

`capture_reset_home.py` 会覆盖 `act/config/reset_home.json`，只在你确实调整了右臂初始姿态或头部相机视角后再运行。

## 常用命令

```bash
# 只验证左推右跟，不录数据
python act/teleoperate_self.py

# 回看数据集
lerobot-dataset-viz \
  --repo-id "$HF_USER/$DATASET_NAME" \
  --root "dataset/$HF_USER/$DATASET_NAME" \
  --episode-index 0

# 训练 ACT
python act/train_act.py

# 真机评估
python act/eval_act.py
```

## 文件索引

- `config.py`: 端口、相机、数据集、训练参数
- `mirror.py`: 左臂到右臂的镜像映射
- `calibrate_xlerobot.py`: 左右臂标定
- `calibrate_head.py`: 头部相机两个电机标定
- `capture_reset_home.py`: 保存右臂和头部 home
- `test_reset_home.py`: 测试固定 home
- `teleoperate_self.py`: 自我遥操作验证
- `record_self_teleop.py`: ACT 数据采集主脚本
- `train_act.py`: ACT 训练入口
- `eval_act.py`: 真机部署评估入口
- `docs/`: 详细手册和排错说明

## 注意

- 左臂会在采集时关闭扭矩，按回车前必须用手扶住。
- `CAM_FPS` 要和相机实际输出一致；当前 640x480 推荐使用 30fps。
- 同一个数据集不要混用不同分辨率、帧率或相机视角。
- 内置标定只适用于当前这台 XLeRobot 或硬件完全一致的机器。
