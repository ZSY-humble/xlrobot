# 0️⃣ 环境与硬件准备

> 自我遥操作模式只需要 **1 台 XLeRobot**，2 路 USB，2 路相机。
> 5 分钟搞定基础准备。

---

## 🐍 一、Python 环境

```bash
conda activate lerobot
which lerobot-calibrate           # 验证 lerobot 已可执行
```

如果模块加载有问题：
```bash
cd /home/zsy/project/lerobot
pip install -e .
```

---

## 🔌 二、串口端口（仅 2 路）

| 角色 | 默认端口 | XLeRobot 字段 | 说明 |
|---|---|---|---|
| **bus1** | `/dev/ttyACM0` | `--robot.port1` | **左臂 6 个电机 + 头部 2 个电机** |
| **bus2** | `/dev/ttyACM1` | `--robot.port2` | **右臂 6 个电机 + 底盘 3 个轮子** |

> 💡 你只需要把 XLeRobot 自身的两根总线插上电脑，**不需要任何额外的主臂 USB**。

### 查端口
```bash
ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null
dmesg | tail -10        # 拔/插某根 USB 后再看
```

### 自动识别哪根 USB 是哪个总线
```bash
lerobot-find-port
```
脚本会让你按提示拔/插某根 USB，自动告诉你它对应 `/dev/ttyACMx`。

### 推荐：用 udev 起固定别名
```bash
udevadm info -a -n /dev/ttyACM0 | grep -E "idVendor|idProduct|serial" | head -3
```
建立 `/etc/udev/rules.d/99-lerobot.rules`：
```
SUBSYSTEM=="tty", ATTRS{serial}=="XXXX_BUS1_SN", SYMLINK+="xlerobot_bus1"
SUBSYSTEM=="tty", ATTRS{serial}=="XXXX_BUS2_SN", SYMLINK+="xlerobot_bus2"
```
重载：
```bash
sudo udevadm control --reload && sudo udevadm trigger
```

---

## 📹 三、相机配置

### 录入数据集的相机（必备 2 路）

| 相机 | 位置 | 命名 | 看什么 |
|---|---|---|---|
| 桌面 | 桌沿 / 三脚架 | `top` | 任务全局场景 |
| 右腕 | 右臂末端（夹爪附近） | `right_wrist` | 右臂抓取近景 |

### 可选：左腕相机
- 装了也没问题（现场监控操作方便）
- **不录入数据集** —— 因为左臂只是动作输入，部署时根本没有左臂动

### 检查相机
```bash
ls /dev/video*
# 单独验每一路（按 q 退出）
ffplay -f v4l2 -i /dev/video0
ffplay -f v4l2 -i /dev/video2
```

```bash
lerobot-find-cameras opencv      # 列所有 OpenCV 可见相机
```

> ⚠️ Linux 下 `index_or_path` 写**完整路径** `/dev/video0`，不要写 `0`，避免热插拔后错位。

---

## 🤗 四、HuggingFace 用户名

```bash
huggingface-cli login                          # 第一次用要登录
export HF_USER=$(huggingface-cli whoami | head -n 1)
echo $HF_USER                                   # 确认有输出
```

> 💡 把 `export HF_USER=...` 写到 `~/.bashrc` 里，省得每次都要 source。

---

## ⚙️ 五、检查 act/config.py 配置

```bash
python -m act.config
```

输出里要重点看：
- `XLeRobot port1` / `port2` 是否对应你实际的 bus1 / bus2
- 两路相机路径是否对
- `分辨率/帧率` 默认 640×480 @ 30fps

如要改默认值，编辑 [../../act/config.py](../../act/config.py)；
如要临时覆盖，用环境变量：
```bash
FOLLOWER_PORT1=/dev/xlerobot_bus1 \
FOLLOWER_PORT2=/dev/xlerobot_bus2 \
CAM_TOP=/dev/video4 \
python act/teleoperate_self.py
```

---

## ✅ 六、最后自检

```bash
conda activate lerobot && lerobot-calibrate --help > /dev/null && echo "✅ env OK"
ls /dev/ttyACM* | wc -l
ls /dev/video* | wc -l
[ -n "$HF_USER" ] && echo "✅ HF_USER=$HF_USER"
python -m act.config
```

全部 ✅ 后进入 → [01_calibrate.md](01_calibrate.md)
