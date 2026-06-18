# 1️⃣ XLeRobot 整机标定

> 一次性标定**双臂 + 头部**共 14 个电机。
> 第一次接入硬件、动过零位、换电机后必做。

---

## 🎯 标定的目的

让每个电机知道：
- **零位**（home pose）：所有关节角 = 0 时是什么物理姿态
- **行程**（range）：每个关节能走的最小/最大角度

不标定 → 主从方向反、行程对不上、归一化失效，必出问题。

---

## 📁 标定文件位置

XLeRobot 自带"整机标定"流程，会把双臂 + 头部全部标完，结果存在：
```
~/.cache/huggingface/lerobot/calibration/robots/xlerobot/<follower_id>.json
```

`<follower_id>` 默认值是 `xlerobot_main`（在 [act/config.py](../../act/config.py) 里设置）。

---

## 🛠️ 二、运行标定

```bash
python act/calibrate_xlerobot.py
```

或直接用 lerobot CLI：
```bash
lerobot-calibrate \
  --robot.type=xlerobot \
  --robot.port1=/dev/ttyACM0 \
  --robot.port2=/dev/ttyACM1 \
  --robot.id=xlerobot_main
```

### 流程（终端会顺序提示）

XLeRobot 把电机分两个总线分别标，**头部跟左臂绑死必须一起标**（XLeRobot 类要求，跳不掉），但这只是一次性的标定动作 —— 后续采集和推理时头部不参与控制、不录入数据集。

**bus1（左臂 + 头部，共 8 个电机）**
1. 把**左臂 + 头部**所有关节摆到行程中段 → 回车
2. 把**左臂 + 头部**所有关节挨个推到极限位 → 回车

**bus2（右臂，共 6 个电机；底盘自动跳过）**
3. 把**右臂**所有关节摆到行程中段 → 回车
4. 把**右臂**所有关节挨个推到极限位 → 回车

> ✅ **底盘 3 个轮子电机不需要你做任何事** —— XLeRobot 标定脚本自动给它们填默认值（`range_min=0, range_max=4095`，整圈电机标准做法）。
> ⚠️ 标定时电机扭矩会被自动关掉，所以推动很轻松；推完后系统会自动把扭矩拉回来。

---

## ✅ 三、验证

```bash
ls ~/.cache/huggingface/lerobot/calibration/robots/xlerobot/
# 应该看到：xlerobot_main.json
```

打开看一眼：每个电机都应该有 `homing_offset / range_min / range_max` 三个字段。

---

## ♻️ 四、什么时候要重标

| 场景 | 是否重标 |
|---|---|
| 第一次接入 | ✅ |
| 换电机 / 拆装关节 | ✅ |
| 改了 `--robot.id` / `FOLLOWER_ID` | ✅（新名字对应新文件） |
| 长时间没用、怀疑零位漂 | 建议 |
| 只是换 USB 端口 | ❌ |

---

## ⚠️ 五、常见标定坑

| 现象 | 原因 | 解决 |
|---|---|---|
| 关节卡住推不动 | 扭矩没松 / 物理障碍 | 断电再上电；移开障碍 |
| 标完跟随方向反 | 极限位推反了 | 重标，注意先推到正方向再回零 |
| 标完行程缩了一半 | 没推到真极限 | 重标，听到电机微响再停 |
| 启动时被问 ENTER 还是 c | XLeRobot 询问是否复用旧标定 | 直接回车 = 用旧标定；输入 `c` 回车 = 重标 |

---

## ➡️ 标完进入下一步

[02_teleoperate.md](02_teleoperate.md)：自我遥操作验证 + 镜像表 probe。
