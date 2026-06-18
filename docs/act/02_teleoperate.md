# 2️⃣ 自我遥操作验证 + 镜像表 probe

> ⭐ **采集前必跑**。验证：左臂能推、右臂能跟、头部不动、镜像方向对。
> 跟随不丝滑就别去录数据，不然全是垃圾样本。

---

## 🚀 一、启动遥操作（验证模式，不录数据）

```bash
python act/teleoperate_self.py
```

可选参数：
```bash
python act/teleoperate_self.py --no-display   # 不开 Rerun
python act/teleoperate_self.py --no-cameras   # 不连相机（更快启动）
python act/teleoperate_self.py --hz 30        # 控制频率
```

启动后，脚本会：
1. 连 XLeRobot
2. **关闭左臂 6 个电机的扭矩**（人手就能推动了）
3. 锁定头部当前位置
4. 进入主循环：读左臂 → 镜像 → 写右臂

---

## ⚠️ 二、操作员安全须知（必读）

1. **关扭矩瞬间左臂会因重力下垂** —— 启动前**先用手扶住**左臂！
2. **起始姿态选"重力影响最小"**：双臂立直、夹爪朝下、远离桌面
3. **不要碰头部 / 底盘电机**（它们扭矩没关，强行推可能损坏齿轮）
4. **现场要预留缓冲空间**（左臂下垂可能撞到桌沿或物体）

---

## ✅ 三、验证清单（按顺序检查）

### ✓ 1. 左臂可被推动
轻推左臂任一关节，应该**毫无阻力**地动。
> ❌ 推不动 → 扭矩没关，看终端日志有没有 `左臂扭矩已关闭`。

### ✓ 2. 右臂跟随左臂
推左臂 shoulder_pan，右臂 shoulder_pan **应该跟着动**。
> ❌ 不跟 → 检查 `--robot.port2` 是否正确接到右臂总线。

### ✓ 3. 各关节方向对
逐个关节推，验证右臂同向。**典型可能反向**：
- `shoulder_pan`（左右肩转）
- `wrist_roll`（腕部翻转）
- `gripper`（夹爪开合 —— 这个比较常见反）

> ❌ 反了 → 见下一节"调镜像表"。

### ✓ 4. 头部纹丝不动
- 头部电机扭矩开着、自然伺服锁位 → 物理上应该不动
- 如果飘了说明 `disable_torque(left_arm_motors)` 的参数错误，误关到了头部

### ✓ 5. Rerun 曲线对齐
窗口里看：
- `action.right_arm_*.pos`（下发目标）
- `observation.right_arm_*.pos`（实测位置）

两条线**几乎重合，从臂滞后 < 1 帧**。

### ✓ 6. 帧率稳定 30 Hz
终端循环消耗的时间应该稳定 ≤ 33ms。

---

## 🔍 四、调镜像表（关键步骤）

如果第 3 项发现某些关节方向反，跑 probe 工具自动帮你找出哪些关节要取反：

```bash
python -m act.mirror probe
```

流程：
1. 连机器人，关左臂扭矩
2. 锁右臂 + 头部当前位置
3. 逐关节提示你推一下，观察右臂方向
4. 你按 `y`（同向）/ `n`（反向）/ `s`（跳过）
5. 全部测完后输出建议的 `NEGATE_JOINTS` 集合

例如得到 `{shoulder_pan, wrist_roll, gripper}`，应用方式：

**临时（环境变量）**
```bash
export NEGATE_JOINTS="shoulder_pan,wrist_roll,gripper"
python act/teleoperate_self.py     # 验证
```

**永久（改代码）**
编辑 [act/mirror.py](../../act/mirror.py) 里的 `DEFAULT_NEGATE_JOINTS`：
```python
DEFAULT_NEGATE_JOINTS: frozenset[str] = frozenset({"shoulder_pan", "wrist_roll", "gripper"})
```

### 查看当前镜像表配置
```bash
python -m act.mirror show
```

---

## 🔧 五、其它常见排查

| 现象 | 可能原因 | 解决 |
|---|---|---|
| 启动时报 `Could not open port` | 端口名错 / 占用 / 没权限 | `ls /dev/ttyACM*`；`sudo usermod -aG dialout $USER` 后重登 |
| 左臂可推但右臂跟得很迟 | bus2 USB 带宽不够 | bus1 / bus2 接到不同 USB 控制器 |
| 右臂某些关节抖 | 镜像后某关节出了行程范围 | `python -m act.mirror probe`；或起始位摆远离极限位 |
| 启动时 ENTER 等待 | XLeRobot 询问是否复用旧标定 | 直接回车 |

---

## 🎮 六、退出

`Ctrl+C` —— 脚本会安全断开机器人；左臂扭矩重新被启用前请扶住臂避免突跳。

---

## ➡️ 跟随丝滑后进入下一步

[03_record.md](03_record.md)：开录数据集！
