"""act —— XLeRobot 自我遥操作 ACT 工作流。

模式：人手推动**左臂**（关扭矩）→ **右臂**实时跟随（开扭矩）→ 头部锁定。
仅 1 台 XLeRobot，2 路 USB（port1+port2），不需要额外主臂硬件。

模块清单：
- config                集中配置（端口/相机/超参）
- mirror                左→右臂关节角镜像表 + probe 子命令
- calibrate_xlerobot    ACT 左右臂标定（默认不标头部/底盘）
- teleoperate_self      自我遥操作（不录数据，调试用）
- record_self_teleop    数据采集（自写主循环）
- train_act             ACT 训练
- eval_act              真机推理（policy 仅控右臂）

文档见：docs/act/
"""
