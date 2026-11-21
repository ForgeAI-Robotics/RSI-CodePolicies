# 技术与真机对接

## 技术路线与边界

PyRUA-Lean 用于开发期反馈、修改和回归，最终执行是 Python Code as Policy。策略通过 OpenCV HSV/连通域找红罐，通过 Canny + 圆拟合 + SciPy 鲁棒优化估计罐口 XY；Z 来自桌高和罐高先验，并非深度测量。静态模型 FK/IK、连续局部 IK 和插值输出 16 维绝对动作。视觉检查位于阶段边界，**不是每个控制周期重新感知**，也不是任意障碍物规划器。

`policy.py` 是经仿真验证的原代码；`perception.py` 与原 primitive 字节一致。其余接口、封装和交付测试是本次新增，不能直接继承原生 12/12 结论。

## 真机接口

实现 `tianji_cola/interface.py` 的 `RobotPort`。仅 `action_sequence` 下发运动；其余模型方法不得暗中执行运动。

| 方法 | 返回约定 |
| --- | --- |
| `initial_action()` | 已确认的启动 16D 指令；不自动移动到此姿态 |
| `image(camera)` | `head` / `left_wrist` / `right_wrist` 的去畸变 RGB uint8，480×640×3；不得 BGR；驱动检查时间戳和新鲜度 |
| `action_sequence(rows)` | 严格按 20Hz 依次执行全部行，完成后返回具有 `commanded_action` 属性的对象；失败抛异常 |
| `solve_pose(arm, xyz, rpy, reference)` | `{"joint_target_rad": {"Joint1_L": …, …, "Joint7_L": …}}`；传入姿态为 XYZ Euler **度**，位置为米 |
| `solve_pose_continuous(...)` | 同上；保持邻近分支和局部步长；参考 `reference/tianji/rgb/continuous.py` 的限位与采样机身间距检查 |
| `fk(arm, reference)` | `{"command_fk": 4×4 chassis_base_T_tool0}`；由传入计划指令计算 |
| `camera_model(camera)` | `intrinsic` 3×3 与 `optical_pose_from_command` 4×4（chassis_base_T_optical） |
| `project(camera,row,col,height)` | `{"estimated_xyz": [x,y,z]}`；射线与显式高度平面相交，不可用场景真值替代 |
| `rim(camera,expected_xyz,height,radius)` | `{"fits": circle_rim(rgb,K,T,expected_xyz,height,radius)}`；使用包内 `perception.py` |
| `gripper_geometry(arm)` | 静态模型几何字典；冻结版初始化读取但后续未使用其内容 |
| `safe_stop()` | 厂家控制器停止/保持接口；不得默认张爪掉落物体 |

动作顺序为 `[Joint1_L…Joint7_L, left_gripper, Joint1_R…Joint7_R, right_gripper]`。关节弧度、夹爪归一化开度（0 闭合、1 张开）。右侧 8 通道必须始终与初始值一致。执行层应在发送前校验整批的有限数、关节限位、速度和时序，并在执行期间监督跟踪误差和急停；不要把整批最后一行直接作为目标。

## 对接步骤

1. 确定厂家 SDK 或 ROS 2 传输接口，实现驱动。控制器内部可用编码器、电流、力和急停反馈；策略端仍仅使用 RGB、静态模型与自身指令历史。
2. 对照实际机器人确认关节顺序、方向、零位、限位、tool0、夹爪 TCP 和 FK/IK。`reference/` 保留原代码和依赖路径供移植；不是可直接运行的真机驱动。原 SDK、模型和网格依赖清单及哈希见 `reference/external_files.json`，不在本包重复打包。
3. 完成三路内参、畸变、外参/手眼标定。`config/sample_calibration.json` 来自样本，`sample_wrist_v1` 腕部配准仅是图像地标近似，不能视作独立真机手眼标定。
4. 确认静态先验：桌高 0.75m、罐高 0.1454m、半径 0.029m、罐盖低于顶面 0.004m。实物不同须修改并重验；当前不是任意高度物体抓取器。
5. 标定夹爪实际间距/指令映射。仿真使用最大驱动角 0.28rad 的近似模型，0.95 约对应 0.266rad 卸载区；不可直接把 0.95 乘真机最大行程。全开仿真约 64mm，仅略大于 58mm 罐径。
6. 确认启动姿态与 `initial_action()` 一致，低速无物空跑，再验证夹持、支撑释放和撤离。真实执行速度由已验证控制器约束；修改原策略时长/插值后需重新验证。
7. 完成每项实际验收后填写 `config/hardware_profile.template.json` 的副本，由集成人员设置 `COMMISSIONED`；不得将模板直接视为已通过。调用以下入口：

```python
import json
from tianji_cola.run import run

# robot = 团队实现且已连接、验收的 RobotPort
with open("config/hardware_profile.real.json", encoding="utf-8") as handle:
    profile = json.load(handle)
result = run(robot, profile)
```

`visually_complete` 只表示视觉规则通过。仿真成功还由原生接触、抬起、释放支撑评估确认；真机应建立对应独立验收，仿真防碰撞中止能力不会随 policy 自动迁移。

## 保留与删除

最终代码只维护 `tianji_cola/policy.py`。仓库 `policies/tianji_cola` 是指向本包的兼容链接。旧策略候选和仅针对退役实现的测试/启动配置已清理，明细及删除前哈希见 `cleanup_manifest.json`。保留原始验证归档和当前实验审计记录，不把这些历史证据当作新的策略入口。
