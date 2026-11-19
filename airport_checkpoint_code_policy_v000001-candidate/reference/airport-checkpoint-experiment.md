# 固定底座机场 checkpoint 实验

本轮对接当前 Isaac Sim 4.5 中的机器人，继续使用 PyRUA-Lean，禁止 VLA。
真机尚未选定；当前结果只适用于仿真，不代表真机部署完成。

## 控制与场景

`airport/checkpoint/scene.py` 构建固定底座、队列、提包旅客、半露供纸盒和回收盒。
腰部和右臂可动，底座平移关节在 PhysX 中改为 FixedJoint。
旅客停留至少三秒，机器人从侧面擦拭包和前臂；三人完成后释放纸张入盒。

`native_runtime.py` 负责原生 PhysX 动力学与独立成功判定；
`sensors.py`、`vision.py` 从 RGB-D、编码器和力反馈生成策略输入。
`motion.py` 使用机器人 CAD/FK 做固定底座 IK 和路径预检；
`backend.py` 将这些能力接入 PyRUA-Lean 的 Backend。
MuJoCo 仅用于读取既有几何和 FK，不执行动力学步进。

策略位于 `policy.py`，文字技能位于
`airport/workflows/airport-checkpoint-workflow/SKILL.md`。
项目 primitive 包括 `measure`、`check_reachability`、`execute_workflow`
及仅保存下一轮参数提案的 `propose_parameters`。
`execute_workflow` 内部逐步检查测量、路径、力和人员运动；策略本身不依赖 LLM。

## 学习与验证

`configs/airport-checkpoint-protocol.json` 固定训练和独立测试划分。
训练使用 Astra；`training_cycle.py` 的 proposer、reviewer、maintainer
分别提出、复核和维护同一版本化候选，保留各自记录。
新提案必须重新跑原生实验，不因代码审阅通过而自动成为有效技能。

`learning.py` 将源码、参数、skill、场景和原生运行证据绑定。
只有全部训练场景的原生判定通过后才能 promotion；独立测试结果不用于调参。
记录失败和实际视频，不将 `protocol_complete`、API 返回或回放当作成功。

## 当前证据与限制

`runs/airport_checkpoint_native/smoke_04/` 保存真实 RGB-D、原生场景和构建检查：
首名旅客进站，其他旅客排队；这不是完整任务成功。
目前正在联调视觉、夹持和力补偿；尚无三人批次闭环成功或泛化通过的结论。

首轮真实 PyRUA-Lean/Astra 训练见
[`train1001_8ca59ec2_001`](../runs/airport_checkpoint_training/train1001_8ca59ec2_001/)。
原生推进 6052 步（6.052 秒仿真时间），未检测到安全违规，任务失败：
旅客驻留后，纸张追踪出现两个候选簇，策略因缺少可靠纸张测量停机，尚未抓纸。
[`native_observations.mp4`](../runs/airport_checkpoint_training/train1001_8ca59ec2_001/native_observations.mp4)
是真实相机记录；墙钟耗时约 252.5 秒，不是视频时长。

首个复盘中，三个 Astra 角色均要求补充候选点云测量，未修改策略。
该复盘 driver 遭遇原因未知的 SIGTERM：三个模型的回答及完成事件均已保存，
但 maintainer 的进程退出码未取得，不能视作完整 driver 正常退出。
详情见 [复盘记录](../runs/airport_checkpoint_learning/cycles/train1001_8ca59ec2_rich_001/cycle_result.json)。

后续三个 Astra 角色完成了三轮正常退出的独立提案、复核与维护：
供纸区域跟踪、每次接近/夹取前检查当前测量，以及供纸阶段的深度几何识别。
候选及各角色的原始记录位于 `runs/airport_checkpoint_learning/cycles/`。
所有候选均未 promotion，`CURRENT` 仍为空。
后续 `_005c` 的三位 Astra 正常退出，均为 `needs_measurement`，没有生成新候选。
审阅输入曾包含原生 GT 字段；现已改为审计白名单，历史限制及哈希见
[`airport-checkpoint-review-input-audit.md`](airport-checkpoint-review-input-audit.md)。

第二次原生运行
[`train1001_3e0321e6_002`](../runs/airport_checkpoint_training/train1001_3e0321e6_002/)
推进 8952 步（8.952 秒仿真时间、321.285 秒墙钟时间），执行 44 次末端移动。
手部遮挡导致纸张识别漂移，路径检查拒绝后停止；没有抓纸或完成采样，原生成功为 false，
独立评估未记录安全违规。原始视频、RGB-D 关键帧和拒绝请求均保留。

当前相机改为 URDF `sens_link` 固定安装，光学原点沿本地 Z 外移 5mm，
该安装修正最初采用头部角度 (-0.2, 0.2)rad，后续视角试验见下；
这是仿真实验安装假设，尚未完成硬件标定。
独立传感探测在两个机器人姿态下获得约 400–600 个纸面深度点，纸背面较暗，
因此新候选以当前深度、实测供纸盒边缘和已声明纸张尺寸进行供纸阶段识别。
合并后的 `c57be124` 在
[`train1001_c57be124_004`](../runs/airport_checkpoint_training/train1001_c57be124_004/)
推进 6752 步，纸张测量保持有效，但第一次接近动作即被保守机器人包络检查拒绝。
IK 位置误差约 0.1µm，因此不能将本次拒绝解释为 IK 不可达；具体障碍正在诊断。
未抓纸、无原生安全违规，任务失败。最终 RGB-D 已记录，夹持后跟踪仍未验证。
同候选 `_003` 因本地 driver SIGTERM 中断，仅推进至初始化 600 步，
已通过原生 RPC 正常关闭并保留证据；不计作有效策略测试。

后续离线重建精确复现该拒绝：两个近灰腕部像素被肤色规则标为人体，
保留在当前机器人 CAD 包围盒内，起点即触发检查。它们同时位于约 1m 深度跳变边缘，
仅 CAD 接近或像素角射线命中不足以安全删除；自身过滤提案因此停止。
证据见 `runs/airport_checkpoint_native/path_rejection_c57be124_01/`
和 `self_surface_edge_review_01/`，不改变人体 25mm 间距。
另对 25 张已保存原生帧离线检查，6 帧触发同类拒绝、最长连续 2 帧，
但这不能证明等待重试在原失败时刻有效。新的有界重检工程提案仍需独立原生验证；
初版普通 `hold` 保留旧位置目标，已暂缓采用，改为审查显式编码器目标冻结。
从固定 Isaac Sim 4.5 镜像读取的官方 Camera 源码使用整数像素网格与
`width/2, height/2` 主点，与项目约定一致。对三张记录帧的反投影数值对照，
每帧 19200 点的最大差约 0.17µm；这只验证公式一致，不验证 RTX 边缘归属或时序。
源码、镜像 digest 和分析记录见 `camera_intrinsics_source_audit_01/`。

工程候选 `a67f4943` 合入显式 `hold_pose`，两位独立 Astra 均仅批准进入原生探针。
工作区 143 项离线测试通过；`encoder_freeze_probe_01/` 进一步推进 1151 原生步，
验证 11 个规划关节目标等于同帧编码器、其余 15 通道目标不变，速度未清零。
1/5mm 动作中途冻结后的 200ms 内，最大 EEF 残余位移分别约 0.0201/0.1613mm，
补偿力峰值约 0.0115/0.2427mN，无原生违规、底座漂移为零。
这是无载荷诊断，不是全场景停车认证；完整任务另在
`train1001_a67f4943_005/` 推进 6803 步，连续三次路径拒绝后按上限停止。
两次 `hold_pose` 实际执行，但同一个腕部深度边缘像素持续被肤色规则标为人体。
纸张检测全程有效，无原生违规，未移动接近、抓纸或采样，任务失败。

独立 `late_sensor_view_probe_01/` 保留相同真实开爪事件，仅将初始头部角度
设为 (-0.45, 0.2)rad；6.703–7.003s 的七帧静态和原首步检查均通过，
且纸、供纸口、包和前臂可测。这不是完整路径或任务成功。
候选 `970fe1b4` 仅更改该初始角度，其他候选源码与 a67 相同；
`train1001_970fe1b4_006/` 已完成：7153 原生步、约 286.78 秒墙钟时间，
执行六次移动，末端净位移约 14.74mm。一次拒绝经重检恢复，随后连续三次拒绝停机；
没有抓纸、抽取或完成旅客采样，无原生违规，任务失败。
不同近灰机器人边缘像素仍被肤色规则标为人体，单独调整视角未解决问题。
复现与原始帧见 `path_rejection_970fe1b4_01/`。

## 相机同步与自身几何过滤诊断

`renderer_aa_stalled_probe_01/` 在相同静止关节姿态下比较 DLSS 与 DLAA：
各十帧的深度数组完全相同，但 RGB 不同，两组均有路径误拒绝。
因此不能把切换抗锯齿视为修复。仅依据肤色或 CAD 包围盒也不能可靠确定像素归属。

`pixel_ray_sampling_validation_02/` 对十二张独立保存帧比较实测深度与自身 CAD 射线：
在预先固定的内部像素集合中，全部帧均以 (+0.5, +0.5) 像素偏移获得最小残差。
这与前述 SDK 整数网格公式对照并不矛盾：公式一致不代表实际渲染采样位置已标定。
运动帧仍出现较大残差，不能靠像素偏移或扩大几何容差解决。

`camera_time_registration_probe_01/` 进一步保存物理步历史：
ReferenceTime 指向当前 world 时间，但运动深度更接近此前最近的 40ms 自动渲染姿态；
官方 Camera 回调及同步图求值没有消除该差异。
这里的历史拟合只用于诊断，不能作为真实图像时间戳替换策略输入。
后续 `camera_same_epoch_render_probe_01/` 与 `camera_zero_delay_render_probe_01/`
各推进 1151 原生步，在相同物理时刻比较三次 render：默认配置的 1/5mm 运动首帧
中位残差为 41.879/359.244µm，第二帧降至 0.217/0.181µm；
镜像自带 zero-delay experience 在首帧即得到相同低残差。
同组关节位置、速度和 world 时间均未变化，未用历史姿态拟合替换时间戳。

新工程候选现接入 `zero_delay_v1`：固定官方 experience 文件哈希、核对实际设置，
记录 ReferenceTime、world 时钟原点和编码器/相机标定绑定；完整策略接线尚待验证。
[独立审阅](../runs/airport_checkpoint_native/zero_delay_profile_review_01/review.json)
支持以 20µm 经验带进入受控仿真候选；它不是全部像素误差上界，带外点仍须保留。

`self_depth_prototype_01/` 保存未应用的自身深度过滤原型及 22 项离线测试。
它保护纸张和确认的人体点，保留比自身表面明显更近的前景；
缺少合格标定或渲染时序信息时不生成自身候选。
误差带内仍可能存在真实前景，尚未选定部署容差，也未证明原生任务成功。
原型使用独立的自身 CAD/FK 模型，不读取旅客或纸张真值。

后续已将隔离自身模型、显式像素采样标定、渲染/编码器绑定和经验深度过滤接入
新候选 `dc8e7f4f`；143 项现有测试与 21 项新增离线检查通过，Ruff 通过。
[独立接线审阅](../runs/airport_checkpoint_native/self_depth_integration_review_01/review.json)
只批准进入原生验证，未确认任务成功。
`train1001_dc8e7f4f_007/` 已在 GPU 1 完成新的 PyRUA-Lean 原生尝试，
run ID 为 `459644d9c0ef42aa9ac6081b0d13c420`：6803 步、328.62 秒墙钟时间，
113 帧自身深度检查均实际执行，渲染/world 时间差为零，纸张全程可见。
但 6703/6753/6803 的接近规划均被点云包络拒绝，两次 `hold_pose` 后停止；
末端未移动、未抓纸，无原生违规，任务失败。
审计仅缺原生成功及终止成功事件，其他绑定与证据检查通过；CURRENT 未 promotion。
实际视频为该目录的 `native_observations.mp4`，155 帧，按记录时间戳约 25fps 编码。
[逐像素重建](../runs/airport_checkpoint_native/path_rejection_dc8e7f4f_01/review.json)
复现三次拒绝：每帧九个唯一阻挡像素均不是人体或纸张保护点，排除了人体关联误保护。
其中八个指尖点来自查询模型遗漏的两块实际渲染指尖垫；复用原渲染器的精确离散形状后，
实测射线残差为 -1.29 至 +7.36µm。修正已接入，实际几何顶点及面索引与原渲染器逐字节一致。
另一个腕部点的中央射线深度差约 30.46mm，但
[精确三角测量](../runs/airport_checkpoint_native/obb_surface_diagnostic_dc8_01/review.json)
表明该三维点距腕部真实三角表面仅约 0.607µm。
OBB 窄相位仍会拒绝该点，单独更换包络无法解决。

工程接线正在加入精确三角表面匹配：在同步、帧健康和保护检查通过后，
仅将距当前自身可见表面不超过 20µm 的实测点列为自身候选。
这是新的三维 Euclidean 距离条件，不等同于旧的沿光轴深度误差带；
仍不能辨认误差带内的真实前景，也不构成真机认证。
三个原失败帧的离线规划均通过，每帧增加一个三角表面候选，纸张和人体保护点数不变。
[独立审阅](../runs/airport_checkpoint_native/self_surface_integration_review_01/review.json)
接受进入原生验证；法向和光轴方向的 162 个毫米级移位样本均未被表面分支接受，
微米带内近前景仍可能成为候选，不能据此宣称普适辨识或安全保证。
合入后的 159 项仓库测试、25 项实际接线测试及 Ruff 检查通过。
新候选 `20f4734e` 已冻结并启动 `train1001_20f4734e_008/`，GPU 1、seed 1001；
仍须检查真实接近、抓纸、抽取、三人采样及回收，未 promotion 或进入 heldout。

原生状态读取和日志已优化：接触日志无损压缩，记录模式的 500 步独立基准约提升 7.5%。
机器人使用原生重力补偿与原有 PD 控制；新增 50ms 动量观察器扣除腕部惯性反力。
独立原生探测见 `runs/airport_checkpoint_native/momentum_probe_01/`：
空载 5mm 动作的估计峰值为 0.3066mN；已知 1mN 外载测得约 1.00465mN。
这些数值来自无纸张载荷的仿真标定，柔性纸张动态和真机精度仍未验证。
`momentum_wiring_smoke_01/` 另外验证了实际 backend 接线：1851 原生步，
25 次观测的力范数与原始窗口一致，1200 连续物理步的窗口有效；没有抓纸或任务成功。
以上性能和传感诊断均不计作任务成功。

纸张为未标定的分段柔性模型；夹持在真实双指接触后使用辅助约束，
不能代表仅靠摩擦的真机抓取。人体为程序控制的运动学角色。
RGB 颜色检测适用于受控资产；遮挡、未观测区域和力传感精度仍需验证。
后续真机必须另行确定传感器、标定、控制接口和停机机制。

## 白色试纸与补光对照

按用户要求，新外观候选 `ddf486e5` 仅校正材质和照明：
`appearance.py` 将试纸设为中性漫反射白 `(0.90,0.90,0.90)`、roughness 0.8，
不添加自发光；保留原环境灯，以两块白色面光源替代单侧球灯，覆盖竖直纸片两面。
面光源工作坐标为 `(0.35,0.8,2.7)`、`(0.35,-1.0,2.7)`m，
面积参数 1.2×0.6m、intensity 1000。未修改曝光、分类阈值、控制器或避碰参数。
这是未做照度测量的室内顶灯近似，不能称为虹桥机场的真实光度标定。

可通过 `CheckpointSceneConfig.paper_material_profile=legacy_warm`、
`lighting_profile=legacy_spot` 恢复旧外观。四条件固定姿态渲染与
`train1001_ddf486e5_009/` 原生执行结果分别报告；长期学习目标保持暂停。

外观验证已结束：`appearance_ab_probe_03/` 四条件对照中，白纸肤色冲突比例约3%，
白纸加补光约4%，原设置约40%；只补光约70%。样本均未过曝，但纸面仍暗，
返回原设置存在渲染历史漂移，因此不能将规划单帧通过视为严格因果或任务成功。
`train1001_ddf486e5_009/` 实际推进9653步，仍未夹持或抽取，终止于
`paper grasp observation lost`。本次校正没有完成取纸；未修改分类器或安全条件。
启动到ready约254.8秒，动作RPC累计174.5秒、观测RPC97.8秒；
详见该运行的 `appearance_validation.json`。外观小试验以后应优先静态渲染，
避免仅为灯光比较重跑数千物理步。
