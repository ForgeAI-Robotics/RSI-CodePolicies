# RSI-CodePolicies

每个任务版本使用独立目录。打开目录内的 `README.md` 查看策略入口、依赖及验证记录。

| 任务 | 版本目录说明 | 验证状态 |
| --- | --- | --- |
| Tianji 左臂抓可乐并放到头部相机中央 | [v000003](tianji_cola_policy_v000003/README.md) | Isaac Sim 原生回归 12/12，真机接口待对接 |
| Tianji 左臂夹杯身并投入垃圾桶 | [v000001-candidate](tianji_cleanup_code_policy_v000001-candidate/README.md) | Isaac Sim 冻结开发回归 1/2，含两条原生判定 |
| 机场试纸 Checkpoint | [v000001-candidate](airport_checkpoint_code_policy_v000001-candidate/README.md) | 仿真源码快照，现有记录尚无完整三人批次成功 |

## 版本管理

- 目录名使用 `<任务>_policy_v<六位版本号>`，候选在版本号后加 `-candidate`。
- 发布时为该任务新建版本目录并更新索引。每个版本保留 `README.md` 和 `manifest.json`，记录入口、来源和验证范围。

RoboCasa365 逐实例代码和早期机场工作流归档在原 RSI 工作区。抓可乐的真机接口见 [DEPLOYMENT.md](tianji_cola_policy_v000003/DEPLOYMENT.md)。
