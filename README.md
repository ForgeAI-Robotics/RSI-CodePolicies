# RSI-CodePolicies

每个发布任务使用固定目录名。各项目 README 提供本次 Isaac Sim 成功回放及原生判定。

| 策略 | 本次仿真 | 成功视频 |
| --- | --- | --- |
| [天机抓可乐](tianji_cola_policy/README.md) | seed 101，`combined_goal`，原生判定成功 | [三视角回放](tianji_cola_policy/evidence/success.mp4) |
| [天机杯子入桶](tianji_cleanup_policy/README.md) | seed 102，`nominal`，原生判定成功 | [回放](tianji_cleanup_policy/evidence/success.mp4) |

两条回放均来自策略从复位独立执行的仿真试跑，运行中 LLM 与 VLA 调用数均为 0。策略哈希、结果和验证范围详见各目录的 `manifest.json` 与 `evidence/`。

机场 Checkpoint 在 seed 1001 的复查中丢失试纸抓取观测并停机，未达到发布门槛。候选源码可在 [历史提交](https://github.com/ForgeAI-Robotics/RSI-CodePolicies/tree/2e7c430) 查看。
