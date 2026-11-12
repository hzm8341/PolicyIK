# Piper S0 小结（2026-11-11，不用 GPU 的部分）

证据等级：本地实测，全部在 CPU 上跑，没有启动仿真器。计划见仓库根目录 `PolicyIK_Piper_仿真计划_V1.md`。

## 做了什么

| 项 | 结果 |
|---|---|
| Franka FK 门槛转动项失败的原因 | **与度量的数值精度一致，未在仿真里确认。** `check_fk_sim.py` 用 `2·arccos(|dot|)` 算转角。同一组旋转只要一侧按 float32 存（仿真器的输出就是 float32），这个式子最大读出 0.032°，门槛是 0.01°；Franka 实测 0.050–0.068°，同一量级，位置差 0.014 mm 是通过的。改用相对四元数向量部分的 atan2 形式后读数为 5×10⁻⁶°。要确认，需要 GPU 空出来后用新度量重跑一次 Franka 检查 |
| Piper 运动学 | `tools_local/policyik_piper/piper_kin.py`：从真机右臂的 URDF 读运动链（文件留在 Intelligence_AOP 仓库原处，不复制进本仓库，路径可用 `PIPER_URDF` 改）。末端取 link6（法兰） |
| 运动学核对 | 与 Intelligence_AOP 里另一份独立实现（来自另一个 URDF 文件）比 200 个构型：位置最大差 0.050 mm，转动最大差 0.0038°。一次性核对，没有代码依赖 |
| 碰地标签 | 用 URDF 自带碰撞网格的凸包顶点：有顶点低于安装面 z = 0 就算碰地（base_link、link1 除外）。不用球模型 |
| 自碰撞标签 | **没有做**，全部记为 False，`source_set.npz` 里 `self_collision_screened = False`。按计划第 6 节第 4 项，S1 在仿真器里抽样测自碰撞比例后再决定这些集合能不能直接用。参考：OpenArm 源集合的自碰撞比例是 1.5 % |
| 数据集 | `outputs/piper/data/`（该目录不进 git），`make_piper_tasks.py` 生成；连跑两次，全部数组的 sha256 相同（见 `manifest.json`） |

## 数据集数字

- 20 万个构型，在 URDF 关节限位内均匀采样（seed 20261111）。和 OpenArm 源集合一样用完整限位，不留余量；
  关节 2 下限、关节 3 上限正好是 0。
- 碰地 26.9 %，其余 73.1 % 记为有效；有效构型里 67.8 % 在部署工作空间内（OpenArm 是 67.3 %）。
- 训练起点池 137,906，训练目标池 92,165，测试任务 4,120，调参任务 1,024；调参集、训练池与测试集不相交（有断言）。
- 测试任务的初始位置误差 P5 / P50 / P95：0.167 / 0.465 / 0.859 m。
- 测试目标构型离最近关节限位的余量：P5 为 0.015 rad，中位数 0.194 rad；15.9 % 的目标在限位 0.05 rad 以内。
  6 轴臂没有零空间可以绕开限位，这部分任务预计最难，S2 和 S4 要按这个余量分层报告。

## 工作空间常数（工程选择）

OpenArm 的常数不能直接用。做法：每个常数取 Piper 有效集合里与 OpenArm 相同的分位数，取整到 5 mm。

| 常数 | OpenArm | 在 OpenArm 有效集合里的分位数 | Piper |
|---|---|---|---|
| R_MAX | 0.49 | r 的 0.962 | 0.585 |
| Z_MIN | 0.15 | z 的 0.110 | 0.185 |
| Z_MAX | 0.50 | z 的 0.808 | 0.625 |
| R_KEEPOUT | 0.128 | r 的 0.092 | 0.065 |
| Z_KEEPOUT | 0.222 | z 的 0.244 | 0.285 |

## 验证

- `python -m pytest tools_local/policyik_piper/test_piper_tasks.py -q`（`simulation/` 下，先 `source env_local.sh`）→ 退出码 0，8 passed。
- `python tools_local/policyik_piper/make_piper_tasks.py` → 退出码 0，输出即上面的数字。

## S0 还没做的（门槛 PG0 未过）

- 对照器里写死的 7 轴（`evaluate_ctrl.py`、`mpc.py`、`sampling._ARM_EXPR`）还没改。改完要用 OpenArm 做逐位相同的回归，
  这一步要 GPU；11-11 上午 GPU 被 `rsi_rl` 的训练和本仓库的微调流水占着，另一个会话也在用这些文件，所以没有动。
- Franka FK 检查用新度量重跑（同样要 GPU）。

## 给 S1 的提醒

- 真机右臂的 USD 已经有了：`Intelligence_AOP/sim/assets/piper_agx/piper_right/piper_right.usd`，不用重新转。
  要先看它转换时有没有合并固定关节，以确定仿真里末端 body 叫什么（link6 还是合并后的名字）。
- 夹爪两个滑动关节在这里取 URDF 零位（`FINGER_POS = 0`），仿真里要设成一样，否则碰地标签对不上。
- PD 增益在 S1 定下并冻结（先用 Kp 80 / Kd 4 做保持测试）。
