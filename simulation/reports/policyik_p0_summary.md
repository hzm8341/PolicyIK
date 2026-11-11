# PolicyIK P0 本地复现总结（OpenArm 仿真）

- 日期：2026-11-10（UTC 10:03–10:19）
- 证据等级：**本地实测**。所有数字都来自 `outputs/` 下的日志和原始 NPZ，可以用 `policyik_sim.metrics` 复算。
- 硬件：RTX 5090 32 GB，驱动 580.178.04，Ubuntu 24.04.4，内核 7.0.0-42
- 软件：conda `policyik`（Python 3.11.18）；torch 2.7.0+cu128；Isaac Sim 5.1.0 standalone（`~/isaac-sim-standalone-5.1.0-linux-x86_64`，build 5.1.0-rc.19+release.26219）；Isaac Lab v2.3.2（37ddf62）；OpenArm Isaac Lab bad82e2；rl_games 1.6.1
- 仓库：22f3a5c。输入哈希见 `outputs/audit/input_sha256.txt`。

## 与官方安装路径的差异（本地适配）

| 项 | 官方 install.sh | 本次 | 对结果的影响 |
|---|---|---|---|
| Isaac Sim | `pip install isaacsim[all,extscache]==5.1.0` | 使用 home 目录下已有的 standalone 5.1.0，通过 `setup_conda_env.sh` 注入 conda 环境（`simulation/env_local.sh`） | 版本相同，仅仿真器二进制来源不同 |
| Isaac Lab | `third_party/IsaacLab` v2.3.2 | 相同；`_isaac_sim` 软链接指向 standalone | 无 |
| 评测脚本 | `scripts/evaluate.sh` | **未修改**，默认 `--num_envs 1024` | 无 |

没有改动任何评测逻辑、PD 参数、步数、任务文件或 checkpoint。显存充足（1024 env 约占 3 GB），不需要 `evaluate_local.sh`。

## P0-E 论文数值核对（4,120 任务 × 3 seeds，均值 ± 样本标准差）

| 指标 | 论文 | 本地 | 差异 |
|---|---|---|---|
| 单次 Strict | 93.49 ± 0.20 % | **93.49 ± 0.16 %**（s1 93.50 / s2 93.64 / s3 93.33） | 0.00 pp |
| 单次 Coarse | 96.56 ± 0.16 % | 96.59 ± 0.14 % | +0.03 pp |
| 单次 median 位置误差 | 0.25 ± 0.02 mm | 0.25 ± 0.02 mm | 0 |
| 单次 median 方向误差 | 0.04° | 0.04° | 0 |
| 单次最后 1 秒 containment | ≈92.4 % | 92.48 ± 0.10 % | +0.08 pp |
| 部署 Strict | 98.92 ± 0.23 % | **98.91 ± 0.18 %** | −0.01 pp |
| 部署 E[N] | ≈1.212 | 1.214（1.21 ± 0.02） | +0.002 |
| 校正目标挽回数 / seed | 54（论文附录 C） | 55 / 49 / 57 | 一致 |
| 部署后仍失败 / seed | 45（论文附录 C） | 44 / 38 / 53 | 一致 |

所有差异都远小于 2 pp 的报警阈值。全部 NPZ（3 seeds × 10 次尝试）的 `pos_t`、`ori_t` 都是有限值。

单次尝试的其他指标：
- Entry 94.75 ± 0.07 %
- 进入时间中位数 1.71 s
- 最后 1 秒净漂移 13.95 µm / 0.0020°（论文 14 µm / 0.002°）
- 平均位置误差 3.28 mm（被少数远端失败拉高）
- 平均方向误差 0.84°

部署协议各步的累计 strict 成功率：

| 步骤 | 累计 Strict |
|---|---|
| 单次 | 93.49 |
| + 校正目标 | 94.79 |
| + transient 0 | 96.94 |
| + transient 1 | 97.75 |
| + transient 2 | 98.16 |
| + transient 3 | 98.42 |
| + transient 4 | 98.61 |
| + transient 5 | 98.71 |
| + transient 6 | 98.84 |
| + transient 7 | 98.91 |

> 注意：98.91 % 是带重试的累计成功率，平均多执行约 21 % 的尝试，不能与单次 DLS 的 50.41 % 当作同预算比较。

## 失败分析（单次，互斥分类，`reports/failure_classes.txt`）

| 类别 | s1 | s2 | s3 |
|---|---|---|---|
| success_contained（均值达标，且末 30 步每一步都达标） | 3810 | 3814 | 3806 |
| success_mean_only（均值达标，但有采样点超阈值） | 42 | 44 | 39 |
| far_from_goal（均值 ≥10 mm 或 ≥2°） | 137 | 147 | 137 |
| near_miss_never_entered（接近目标但从未进入严门槛） | 77 | 76 | 86 |
| entered_then_drifted（进入过严门槛后漂出） | 42 | 29 | 46 |
| oscillating（末 30 步位置标准差 >1 mm） | 12 | 10 | 6 |
| invalid_nonfinite | 0 | 0 | 0 |

- **最大的失败类是 far_from_goal**（约 3.4 %）。极端案例收敛到错误的位姿或分支，例如 s1 的 task 573 末端偏差 854 mm / 169°，task 4082 偏差 724 mm / 151°。这类大多能被 transient 重试挽回，说明是局部陷阱，而不是目标不可达。
- 跨 seed 一致性：
  - 3605 个任务三个 seed 都成功；
  - 304 个只在 1 个 seed 失败，132 个在 2 个 seed 失败；
  - **79 个在三个 seed 都失败**。
- 关节限位层面的分析见 P1 报告（`reports/policyik_p1_summary.md`）。P1 的评测器额外导出了 q/z 时序。

## 交付清单

- `outputs/audit/`：device.txt、version.txt、pip_freeze.txt、input_sha256.txt、各安装日志
- `outputs/smoke64/`：tasks.npz、task_ids.npy、run.log、single.npz、metrics.txt（64 任务，strict 92.19 %，只验证环境）
- `outputs/eval/s{1,2,3}/`：single / corrected_tasks / corrected / transient0-7 .npz、full_protocol.log、single_metrics.txt
- `outputs/eval/three_seed_summary.txt`、`outputs/eval/run_times.txt`
- `outputs/manifests/` 下的 E00、E01、E02、E03_s2、E03_s3 .yaml
- `reports/policyik_errors.csv`：12,360 行（3 seeds × 4,120 任务），包含逐任务指标、失败分类和部署成功的尝试序号
- `tools_local/analyze_p0.py`：本地分析脚本，复用 `policyik_sim.metrics` 的定义

## 复现命令

```bash
cd PolicyIK/simulation && source env_local.sh
for s in 1 2 3; do bash scripts/evaluate.sh $s checkpoints/policyik_s$s.pth; done
python -m policyik_sim.metrics --seed_dir outputs/eval/s1 --seed_dir outputs/eval/s2 --seed_dir outputs/eval/s3
python tools_local/analyze_p0.py --seed_dir outputs/eval/s1 --seed_dir outputs/eval/s2 --seed_dir outputs/eval/s3 --csv reports/policyik_errors.csv
```

3 seeds 的完整协议共耗时约 13 分钟（每个 seed 约 4.3 分钟）。
