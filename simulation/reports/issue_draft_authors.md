<!-- 给作者的 issue 草稿（英文）。尚未发送，是否发出由你决定。 -->

**Title:** Reproduction notes: main results match; two Appendix numbers we could not reproduce (gravity FF 94.3 %, default DLS 1.60 %)

Thanks for releasing the code and checkpoints. We reproduced the OpenArm simulation results on Isaac Sim 5.1.0 + Isaac Lab v2.3.2 (RTX 5090), using the unmodified `scripts/evaluate.sh`. The results match closely:

| Metric | Paper | Ours |
|---|---|---|
| Single-attempt strict | 93.49 ± 0.20 % | 93.49 ± 0.16 % |
| Deployment strict | 98.92 % | 98.91 % |
| E[N] | ≈ 1.212 | 1.214 |

We also re-implemented the tuned DLS / QP / MPC baselines of Appendix E/I. Tuning on a disjoint task set selected the same parameters as the paper: k = 2 for DLS and k = 1.5 for QP/MPC, with a 0.15 rad lead. On the 4,120 test tasks we got 52.9 % / 57.9 % / 58.5 % strict, against 50.4 % / 56.2 % / 56.3 % in the paper.

We could not reproduce two numbers and would appreciate clarification.

**1. PolicyIK with gravity feedforward (Appendix E: "94.3 % strict")**

We added the simulator's exact gravity compensation (`get_gravity_compensation_forces()`) as a joint effort target at every physics substep, on top of the released `policyik_s{1,2,3}.pth`. Strict success drops to 0.0 %, with the final error stalling at about 38 mm.

Our explanation: the command-state observation d = z_{t-1} − q_t carries the holding offset Kp⁻¹τg(q) during training (for example, about −0.09 rad on joint 4). With FF, d ≈ 0, which the policy never saw in training.

Was the 94.3 % obtained with a policy trained with FF? If so, could that checkpoint (or its training config) be released? We ask because it determines whether the released sim checkpoint can be used with an FF motor layer.

**2. Default Isaac Lab DLS (Table 6: 1.60 % strict / 42.86 % coarse, median 5.88 mm)**

Our implementation:
- q + Jᵀ(JJᵀ + 0.01²I)⁻¹e, where e is the world-frame axis-angle pose error;
- full step, anchored at the measured joints;
- targets clamped to the soft limits;
- gravity FF on;
- 30 Hz.

This gives 55.3 % strict / 59.7 % coarse (median 0.16 mm). Without FF it gives 0.0 % / 0.02 %. Neither matches the paper. Could you share the exact action configuration used for this row (for example `DifferentialInverseKinematicsActionCfg` fields such as scale, body_offset and relative mode, and whether gravity compensation was enabled)?

**3. An observation (no answer needed)**

The released checkpoint's 1 mm precision is very sensitive to PD stiffness. Scaling Kp by 0.95 / 1.05 reduces strict success to about 68–80 %, and by 0.9 / 1.1 to about 28–40 %. Coarse success stays at about 96 %. We mention it in case it is useful for others attempting sim-to-real with the sim checkpoints.

We can share scripts and per-task data if helpful.
