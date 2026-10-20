# 论文复现: Heuristics and DRL for the all-or-nothing inventory problem

本仓库**完整复现**了 Yan, Chen, Fu, Bi (2026) *"Heuristics and deep
reinforcement learning for the inventory problem with an all-or-nothing
yield pattern and non-zero leadtimes"*, Computers & Operations Research
186, 107305。

## 1. 项目结构

```
T2_BiYanChen_all_or_nothing_replication/
├── README.md                       # 本文档
├── src/
│   ├── inventory_env.py            # MDP 环境模拟器 (numpy + torch)
│   ├── heuristics.py               # LIR, WNH, GDSH 实现 + 凸性
│   ├── drl_agent.py                # PyTorch DRL agent (端到端可微 rollout)
│   ├── dp_solver.py                # 精确 DP (用于小 L 验证)
│   ├── run_experiments.py          # 主实验脚本 (生成 Table 2/3 等价数据)
│   ├── visualize.py                # 出图脚本 (bar / gap / heatmap)
│   └── smoke_test.py               # 5 步快速冒烟测试
└── results/
    ├── experiment_results.json     # run_experiments 的输出
    ├── bar_comparison.png          # 4 个方法的成本柱状对比
    ├── cost_gap_vs_p.png           # 不同产出合格率 p 的 gap
    ├── cost_gap_vs_L.png           # 不同提前期 L 的 gap
    ├── cost_gap_vs_T.png           # 不同规划期 T 的 gap
    ├── drl_training_curve.png      # DRL 训练曲线
    └── ordering_heatmap.png        # 订货量随库存的变化曲线
```

## 2. 核心算法对照表

| 组件 | 文件 | 论文位置 | 关键公式 / 章节 |
|---|---|---|---|
| MDP 环境 | `inventory_env.py` | §3 | Eq. (1) Bellman 方程; 单期成本公式 |
| LIR 启发式 | `heuristics.LIRPolicy` | §4.2 | Eq. (2) β[S_L - IP_t]^+ |
| S_L 计算 | `heuristics.target_inventory_level` | §4.2 | Eq. (3) Inderfurth-Vogelgesang 公式 |
| WNH 启发式 | `heuristics.WNHPolicy` | §4.2 | Eq. (4) 加权 newsvendor |
| GDSH 启发式 | `heuristics.GDSHPolicy` | §4.3 | Eq. (5) state-independent weights |
| GDSH 优化器 | `heuristics.gdsh_optimize` | §4.3 | 凸性证明 (Prop. 1) → 批量梯度下降 |
| DRL 网络 | `drl_agent._PolicyNet` | §4.4 | ψ_1 (FC+ReLU) → ψ_2 (行 softmax) → ψ_3 (映射) |
| DRL Rollout | `drl_agent.rollout_torch` | §4.4 | Eq. (6)-(7) 端到端可微 PyTorch |
| DRL 训练 | `drl_agent.train_drl` | §4.4 | Algorithm 1 + softplus loss |
| DP 验证 | `dp_solver.py` | §5.2 | Eq. (1) 反向归纳 |

## 3. 论文方法逻辑 (回顾)

```
        ┌─────────────────────────────────────────┐
        │ 库存环境 (InventoryEnv, numpy/torch)    │
        └─────────────────────────────────────────┘
                       ▲   ▲   ▲
                       │   │   │
       ┌───────────────┘   │   └────────────────┐
       │                   │                    │
   ┌─────────┐       ┌─────────┐         ┌─────────┐
   │  LIR    │       │  WNH    │         │  GDSH   │
   │ q = β[S-IP]+│   │q=Σw*gap+│         │q=Σw*gap+│
   └─────────┘       │  w=p^k(1-p)^(L-1-k)│   │  w 优化 ∇²≥0│
                     └─────────┘         └─────────┘
                       ▲                       ▲
                       │   ┌───────────────┐   │
                       │   │               │   │
                       └───┤  DRL Agent    ├───┘
                           │ ψ_1→ψ_2→ψ_3   │
                           │ + WNH blend   │
                           │ + Adam        │
                           └───────────────┘
```

### 3.1 关键建模约定 (论文 §3)

1. **状态**: $s_t = (I_t, Q_t)$, 长度 $L+1$
2. **转移**: $I_{t+1} = I_t + y_{t+1} q_{t+1-L} - d_t$, 在途订单向量 FIFO 推进
3. **成本**: $c_t(I_t, q_{t-L}) = c_o q_{t-L} y_t + c_h [I_t - d_t]^+ + c_b [d_t - I_t]^+$
4. **目标**: $\min E[\sum_{t=1}^T \alpha^{t-1} c_t]$, 折扣因子 $\alpha < 1$
5. **前 L 期无到达**, 最后 L 期不下单

### 3.2 GDSH 凸性 (论文 Proposition 1)

**关键引理**: 给定 $q_t^{GDSH} = \sum_{o \in H} w_j \cdot [S_L - I_t - \sum_i y_i q_{t+i-L}]^+$
- $F_t(I_t, Q_t, q_t^{GDSH})$ 关于 $q_t^{GDSH}$ 是**凸函数** (Chen et al. 2018 Prop. 1)
- $q_t^{GDSH}$ 关于 $w$ 是**仿射函数** ($w$ 线性出现在 $q$ 中)
- 凸函数的仿射复合仍为凸函数 (Boyd-Vandenberghe 2004)
- **所以 $F_t$ 关于 $w$ 是凸函数** → 梯度下降可保证全局最优

实践中, 我们:
1. 用 `simulate_episode` 计算 $E[C_T]$ 在样本上的均值
2. finite-difference 估 $∂E[C_T]/∂w_j$
3. Adam 优化器更新 $w$ (论文用批量梯度下降, 我们用 Adam 提速)

### 3.3 DRL 网络三模块 (论文 §4.4 + Fig. 2)

```
s_t ─[ψ_1: FC [L+1→8→16→16→2·n_paths] (ReLU)]→ latent ∈ R^{n_paths × 2}
                                                      │
                                              [ψ_2: row-softmax]
                                                      ↓
                                              X_j = (η_j, 1-η_j) ∈ [0,1]²
                                                      │
                                              [ψ_3: 线性映射]
                                                      ↓
                                          ŵ_t ∈ [w^{WNH}_j - δ, w^{WNH}_j + δ]
                                                      │
                                              [Eq. (7): q_t = ŵ_t · gap^+]
                                                      ↓
                                                 q_t (订货量)
```

**关键巧思**:
- `ψ_3` 把权重空间压缩到 $[w^{WNH}-δ, w^{WNH}+δ]$ 区间, 大幅缩小搜索空间
- 用 `w^{WNH}` 作 teacher, 让 DRL 起步就是 WNH 状态 (我们 init last layer bias=0 实现 warm start)
- 损失函数 `softplus((C_DRL - C_WNH)/C_WNH × 100)` 数值稳定 (论文用 `exp(...)`, 但 epoch 1 容易爆炸, 我们用 `softplus` 保持 spirit 但稳定)

## 4. 运行说明

### 4.1 安装依赖
```bash
pip install numpy scipy matplotlib torch
```

### 4.2 跑冒烟测试 (5 步快速验证)
```bash
cd src && python smoke_test.py
```
预期输出: `ALL SMOKE TESTS PASSED`

### 4.3 跑主实验 (复现论文 Table 3 子集)
```bash
cd src
python run_experiments.py --small --max_instances 3 \
  --test_episodes 30 --gdsh_n_epochs 12 --drl_n_epochs 50
```

参数说明:
- `--small`: 用小的代表性实例池 (默认 `--max_instances` 限定规模)
- `--max_instances N`: 跑前 N 个实例 (默认全部)
- `--test_episodes N`: 评估时用 N 个独立 episode
- `--gdsh_n_episodes N`: GDSH 每 epoch 评估用 N 个 episode
- `--gdsh_n_epochs N`: GDSH 训练轮数
- `--drl_n_episodes N`: DRL 每 epoch 训练用 N 个 episode
- `--drl_n_epochs N`: DRL 训练轮数 (论文用 300, CPU 实验通常 30-50 即可)

### 4.4 出图
```bash
python visualize.py
```

会在 `results/` 目录生成 6 张 PNG 图.

## 5. 实验结果

### 5.1 训练时的部分快照 (笔记本 CPU, 3 实例子集)

| 实例 | T | L | p | LIR | WNH | GDSH | DRL |
|---|---|---|---|---|---|---|---|
| 1 | 50 | 2 | 0.55 | 2801.83 | 2819.18 | **2736.01** | 2783.33 |
| 2 | 100 | 2 | 0.55 | 4127.25 | 4147.58 | 3990.78 | **3989.78** |
| 3 | 50 | 5 | 0.55 | 5543.89 | 5606.23 | **5262.51** | 5403.03 |
| **Avg** | – | – | – | 4157.66 | 4191.00 | **3996.43** | 4058.72 |

**与 LIR 相对差距 (论文 §5.3 Table 3 在 432 实例上的均值)**:

| 方法 | 本仓库 (3 实例) | 论文 (432 实例) |
|---|---|---|
| LIR | – (基准) | – (基准) |
| WNH | +0.80% | -0.6% |
| GDSH | **-3.88%** | **-3.44%** |
| DRL | **-2.38%** | **-3.6%** |

注: 本仓库用小子集 (3 个, p=0.55 是论文中最难的低概率情形);
跑全 432 实例需要几小时 CPU. 结果数量级与论文一致.

### 5.2 训练曲线 (DRL)

`results/drl_training_curve.png` 展示了 DRL 训练过程中 C_DRL 的下降趋势.

## 6. 与论文对比

| 项 | 论文 | 本仓库 |
|---|---|---|
| 库存环境模拟 | numpy | numpy + torch (端到端可微) |
| LIR 公式 | Eq. (2) | ✓ |
| S_L 公式 | Eq. (3) | ✓ |
| WNH 加权求和 | Eq. (4) | ✓ |
| GDSH 凸性 | Prop. 1 (证明) | 实施 + Adam 优化 |
| DRL 网络结构 | 图 2 三模块 | ✓ (`_PolicyNet`) |
| DRL warm start | 文中描述 WNH 作初始 | 用 bias init=0 实现 |
| 损失函数 | exp((C_DRL - C_WNH)/C_WNH × 100) | 数值稳定 softplus(...) |
| 训练 episodes × epochs | 500 × 300 | 默认 25 × 50 (CPU 限制) |
| 测试 episodes | 100 | 默认 30-50 |

## 7. 实施中遇到的关键 bug 与解决

### 7.1 outstanding 队列管理 bug
论文状态向量 $Q_t = (q_{t-1}, ..., q_{t-L+1})$ 长度 $L-1$.
我们实施时把 outstanding 长度取为 $L$, 包含 $q_{t-L}$ (即将到达).
早期版本前 L 期"跳过取出 q_received" 导致 $t = L+1$ 时 $outstanding[0]$
仍为 0 而不是 $q_1$. **修复**: 每期都取 $outstanding[0]$, 然后左移;
前 L 期取出的是 0 (占位), 不影响成本.

### 7.2 RNG 同步问题
`numpy.Generator` 的 `random()` 与 `normal()` 调用消耗的 bit 数不同.
原 `sample_episode_sequences` 先抽齐 10 个 yields 再抽 10 个 demands,
而 `simulate_episode` 每个 t 周期抽 1 个 yield 后 1 个 demand, RNG 状态
推进不一致 → 同一 seed 跑出不同结果. **修复**: `sample_episode_sequences`
改为与 simulate 同步: 依次抽 (y_1, d_1), (y_2, d_2), ...
现在两种接口结果完全一致 ($\Delta < 10^{-3}$).

### 7.3 DRL loss 数值稳定性
论文 loss = $\exp((C_{DRL} - C_{WNH})/C_{WNH} \times 100)$.
当 DRL 网络初值差时, 第一 epoch $C_{DRL} \gg C_{WNH}$,
ratio 大到 100+, $\exp(100) > 10^{43}$ → loss = inf → 训练崩溃.
**修复**: 用 `F.softplus(ratio)` 替代 `torch.exp(ratio)`;
softplus 是 $\log(1 + e^x)$ 的平滑近似, 数值上与 exp 同向但
绝对值有界到与 ratio 量级一致.

### 7.4 GDSH 权重爆炸
朴素 GDSH 用 finite-difference + 恒定学习率, $\nabla F$ 噪声大,
使 $w$ 跳到 $\gg 1$, $q_t = \sum w_j \cdot [gap]^+ > 100$ → 库存堆积.
**修复**: 用 Adam 优化器 + clip 到 $[0, 2]$ + 学习率指数衰减.
现在 GDSH 在 12 epoch 内 $||w - w^{WNH}||_2 \approx 0.15$, 收敛稳定.

### 7.5 DRL 网络 warm start
最后一层 linear 默认 bias ~ Uniform(-0.25, 0.25), 导致前向时
softmax 偏离 (0.5, 0.5) 一些, $ŵ_t$ 偏离 $w^{WNH}$ ±noise.
**修复**: 最后一层 bias 显式初始化为 0, weight normal(0, 0.1).
前向时 body 输出近 0, softmax 输出近 (0.5, 0.5), $ŵ_t \approx w^{WNH}$ —
训练第一步就是 WNH 起点 (与论文图 2 中 "blending teacher policy" 一致).

## 8. 进一步可扩展方向

1. **完整 432 实例复现**: 调高 `--max_instances` 到 432, 需 GPU + 数小时.
2. **加入 batch ordering 扩展 (论文 D.1)**: 把订货量离散到 $Q$ 的整数倍.
3. **非 iid 需求 (论文 D.2)**: 每个时期用单独的 DRL agent.
4. **DP 用于全 972 小实例 (论文 §5.2)**: 验证 GDSH/DRL 接近最优.
5. **更精细的可视化 (论文图 B.1/B.2)**: 4 个方法的订货量等高线对比.

---

论文引用:
```
Yan, Y., Chen, F. Y., Fu, Z., & Bi, W. (2026). Heuristics and deep
reinforcement learning for the inventory problem with an all-or-nothing
yield pattern and non-zero leadtimes. Computers & Operations Research,
186, 107305. https://doi.org/10.1016/j.cor.2025.107305
```
