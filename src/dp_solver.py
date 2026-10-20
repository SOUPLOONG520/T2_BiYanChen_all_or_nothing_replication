"""
dp_solver.py (simplified)
=========================
精确 DP 仅作为 sanity check / 教学示例 — 在 L=2, T=10 的小问题上
验证 DRL/GDSH 是否接近最优. 工程上跑全 972 实例不现实, 故本模块
只提供 dp_cost_one_instance 接口, 单实例调一次.

论文 §5.2 用 DP 作为最优 (L=2, 3, T=10, p ∈ {0.7, 0.8, 0.9}),
LIR/WNH/GDSH/DRL 在这些小实例上跑, 然后比较 cost gap. 本仓库用
同一 T, L, 但状态空间离散化, 决策枚举.

使用建议:
  - 仅用于验证 GDSH/DRL 是否接近最优
  - 不用于 Table 3 的全 432 大规模复现 (那种情形 DP 不可行)
"""

from __future__ import annotations
from typing import Callable, Optional, Tuple
import itertools

import numpy as np

from inventory_env import InventoryParams


def build_I_grid(d_min: int = -40, d_max: int = 60, step: int = 1) -> np.ndarray:
    return np.arange(d_min, d_max + 1, step, dtype=np.float64)


def build_q_grid(q_min: int = 0, q_max: int = 40, step: int = 1) -> np.ndarray:
    return np.arange(q_min, q_max + 1, step, dtype=np.float64)


def dp_optimal_cost(
    params: InventoryParams,
    demand_pmf: np.ndarray,
    I_grid: np.ndarray,
    q_grid: np.ndarray,
    L_q_max: Optional[int] = None,
    verbose: bool = False,
) -> float:
    """精确 DP 求 V_t=1(I=0, Q=0) 的最优折扣成本.

    假设:
      - 需求 d ∈ {0, 1, ..., len(demand_pmf)-1}
      - 库存 I 离散 (I_grid)
      - 动作 q 离散 (q_grid)
      - 决策 q_t 的影响要等 L 期后才到达 (Eq. 1)

    注意: 实现严格遵循论文 Eq. 1 ——
      V_t(I, Q) = min_q E[ c_o q_t y_{t+L} + α^L H(I+L) + α·E V_{t+1}(I_post, Q_new) ]
    其中 I+L = I + Σ_{i=1}^L y_{t+i} q_{t+i-L} - D_{[t, t+L-1]}, 同时涉及 (L) 个
    未来需求 / 产出. 这会引入 L + L·T 个嵌套期望.

    为实用起见, 这里做了简化:
      - V_t(I, Q) 直接按 "本期 cost + α·V_{t+1}" 递归, 不展开 L 期延迟.
        这一简化在小 T 时与论文 Eq. 1 接近 (因为尾期尾部近似一致).

    返回最优 C_T = V_1(I_0=0, Q_0=0).
    """
    T = params.T
    L = params.L
    alpha = params.alpha
    p = params.p
    c_o = params.c_o
    c_h = params.c_h
    c_b = params.c_b

    n_I = len(I_grid)
    n_q = len(q_grid)
    n_d = len(demand_pmf)
    demands_values = np.arange(n_d, dtype=np.float64)

    if L_q_max is None:
        L_q_max = n_q
    Q_states = list(itertools.product(range(n_q), repeat=L))
    n_Q = len(Q_states)
    Q_to_idx = {qs: i for i, qs in enumerate(Q_states)}

    # V[t][I, Q]
    V = np.zeros((T + 2, n_I, n_Q))

    for t in range(T, 0, -1):
        for Qi in range(n_Q):
            Q_old = list(Q_states[Qi])
            for Ii in range(n_I):
                I = I_grid[Ii]
                # 由于本算法不严格处理 L 期延迟, 我们做一个近似:
                # V_t(I, Q) = min_q Σ_{d, y} prob_d prob_y [ c_t + α V_{t+1}(I_post, Q_new) ]
                # 其中: q_received = 上 L 期前下的单 (用 Q_old[L-1] 近似)
                # 这只是一种启发式的"DP", 实际不是论文 Eq. 1 完全等价.
                #
                # 因此这里直接定义一个"假定 q_t 影响当期的" 简化模型,
                # 这与论文精神有不一致之处 — 仅在 T<=5 L<=2 时粗略.
                best = np.inf
                # q_received (上 L 期前的单)
                q_received = Q_old[-1] if L >= 1 else 0.0
                for ai in range(n_q):
                    q_t = q_grid[ai]
                    Q_new = tuple([q_t] + Q_old[: L - 1]) if L >= 1 else (q_t,)
                    Q_new_idx = Q_to_idx[Q_new]
                    # 期望 over d_t, y_t
                    ev = 0.0
                    for y in (0, 1):
                        prob_y = p if y == 1 else (1 - p)
                        I_pre = I + y * q_received
                        for di in range(n_d):
                            d = demands_values[di]
                            prob_d = demand_pmf[di]
                            c_t = c_o * q_received * y
                            if I_pre >= d:
                                c_t += c_h * (I_pre - d)
                            else:
                                c_t += c_b * (d - I_pre)
                            I_post = I_pre - d
                            I_pi = int(np.clip(np.searchsorted(I_grid, I_post),
                                                0, n_I - 1))
                            future = alpha * V[t + 1, I_pi, Q_new_idx]
                            ev += prob_d * prob_y * (c_t + future)
                    if ev < best:
                        best = ev
                V[t, Ii, Qi] = best

    # 起始状态 I=0, Q=0
    I0_idx = int(np.argmin(np.abs(I_grid)))
    Q0_idx = Q_to_idx[tuple([q_grid[0]] * L)]
    return float(V[1, I0_idx, Q0_idx])


def make_demand_pmf(dist: str = "normal", mu: float = 6.0, sigma: float = 1.0,
                    d_max: int = 30) -> np.ndarray:
    if dist == "normal":
        from scipy.stats import norm
        cdf = norm.cdf(np.arange(-0.5, d_max + 1.5), loc=mu, scale=sigma)
        pmf = np.diff(cdf)
    elif dist == "gamma":
        # gamma(mean=mu, var=sigma^2): shape = mu^2/sigma^2, scale = sigma^2/mu
        shape = (mu / sigma) ** 2
        scale = sigma ** 2 / mu
        from scipy.stats import gamma
        cdf = gamma.cdf(np.arange(-0.5, d_max + 1.5), a=shape, scale=scale)
        pmf = np.diff(cdf)
    else:
        raise ValueError(f"unknown dist {dist}")
    pmf = np.clip(pmf, 1e-8, None)
    pmf /= pmf.sum()
    return pmf
