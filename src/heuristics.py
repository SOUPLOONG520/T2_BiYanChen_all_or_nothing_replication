"""
heuristics.py
=============
复现论文 §4.2-4.3 的三种启发式策略:
  - LIR (Linear Inflation Rule, Inderfurth 2009; Inderfurth-Vogelgesang 2013)
  - WNH (Weighted Newsvendor Heuristic, Chen et al. 2018)
  - GDSH (Gradient Descent Search Heuristic, 本文 §4.3)
"""

from __future__ import annotations
from itertools import product
from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import norm

from inventory_env import InventoryParams


# ---------------------------------------------------------------------------
# 1) 目标库存水平 S_L (论文 Eq. 3)
# ---------------------------------------------------------------------------
def target_inventory_level(params: InventoryParams, mu_D: float, sigma_D: float) -> float:
    """计算 S_L = (L+1)μ_D + k √[ (L+1)σ²_D + max(L,1)·ρ²_Y/(1-ρ²_Y)·(μ²_D+σ²_D) ]
    其中 ρ_Y = √((1-p)/p), k = Φ^{-1}(c_b/(c_b+c_h)).

    仅在已知需求均值/标准差时使用 (论文 §4.2 假设样本估计, 这里直接给定参数).
    """
    if not (0.0 < params.p < 1.0):
        # p=0 或 p=1 退化为更简单形式, 此处不发散处理
        raise ValueError("p must be strictly inside (0,1) for S_L formula")
    rho2_Y = (1.0 - params.p) / params.p
    k = norm.ppf(params.c_b / (params.c_b + params.c_h))
    var_term = (params.L + 1) * sigma_D ** 2 + max(params.L, 1) * (
        rho2_Y / (1.0 - rho2_Y)
    ) * (mu_D ** 2 + sigma_D ** 2)
    S_L = (params.L + 1) * mu_D + k * np.sqrt(max(var_term, 0.0))
    return float(S_L)


# ---------------------------------------------------------------------------
# 2) LIR (Linear Inflation Rule, 论文 Eq. 2)
# ---------------------------------------------------------------------------
class LIRPolicy:
    """q_t^{LIR} = β [S_L - IP_t]^+, IP_t = I_t + p·sum(q_{t-i}).
    论文选 β = 1 (其较 β = 1/p 更优).
    """

    def __init__(self, params: InventoryParams, S_L: float, beta: float = 1.0):
        self.params = params
        self.S_L = S_L
        self.beta = beta

    def __call__(self, state: np.ndarray) -> float:
        # state = [I_pre, q_{t-1}, q_{t-2}, ..., q_{t-L+1}]
        I_pre = state[0]
        Q_tail = state[1:]
        # 使用 IP_t = I_t + p·Σ q_{t-i}
        IP = I_pre + self.params.p * np.sum(Q_tail)
        return self.beta * max(self.S_L - IP, 0.0)


# ---------------------------------------------------------------------------
# 3) WNH (Weighted Newsvendor Heuristic, 论文 Eq. 4)
# ---------------------------------------------------------------------------
def enumerate_yield_paths(L: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成 L-1 个在途订单的 2^{L-1} 个产出样本路径.

    返回:
      paths:   shape (2^{L-1}, L-1) 的 0/1 矩阵, 每行一个样本路径
      probs:   shape (2^{L-1},),    对应路径的概率
      indices: shape (2^{L-1},),    j = 1 + Σ 2^{i-1} y_i, j ∈ [1, 2^{L-1}]
    """
    if L == 1 or L == 0:
        # L=1 或 0, 路径只有一条 (空), 索引 j=1
        return np.zeros((1, 0)), np.array([1.0]), np.array([1], dtype=np.int64)
    paths = np.array(list(product([0, 1], repeat=L - 1)), dtype=np.int64)
    indices = np.array([1 + sum((2 ** i) * y_i for i, y_i in enumerate(row))
                        for row in paths], dtype=np.int64)
    return paths, indices, paths


class WNHPolicy:
    """WNH 策略. 使用 q_t^{WNH} = Σ_j w_j^{WNH} · [S_L - I_t - Σ_i y_i q_{t+i-L}]^+

    w^{WNH}_j 即对应路径的产出分布概率.
    """

    def __init__(self, params: InventoryParams, S_L: float):
        self.params = params
        self.S_L = S_L
        self.paths, self.indices, self._ = enumerate_yield_paths(params.L)
        # 论文里 WNH 的权重就是对应路径的概率: p^{k} (1-p)^{L-1-k}
        n_success = self.paths.sum(axis=1)  # 每条路径的合格数
        self.w_WNH = (params.p ** n_success) * ((1 - params.p) ** (params.L - 1 - n_success))
        # LIR 也用 S_L, 这是统一的

    def get_weight_vector(self) -> np.ndarray:
        return self.w_WNH.copy()

    def __call__(self, state: np.ndarray) -> float:
        I_pre = state[0]
        if self.params.L == 0 or self.params.L == 1:
            return max(self.S_L - I_pre, 0.0)
        Q_tail = state[1:self.params.L]  # q_{t-1}, ..., q_{t-L+1}, 共 L-1 个
        total = 0.0
        for k in range(self.paths.shape[0]):
            y_path = self.paths[k]
            gap = self.S_L - I_pre - np.dot(y_path, Q_tail)
            if gap > 0.0:
                total += self.w_WNH[k] * gap
        return total


# ---------------------------------------------------------------------------
# 4) GDSH (Gradient Descent Search Heuristic, 论文 §4.3)
# ---------------------------------------------------------------------------
class GDSHPolicy:
    """带状态无关、时间无关权重向量 w 的 GDSH 启发式.

    使用策略:
        q_t^{GDSH} = Σ_j w_j · [S_L - I_t - Σ_i y_i q_{t+i-L}]^+

    其中 j 由 y 路径编码得到, j=1+Σ 2^{i-1} y_i.

    凸性 (Corollary 1): F_t(I_t, Q_t, q_t^{GDSH}) 关于 w 是凸函数.
    """

    def __init__(self, params: InventoryParams, S_L: float, init_w: Optional[np.ndarray] = None):
        self.params = params
        self.S_L = S_L
        self.paths, self.indices, _ = enumerate_yield_paths(params.L)
        n_paths = max(1, 2 ** max(params.L - 1, 0))
        if init_w is None:
            # 初始化为 WNH 概率权重
            if params.L <= 1:
                self.w = np.ones(n_paths)
            else:
                n_success = self.paths.sum(axis=1)
                self.w = (params.p ** n_success) * ((1 - params.p) ** (params.L - 1 - n_success))
        else:
            self.w = init_w.astype(np.float64).copy()

    def get_weight_vector(self) -> np.ndarray:
        return self.w.copy()

    def set_weight_vector(self, w: np.ndarray) -> None:
        self.w = np.array(w, dtype=np.float64).copy()

    def __call__(self, state: np.ndarray) -> float:
        I_pre = state[0]
        if self.params.L == 0 or self.params.L == 1:
            return max(self.S_L - I_pre, 0.0)
        Q_tail = state[1:self.params.L]
        total = 0.0
        for k in range(self.paths.shape[0]):
            y_path = self.paths[k]
            gap = self.S_L - I_pre - np.dot(y_path, Q_tail)
            if gap > 0.0:
                total += self.w[k] * gap
        return total


def gdsh_optimize(
    params: InventoryParams,
    S_L: float,
    demand_sampler,
    n_episodes: int = 200,
    n_epochs: int = 300,
    lr: float = 5e-3,
    seed: int = 7,
    base_seed_offset: int = 0,
    verbose: bool = False,
    max_w: float = 2.0,
) -> Tuple[np.ndarray, List[float]]:
    """论文 §4.3: 用样本平均近似 E[F_t] 并用批量梯度下降优化权重向量 w.

    流程 (与 DRL 一致):
      对每个 epoch:
        收集 (用当前 w) n_episodes 个轨迹的折扣总成本 C_T
        计算平均 C_T 关于 w 的梯度 (用 finite-difference 或基于 subgradient 的自动微分).
        更新 w.

    为与论文 spirit 一致, 这里使用 finite-difference 数值梯度 + 投影 + Adam-style
    学习率衰减, 并加入 "max weight" 约束防止 w 数值爆炸.
    """
    from inventory_env import simulate_episode

    n_paths = max(1, 2 ** max(params.L - 1, 0))
    paths, indices, _ = enumerate_yield_paths(params.L)
    if params.L >= 2:
        n_success = paths.sum(axis=1)
        w_init = (params.p ** n_success) * ((1 - params.p) ** (params.L - 1 - n_success))
    else:
        w_init = np.ones(n_paths)
    w = w_init.astype(np.float64).copy()

    # Adam 优化器 (m, v)
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    beta1, beta2 = 0.9, 0.99
    eps_adam = 1e-8
    history: List[float] = []

    def policy_with_w(w_vec):
        agent = GDSHPolicy(params, S_L, init_w=w_vec)
        return lambda s: agent(s)

    def average_cost(w_vec: np.ndarray) -> float:
        agent = policy_with_w(w_vec)
        C_total = 0.0
        for i in range(n_episodes):
            rng = np.random.default_rng(seed * 100003 + base_seed_offset + i)
            C, _, _ = simulate_episode(agent, params, demand_sampler, rng)
            C_total += C
        return C_total / n_episodes

    eps_fd = 1e-3

    for epoch in range(n_epochs):
        f0 = average_cost(w)
        history.append(f0)
        grad = np.zeros_like(w)
        for j in range(n_paths):
            w_p = w.copy()
            w_p[j] = min(w_p[j] + eps_fd, max_w)
            f_p = average_cost(w_p)
            grad[j] = (f_p - f0) / eps_fd
        # Adam 步进
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** (epoch + 1))
        v_hat = v / (1 - beta2 ** (epoch + 1))
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps_adam)
        # 学习率指数衰减
        lr = lr * 0.98
        # 投影: 保持非负且不超过 max_w
        w = np.clip(w, 0.0, max_w)
        if verbose and (epoch + 1) % max(1, n_epochs // 10) == 0:
            print(f"[GDSH] epoch {epoch+1:>4d}/{n_epochs}, avg cost={f0:8.2f}, "
                  f"||w - w_init||_2={np.linalg.norm(w - w_init):8.4f}, "
                  f"||grad||={np.linalg.norm(grad):8.4f}")

    return w, history
