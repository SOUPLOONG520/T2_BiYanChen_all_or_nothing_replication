"""
inventory_env.py (v4 - corrected outstanding queue)
==================================================
修复: outstanding 队列每期都取出 q_received (不区分 t<=L 或 t>L),
     因为即便在前 L 期 outstanding[0]=0, 我们仍然要"移位 + 写新单"以保持
     outstanding 长度正确反映后续到达的订单.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class InventoryParams:
    T: int
    L: int
    c_o: float
    c_h: float
    c_b: float
    p: float
    alpha: float = 1.0
    seed: int = 0

    state_dim: int = field(init=False)

    def __post_init__(self):
        if self.L < 0:
            raise ValueError("L >= 0")
        if not (0.0 <= self.p <= 1.0):
            raise ValueError("p in [0,1]")
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError("alpha in (0,1]")
        self.state_dim = self.L + 1


def make_demand_sampler(
    dist: str = "normal",
    mu: float = 10.0,
    sigma: float = 2.0,
    gamma_shape: Optional[float] = None,
    gamma_scale: Optional[float] = None,
) -> Callable[[np.random.Generator], float]:
    def sampler(rng: np.random.Generator) -> float:
        if dist == "normal":
            d = rng.normal(mu, sigma)
        elif dist == "gamma":
            if gamma_shape is None or gamma_scale is None:
                raise ValueError("gamma shape/scale required")
            d = rng.gamma(gamma_shape, gamma_scale)
        elif dist == "uniform":
            d = rng.uniform(mu - sigma, mu + sigma)
        else:
            raise ValueError(f"unknown dist {dist}")
        return max(0.0, float(d))
    return sampler


def one_period_cost(I_pre: float, d: float, q_received: float, y: int,
                    params: InventoryParams) -> float:
    purchase = params.c_o * q_received * y
    if I_pre >= d:
        return purchase + params.c_h * (I_pre - d)
    return purchase + params.c_b * (d - I_pre)


def simulate_episode_with_sampler(
    policy_fn,
    params: InventoryParams,
    demand_sampler: Callable[[np.random.Generator], float],
    rng: np.random.Generator,
    state_init: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """用 demand_sampler 在每 t 期抽 d_t. RNG 顺序: (y_t, d_t) per t.

    outstanding 长度 = L. outstanding[i] = 第 i 期后到达的订单量.
    第一个取出 outstanding[0] = q_{t-L} (本期到达).
    然后左移; 下单 q_t 写在 outstanding[L-1] (即 L 期后到达).
    """
    if state_init is None:
        state_init = np.zeros(params.state_dim)
    state = np.array(state_init, dtype=np.float64).copy()
    outstanding = list(state[1:])
    if len(outstanding) < params.L:
        outstanding = outstanding + [0.0] * (params.L - len(outstanding))
    elif len(outstanding) > params.L:
        outstanding = outstanding[-params.L:]
    C = 0.0
    states = np.zeros((params.T + 1, params.state_dim))
    orders = np.zeros(params.T)
    states[0] = state
    for t in range(1, params.T + 1):
        y = int(rng.random() < params.p)
        # 每期取 q_received, 不论 t 如何 (因为前 L 期 outstanding[0]=0, 取出来不影响)
        if params.L >= 1:
            q_received = outstanding[0]
            outstanding = outstanding[1:] + [0.0]
        else:
            q_received = 0.0
        I_pre = state[0] + y * q_received
        # 下单 (如果 t <= T - L)
        if t <= params.T - params.L:
            s_for_policy = np.array([I_pre] + outstanding, dtype=np.float64)
            q_t = float(max(0.0, policy_fn(s_for_policy)))
        else:
            q_t = 0.0
        d = demand_sampler(rng)
        c_t = one_period_cost(I_pre, d, q_received, y, params)
        C += (params.alpha ** (t - 1)) * c_t
        I_post = I_pre - d
        # 写 outstanding (新接 q_t 在队尾)
        if params.L >= 1:
            outstanding[-1] = q_t
            if len(outstanding) > params.L:
                outstanding = outstanding[-params.L:]
        state = np.array([I_post] + outstanding[: params.L], dtype=np.float64)
        states[t] = state
        orders[t - 1] = q_t
    return float(C), states[: params.T], orders


def simulate_episode(policy_fn, params, demand_sampler, rng, state_init=None):
    return simulate_episode_with_sampler(policy_fn, params, demand_sampler, rng, state_init)


def simulate_episode_sequences(
    policy_fn, params, demands_seq, yields_seq, state_init=None,
) -> Tuple[float, np.ndarray]:
    if state_init is None:
        state_init = np.zeros(params.state_dim)
    state = np.array(state_init, dtype=np.float64).copy()
    outstanding = list(state[1:])
    if len(outstanding) < params.L:
        outstanding = outstanding + [0.0] * (params.L - len(outstanding))
    C = 0.0
    states = np.zeros((params.T + 1, params.state_dim))
    states[0] = state
    for t in range(1, params.T + 1):
        y = int(yields_seq[t - 1])
        if params.L >= 1:
            q_received = outstanding[0]
            outstanding = outstanding[1:] + [0.0]
        else:
            q_received = 0.0
        I_pre = state[0] + y * q_received
        if t <= params.T - params.L:
            s_for_policy = np.array([I_pre] + outstanding, dtype=np.float64)
            q_t = float(max(0.0, policy_fn(s_for_policy)))
        else:
            q_t = 0.0
        d = float(demands_seq[t - 1])
        c_t = one_period_cost(I_pre, d, q_received, y, params)
        C += (params.alpha ** (t - 1)) * c_t
        I_post = I_pre - d
        if params.L >= 1:
            outstanding[-1] = q_t
            if len(outstanding) > params.L:
                outstanding = outstanding[-params.L:]
        state = np.array([I_post] + outstanding[: params.L], dtype=np.float64)
        states[t] = state
    return float(C), states[: params.T]


def sample_episode_sequences(params, demand_sampler, rng):
    T = params.T
    yields_seq = np.zeros(T, dtype=np.float32)
    demands_seq = np.zeros(T, dtype=np.float32)
    for t in range(T):
        yields_seq[t] = int(rng.random() < params.p)
        demands_seq[t] = demand_sampler(rng)
    return demands_seq, yields_seq


def evaluate_policy(policy_fn, params, demand_sampler, n_episodes=100, base_seed=12345):
    costs = np.zeros(n_episodes)
    for i in range(n_episodes):
        rng = np.random.default_rng(base_seed + i)
        C, _, _ = simulate_episode_with_sampler(policy_fn, params, demand_sampler, rng)
        costs[i] = C
    return {
        "mean": float(np.mean(costs)),
        "std": float(np.std(costs, ddof=1)),
        "median": float(np.median(costs)),
        "all": costs,
    }
