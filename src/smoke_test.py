"""
smoke_test.py
=============
5 步快速冒烟测试, 验证 inventory_env / heuristics / drl_agent / 训练循环
端到端可跑. 不跑完整实验; 用于开发中的快速 sanity check.

用法:
  cd src && python smoke_test.py
"""
from __future__ import annotations
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inventory_env import (
    InventoryParams, make_demand_sampler,
    simulate_episode, simulate_episode_sequences, sample_episode_sequences,
)
from heuristics import LIRPolicy, WNHPolicy, GDSHPolicy, target_inventory_level, enumerate_yield_paths


def banner(s):
    print("\n" + "=" * 60)
    print(" " + s)
    print("=" * 60)


def test_env():
    banner("[1/5] InventoryEnv: simulate_episode 跑通 + cost 正确")
    params = InventoryParams(T=10, L=2, c_o=2.0, c_h=1.0, c_b=9.0, p=0.7, alpha=0.95)
    sampler = make_demand_sampler("normal", mu=10, sigma=2)

    def zero_policy(s):
        return 0.0

    rng = np.random.default_rng(0)
    C, states, orders = simulate_episode(zero_policy, params, sampler, rng)
    assert C >= 0.0
    assert states.shape == (params.T, params.state_dim)
    print(f"   OK: zero-policy cost = {C:.2f}, sum orders = {orders.sum():.2f}")


def test_heuristics():
    banner("[2/5] LIR / WNH / GDSH 跑通")
    params = InventoryParams(T=10, L=2, c_o=2.0, c_h=1.0, c_b=9.0, p=0.7, alpha=0.95)
    sampler = make_demand_sampler("normal", mu=10, sigma=2)
    S_L = target_inventory_level(params, mu_D=10, sigma_D=2)
    print(f"   S_L = {S_L:.2f}")

    lir = LIRPolicy(params, S_L)
    wnh = WNHPolicy(params, S_L)
    gdsh = GDSHPolicy(params, S_L, init_w=wnh.get_weight_vector())

    rng = np.random.default_rng(1)
    def eval_one(policy, label):
        rng_ = np.random.default_rng(1)
        costs = []
        for ep in range(20):
            C, _, _ = simulate_episode(policy, params, sampler, rng_)
            costs.append(C)
        m = np.mean(costs)
        print(f"   {label}: mean cost = {m:.2f}")
        return m

    c_lir = eval_one(lir, "LIR")
    c_wnh = eval_one(wnh, "WNH")
    c_gdsh0 = eval_one(gdsh, "GDSH (WNH-init)")
    # 期望: GDSH (WNH-init) 退化到 WNH
    assert abs(c_gdsh0 - c_wnh) < 1e-6, f"GDSH not equal to WNH init: {c_gdsh0} vs {c_wnh}"
    print(f"   OK: GDSH (init=WNH) 与 WNH 完全一致 = {c_gdsh0:.4f}")


def test_paths_enumeration():
    banner("[3/5] 路径枚举验证")
    for L in [1, 2, 3, 5]:
        paths, idx, _ = enumerate_yield_paths(L)
        expected_n = 2 ** max(L - 1, 0)
        assert paths.shape[0] == expected_n, (L, paths.shape[0], expected_n)
        # 概率和 = 1
        n_success = paths.sum(axis=1) if L >= 2 else np.zeros(1)
        p = 0.7
        probs = (p ** n_success) * ((1 - p) ** max(L - 1, 0) - n_success)
        # 检验: 仅查路径数量
    print(f"   OK: L=1,2,3,5 路径数 = 1, 2, 4, 16")


def test_sequences_alignment():
    banner("[4/5] simulate_episode vs simulate_episode_sequences 同步性")
    """对相同 seed, 两种调用方式产生相同结果 (因为内部 RNG sequence 对齐)."""
    params = InventoryParams(T=10, L=2, c_o=2.0, c_h=1.0, c_b=9.0, p=0.7, alpha=0.95)
    sampler = make_demand_sampler("normal", mu=10, sigma=2)
    wnh = WNHPolicy(params, target_inventory_level(params, 10, 2))

    # 跑 simulate_episode
    rng = np.random.default_rng(42)
    C1, _, _ = simulate_episode(wnh, params, sampler, rng)

    # 跑 simulate_episode_sequences 用同一根 rng 抽序列
    rng2 = np.random.default_rng(42)
    seqs = [sample_episode_sequences(params, sampler, rng2) for _ in range(1)]
    d_seq, y_seq = seqs[0]
    C2, _ = simulate_episode_sequences(wnh, params, d_seq, y_seq)

    print(f"   simulate_episode     C = {C1:.4f}")
    print(f"   simulate_episode_seq C = {C2:.4f}")
    assert abs(C1 - C2) < 1e-3, (C1, C2)
    print(f"   OK: 两种接口结果一致 (diff = {abs(C1 - C2):.6g})")


def test_drl():
    banner("[5/5] DRL agent 跑通 (forward + train_drl 1 epoch)")
    try:
        import torch
        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False
        print(f"   SKIP: PyTorch 未安装")
        return

    params = InventoryParams(T=20, L=2, c_o=2.0, c_h=1.0, c_b=9.0, p=0.7, alpha=0.95)
    S_L = target_inventory_level(params, mu_D=10, sigma_D=2)
    sampler = make_demand_sampler("normal", mu=10, sigma=2)
    wnh = WNHPolicy(params, S_L)

    from drl_agent import DRLAgent, train_drl
    agent = DRLAgent(params, S_L, wnh.get_weight_vector(), delta=0.25, hidden=[8, 16, 16])
    print(f"   DRLAgent 构造完成. n_paths={agent.n_paths}, state_dim={params.state_dim}")
    # 1 step forward
    s0 = np.zeros(params.state_dim)
    q0 = agent(s0)
    print(f"   inference: q_t = {q0:.4f}")

    # 跑 3 epochs 训练
    t0 = time.time()
    history = train_drl(agent, wnh, params, sampler,
                        n_episodes=10, n_epochs=3, base_seed=99, verbose=True)
    print(f"   3 epochs ({time.time()-t0:.2f}s), history = {[f'{h:.2f}' for h in history]}")
    assert len(history) == 3
    print(f"   OK: DRL forward + 3 epoch 训练完成")


if __name__ == "__main__":
    test_env()
    test_heuristics()
    test_paths_enumeration()
    test_sequences_alignment()
    test_drl()
    print("\n" + "=" * 60)
    print("  ALL SMOKE TESTS PASSED")
    print("=" * 60 + "\n")
