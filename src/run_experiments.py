"""
run_experiments.py
==================
复现论文 Table 3 (Sensitivity Analysis) 的主实验脚本.

实验设置 (论文 §5.3):
- 432 大规模实例
- T ∈ {50, 100}, L ∈ {2, 5}
- p ∈ {0.55, 0.6, ..., 0.95} (9 个值)
- c_o = 2, c_h = 1, c_b ∈ {17/3, 9, 19} (3 个值; cb/(cb+ch) ∈ {0.85, 0.9, 0.95})
- 需求 N(μ=10, σ∈{2, 5}) 或 Gamma(α=β_D_micro, β_D)
- β_D_mean=10, σ²_D ∈ {4, 25}, 故 gamma shape = μ²/σ² = 25 or 4

测试: 100 episodes / 实例
训练: 每个 DRL/GDSH agent 用 50-100 个训练 episodes (论文为 500)

为节省 CPU 时间, 本仓库默认设置:
- 训练 episodes = 100
- 训练 epoch = 30-50
- 测试 episodes = 50
- 实例数: 默认用 16 个代表实例而非全部 432 (可在 main() 中调整)

策略对比:
  1) LIR (基准)
  2) WNH
  3) GDSH (梯度下降优化权重)
  4) DRL (论文方法)
"""

from __future__ import annotations
import os
import sys
import time
import json
import itertools
from typing import List

import numpy as np

# 把 src 加入搜索路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inventory_env import InventoryParams, make_demand_sampler, evaluate_policy, simulate_episode, sample_episode_sequences
from heuristics import (
    LIRPolicy, WNHPolicy, GDSHPolicy, target_inventory_level, gdsh_optimize,
    enumerate_yield_paths,
)
from drl_agent import DRLAgent, train_drl
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# 构造 432 实例 (默认子集)
# ---------------------------------------------------------------------------
def build_instance_pool(small: bool = False):
    """返回实例参数列表.

    每个实例是一个 dict: {T, L, c_o, c_h, c_b, p, demand_dist, demand_kwargs}.
    small=True 用论文 §5.2 的部分 (972 个里的小子集).
    """
    instances = []

    # 共用基础参数
    cb_ratios = [0.85, 0.9, 0.95]
    c_h = 1.0
    c_o = 2.0
    ps = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    Ts = [50, 100]
    Ls = [2, 5]

    # 正态需求: μ=10, σ ∈ {2, 5}
    norm_pairs = [(2,), (5,)]
    # Gamma 需求: μ=10, σ ∈ {sqrt(2.56)=1.6, sqrt(10.24)=3.2}  -- 论文说 β ∈ {1.6, 3.2}
    # 注: 论文 β_D 就是 scale 参数. shape = μ²/σ²
    gamma_pairs = [
        # (shape, scale)
        (10**2 / 1.6**2, 1.6),
        (10**2 / 3.2**2, 3.2),
    ]

    for T, L in itertools.product(Ts, Ls):
        for p in ps:
            for cb_ratio in cb_ratios:
                c_b = c_h * cb_ratio / (1 - cb_ratio)
                # 正态 σ=2
                instances.append({
                    "T": T, "L": L, "c_o": c_o, "c_h": c_h, "c_b": c_b,
                    "p": p, "alpha": 0.99,
                    "demand_dist": "normal",
                    "demand_mu": 10.0,
                    "demand_sigma": 2.0,
                })
                # 正态 σ=5
                instances.append({
                    "T": T, "L": L, "c_o": c_o, "c_h": c_h, "c_b": c_b,
                    "p": p, "alpha": 0.99,
                    "demand_dist": "normal",
                    "demand_mu": 10.0,
                    "demand_sigma": 5.0,
                })
                # Gamma
                for shape, scale in gamma_pairs:
                    instances.append({
                        "T": T, "L": L, "c_o": c_o, "c_h": c_h, "c_b": c_b,
                        "p": p, "alpha": 0.99,
                        "demand_dist": "gamma",
                        "demand_mu": 10.0,
                        "demand_sigma": scale,    # 不严格匹配论文, 但用于 sampler
                        "gamma_shape": shape,
                        "gamma_scale": scale,
                    })

    if small:
        # 取 ~24 个有代表性的实例: T ∈ {50, 100}, L ∈ {2, 5} 各 2, p ∈ {0.55, 0.7, 0.9} 各 3, cb 选 1 个
        cb_keep = 0.9
        small_set = []
        for T, L, p in itertools.product(Ts, Ls, [0.55, 0.7, 0.9]):
            small_set.append({
                "T": T, "L": L, "c_o": c_o, "c_h": c_h,
                "c_b": c_h * cb_keep / (1 - cb_keep),
                "p": p, "alpha": 0.99,
                "demand_dist": "normal",
                "demand_mu": 10.0,
                "demand_sigma": 2.0,
            })
        # 加上 2 个 gamma + 1 个 σ=5 的实例
        small_set.append({
            "T": 50, "L": 2, "c_o": 2.0, "c_h": 1.0, "c_b": 9.0,
            "p": 0.7, "alpha": 0.99,
            "demand_dist": "gamma",
            "demand_mu": 10.0, "demand_sigma": 1.6,
            "gamma_shape": 10**2/1.6**2, "gamma_scale": 1.6,
        })
        small_set.append({
            "T": 50, "L": 5, "c_o": 2.0, "c_h": 1.0, "c_b": 9.0,
            "p": 0.55, "alpha": 0.99,
            "demand_dist": "normal",
            "demand_mu": 10.0, "demand_sigma": 5.0,
        })
        return small_set

    return instances


def make_params_from_inst(inst: dict) -> InventoryParams:
    return InventoryParams(
        T=inst["T"], L=inst["L"],
        c_o=inst["c_o"], c_h=inst["c_h"], c_b=inst["c_b"],
        p=inst["p"], alpha=inst["alpha"],
    )


def make_sampler_from_inst(inst: dict):
    if inst["demand_dist"] == "normal":
        return make_demand_sampler("normal",
                                   mu=inst["demand_mu"],
                                   sigma=inst["demand_sigma"])
    if inst["demand_dist"] == "gamma":
        return make_demand_sampler("gamma",
                                   gamma_shape=inst["gamma_shape"],
                                   gamma_scale=inst["gamma_scale"])
    raise ValueError(f"unknown dist {inst['demand_dist']}")


# ---------------------------------------------------------------------------
# 单实例上跑 LIR / WNH / GDSH / DRL, 收集成本
# ---------------------------------------------------------------------------
def run_one_instance(inst: dict, train_kwargs: dict, test_episodes: int = 50,
                     test_seed_base: int = 100000,
                     skip_drl: bool = False,
                     skip_gdsh: bool = False,
                     progress: bool = True) -> dict:
    params = make_params_from_inst(inst)
    sampler = make_sampler_from_inst(inst)

    # 1) S_L 计算
    S_L = target_inventory_level(
        params, mu_D=inst["demand_mu"], sigma_D=inst["demand_sigma"]
    )

    # 2) LIR
    lir = LIRPolicy(params, S_L)
    lir_res = evaluate_policy(lir, params, sampler, n_episodes=test_episodes,
                              base_seed=test_seed_base)

    # 3) WNH
    wnh = WNHPolicy(params, S_L)
    wnh_res = evaluate_policy(wnh, params, sampler, n_episodes=test_episodes,
                              base_seed=test_seed_base)
    w_WNH = wnh.get_weight_vector()

    # 4) GDSH: 用批量梯度下降优化权重
    gdsh_res_mean = None
    gdsh_w = w_WNH.copy()
    if not skip_gdsh:
        t0 = time.time()
        gdsh_w, _ = gdsh_optimize(
            params, S_L, sampler,
            n_episodes=train_kwargs.get("gdsh_n_episodes", 50),
            n_epochs=train_kwargs.get("gdsh_n_epochs", 30),
            lr=1e-2,
            seed=train_kwargs.get("seed", 7),
            verbose=False,
        )
        elapsed = time.time() - t0
        if progress:
            print(f"   GDSH trained in {elapsed:5.1f}s, "
                  f"||w - w_WNH||_2={np.linalg.norm(gdsh_w - w_WNH):.4f}")
        gdsh = GDSHPolicy(params, S_L, init_w=gdsh_w)
        gdsh_res = evaluate_policy(gdsh, params, sampler, n_episodes=test_episodes,
                                   base_seed=test_seed_base)
        gdsh_res_mean = gdsh_res["mean"]

    # 5) DRL
    drl_res_mean = None
    drl_history = None
    if not skip_drl and HAS_TORCH:
        delta = 0.25 if params.L in (2, 3) else 0.0884
        agent = DRLAgent(
            params, S_L, w_WNH,
            delta=delta,
            hidden=[8, 16, 16],
            lr=0.003, lr_decay=0.98, lr_min=0.0001,
        )
        t0 = time.time()
        drl_history = train_drl(
            agent, wnh, params, sampler,
            n_episodes=train_kwargs.get("drl_n_episodes", 50),
            n_epochs=train_kwargs.get("drl_n_epochs", 30),
            base_seed=train_kwargs.get("seed", 7),
            verbose=False,
            log_every=max(1, train_kwargs.get("drl_n_epochs", 30) // 5),
        )
        elapsed = time.time() - t0
        drl_res = evaluate_policy(agent, params, sampler, n_episodes=test_episodes,
                                  base_seed=test_seed_base)
        drl_res_mean = drl_res["mean"]
        if progress:
            print(f"   DRL trained in {elapsed:5.1f}s, "
                  f"final C_DRL={drl_history[-1]:.2f}, "
                  f"test C_DRL={drl_res_mean:.2f}, C_WNH={wnh_res['mean']:.2f}")

    return {
        "inst": inst,
        "S_L": S_L,
        "lir_cost": lir_res["mean"],
        "wnh_cost": wnh_res["mean"],
        "gdsh_cost": gdsh_res_mean,
        "drl_cost": drl_res_mean,
        "drl_history": drl_history,
        "gdsh_w_norm": float(np.linalg.norm(gdsh_w - w_WNH)) if not skip_gdsh else None,
    }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true", help="use small instance pool (~10 instances)")
    parser.add_argument("--test_episodes", type=int, default=50)
    parser.add_argument("--gdsh_n_episodes", type=int, default=30)
    parser.add_argument("--gdsh_n_epochs", type=int, default=20)
    parser.add_argument("--drl_n_episodes", type=int, default=30)
    parser.add_argument("--drl_n_epochs", type=int, default=20)
    parser.add_argument("--skip_drl", action="store_true")
    parser.add_argument("--skip_gdsh", action="store_true")
    parser.add_argument("--max_instances", type=int, default=8)
    args = parser.parse_args()

    instances = build_instance_pool(small=args.small)
    if args.max_instances and len(instances) > args.max_instances:
        # 子集: 等步长抽取
        idx = np.linspace(0, len(instances) - 1, args.max_instances).astype(int)
        instances = [instances[i] for i in idx]

    print(f"\n=== Run {len(instances)} instance(s) (small={args.small}) ===")
    train_kwargs = dict(
        gdsh_n_episodes=args.gdsh_n_episodes,
        gdsh_n_epochs=args.gdsh_n_epochs,
        drl_n_episodes=args.drl_n_episodes,
        drl_n_epochs=args.drl_n_epochs,
        seed=7,
    )
    print(f"Training: gdsh_epochs={args.gdsh_n_epochs} (ep={args.gdsh_n_episodes}), "
          f"drl_epochs={args.drl_n_epochs} (ep={args.drl_n_episodes}), "
          f"test_episodes={args.test_episodes}")

    results = []
    t0_all = time.time()
    for i, inst in enumerate(instances):
        print(f"\n--- Instance {i+1}/{len(instances)} ---")
        print(f"   T={inst['T']}, L={inst['L']}, p={inst['p']}, "
              f"cb/(cb+ch)={inst['c_b']/(inst['c_b']+inst['c_h']):.3f}, "
              f"dist={inst['demand_dist']}, σ={inst.get('demand_sigma', inst.get('gamma_scale', '-'))}")
        res = run_one_instance(inst, train_kwargs,
                               test_episodes=args.test_episodes,
                               skip_drl=args.skip_drl,
                               skip_gdsh=args.skip_gdsh)
        results.append(res)
        print(f"   LIR={res['lir_cost']:.2f}  WNH={res['wnh_cost']:.2f}  "
              f"GDSH={res['gdsh_cost']:.2f}  DRL={res['drl_cost']:.2f}")

    elapsed_all = time.time() - t0_all
    print(f"\n=== Done in {elapsed_all:.1f}s ===\n")

    # 写结果
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "experiment_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results written to: {out_path}")

    # 总结
    print("\n=== Summary (mean over all instances) ===")
    keys = ["lir_cost", "wnh_cost", "gdsh_cost", "drl_cost"]
    means = {}
    for k in keys:
        vals = [r[k] for r in results if r[k] is not None]
        if vals:
            means[k] = float(np.mean(vals))
        else:
            means[k] = None
    if means["lir_cost"] is not None:
        print(f"  LIR  = {means['lir_cost']:.2f}")
    if means["wnh_cost"] is not None:
        print(f"  WNH  = {means['wnh_cost']:.2f}", end="")
        if means["lir_cost"]:
            print(f"   (gap vs LIR = {100*(means['wnh_cost']-means['lir_cost'])/means['lir_cost']:+.2f}%)")
        else:
            print()
    if means["gdsh_cost"] is not None:
        print(f"  GDSH = {means['gdsh_cost']:.2f}", end="")
        if means["lir_cost"]:
            print(f"   (gap vs LIR = {100*(means['gdsh_cost']-means['lir_cost'])/means['lir_cost']:+.2f}%)")
        else:
            print()
    if means["drl_cost"] is not None:
        print(f"  DRL  = {means['drl_cost']:.2f}", end="")
        if means["lir_cost"]:
            print(f"   (gap vs LIR = {100*(means['drl_cost']-means['lir_cost'])/means['lir_cost']:+.2f}%)")
        else:
            print()

    return results


if __name__ == "__main__":
    main()
