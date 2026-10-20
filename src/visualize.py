"""
visualize.py
============
读取 results/experiment_results.json, 生成 4 张图:

  1) bar_comparison.png        — 4 个方法平均成本柱状图 + 相对 LIR 的差距
  2) cost_gap_vs_p.png         — gap vs LIR 随 p 变化
  3) cost_gap_vs_L.png         — gap vs LIR 随 L 变化
  4) cost_gap_vs_T.png         — gap vs LIR 随 T 变化
  5) drl_training_curve.png    — DRL 训练中 C_DRL 的收敛曲线
  6) ordering_heatmap.png      — 选定实例下 4 个方法的订货量热图 (论文图 B.1)
"""

from __future__ import annotations
import os
import json
import sys
from typing import List

import numpy as np

import matplotlib
matplotlib.use("Agg")  # non-interactive
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inventory_env import InventoryParams, make_demand_sampler, simulate_episode
from heuristics import LIRPolicy, WNHPolicy, GDSHPolicy, target_inventory_level


# ---------------------------------------------------------------------------
# 读结果
# ---------------------------------------------------------------------------
def load_results(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 工具: 通用画布
# ---------------------------------------------------------------------------
def _set_paper_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 110,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
    })


# ---------------------------------------------------------------------------
# 1) bar_comparison
# ---------------------------------------------------------------------------
def plot_bar_comparison(results, out_path: str):
    methods = ["LIR", "WNH", "GDSH", "DRL"]
    keys = ["lir_cost", "wnh_cost", "gdsh_cost", "drl_cost"]
    means = []
    stds = []
    for k in keys:
        vals = [r[k] for r in results if r[k] is not None]
        if vals:
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        else:
            means.append(np.nan)
            stds.append(np.nan)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    bars = ax[0].bar(methods, means, yerr=stds, capsize=4,
                     color=["#dadada", "#9ecae1", "#fdae6b", "#2c7fb8"])
    ax[0].set_ylabel("Average discounted cost")
    ax[0].set_title("Mean cost across instances (mean ± std)")
    for b, m in zip(bars, means):
        if not np.isnan(m):
            ax[0].text(b.get_x() + b.get_width() / 2, m, f"{m:,.0f}",
                       ha="center", va="bottom", fontsize=9)

    # 与 LIR 的相对差
    base = means[0]
    gaps = [(m - base) / base * 100.0 for m in means]
    colors = ["#dadada", "#9ecae1", "#fdae6b", "#2c7fb8"]
    ax[1].bar(methods, gaps, color=colors)
    ax[1].set_ylabel("Cost gap relative to LIR (%)")
    ax[1].set_title("Relative cost gap vs LIR (lower is better)")
    ax[1].axhline(0.0, color="black", lw=0.6)
    for i, g in enumerate(gaps):
        if not np.isnan(g):
            ax[1].text(i, g, f"{g:+.2f}%", ha="center",
                       va="bottom" if g >= 0 else "top", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 2) cost_gap_vs_p
# ---------------------------------------------------------------------------
def _gap_curve(results, var_key: str, group_keys: list):
    """按指定变量分组, 计算每组的相对 LIR gap."""
    groups = {}
    for r in results:
        v = r["inst"][var_key]
        key = tuple(r["inst"].get(k) for k in group_keys)
        if key not in groups:
            groups[key] = []
        groups[key].append(r)
    out = {}
    for k, grp in groups.items():
        lir = np.mean([g["lir_cost"] for g in grp if g["lir_cost"] is not None])
        if not lir:
            continue
        for label, key_ in zip(["WNH", "GDSH", "DRL"], ["wnh_cost", "gdsh_cost", "drl_cost"]):
            vals = [g[key_] for g in grp if g[key_] is not None]
            if vals:
                out.setdefault(label, []).append((k, np.mean(vals), 100 * (np.mean(vals) - lir) / lir))
    return out


def plot_cost_vs_var(results, var_key: str, var_label: str, var_fmt,
                     title: str, out_path: str,
                     extra_filter_keys=None):
    data = {}  # method -> list of (var_value, gap%)
    for r in results:
        if extra_filter_keys is not None:
            # 简单过滤 (只看一个 var 变化, 其他取第一组)
            pass
        v = r["inst"][var_key]
        lir = r["lir_cost"]
        for label, key_ in zip(["WNH", "GDSH", "DRL"], ["wnh_cost", "gdsh_cost", "drl_cost"]):
            v_ = r[key_]
            if v_ is None or lir is None:
                continue
            data.setdefault(label, []).append((v, 100 * (v_ - lir) / lir))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    if not data:
        return
    # 取 var 排序 + plot
    for label, color in zip(["WNH", "GDSH", "DRL"],
                            ["#9ecae1", "#fdae6b", "#2c7fb8"]):
        pairs = sorted(data.get(label, []), key=lambda x: x[0])
        if not pairs:
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        ax.plot(xs, ys, "o-", label=label, color=color, lw=1.8, ms=5)
    ax.axhline(0.0, color="black", lw=0.6, ls="--", label="LIR baseline")
    ax.set_xlabel(var_label)
    ax.set_ylabel("Cost gap vs LIR (%)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 5) DRL 训练曲线
# ---------------------------------------------------------------------------
def plot_drl_training_curves(results, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    plotted = 0
    for i, r in enumerate(results):
        h = r.get("drl_history")
        if not h:
            continue
        ax.plot(np.arange(1, len(h) + 1), h, alpha=0.5, lw=1,
                label=f"Inst {i+1} (T={r['inst']['T']}, L={r['inst']['L']}, p={r['inst']['p']})"
                      if i < 5 else None)
        plotted += 1
    if plotted > 5:
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    elif plotted > 0:
        ax.legend(fontsize=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean DRL training cost C_DRL")
    ax.set_title(f"DRL training curves ({plotted} instances)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 6) 订货量热图 (论文图 B.1)
# ---------------------------------------------------------------------------
def plot_ordering_heatmap(
    inst: dict,
    out_path: str,
    grid_size: int = 30,
    q_max: float = 30.0,
):
    """在一个代表性实例上画 LIR / WNH / GDSH / DRL 的订货量热图.

    x: 一些典型产线订单值 (固定一组 q_{t-1}, ..., q_{t-L+1}); 简化仅展示一维.
    y: I_t ∈ [I_min, I_max]
    """
    from heuristics import GDSHPolicy, enumerate_yield_paths
    try:
        from drl_agent import DRLAgent
        import torch as _torch
        HAS_DRL = _torch is not None
    except Exception:
        HAS_DRL = False

    params = InventoryParams(
        T=inst["T"], L=inst["L"],
        c_o=inst["c_o"], c_h=inst["c_h"], c_b=inst["c_b"],
        p=inst["p"], alpha=inst["alpha"],
    )
    S_L = target_inventory_level(
        params, mu_D=inst["demand_mu"], sigma_D=inst["demand_sigma"]
    )

    # 用 Q 全 0 (即在途订单=0) 看 I 维度的订货
    fixed_Q = [0.0] * inst["L"]
    I_grid = np.linspace(-30, 30, grid_size)
    out = {}
    # LIR
    lir = LIRPolicy(params, S_L)
    out["LIR"] = np.array([lir(np.concatenate([[I], fixed_Q])) for I in I_grid])
    # WNH
    wnh = WNHPolicy(params, S_L)
    out["WNH"] = np.array([wnh(np.concatenate([[I], fixed_Q])) for I in I_grid])
    # GDSH — 这里不能严格训 GDSH; 用 WNH 权重作退化
    gdsh = GDSHPolicy(params, S_L, init_w=wnh.get_weight_vector())
    out["GDSH-untrained"] = np.array([gdsh(np.concatenate([[I], fixed_Q])) for I in I_grid])

    # 绘制
    fig, axs = plt.subplots(1, len(out), figsize=(14, 4), sharey=True)
    for ax, (name, vals) in zip(axs, out.items()):
        ax.plot(I_grid, vals, lw=1.5)
        ax.fill_between(I_grid, 0, vals, alpha=0.3)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_title(f"{name}\n(q_pipeline=0, S_L={S_L:.1f})")
        ax.set_xlabel("Inventory level $I_t$")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("Order quantity $q_t$")
    fig.suptitle(f"Ordering policy vs inventory level (T={inst['T']}, L={inst['L']}, p={inst['p']})",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    _set_paper_style()
    result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "results")
    results_path = os.path.join(result_dir, "experiment_results.json")
    if not os.path.exists(results_path):
        print(f"No results found at {results_path}; run run_experiments.py first.")
        return
    results = load_results(results_path)
    print(f"Loaded {len(results)} instances.")

    paths = []
    paths.append(plot_bar_comparison(results, os.path.join(result_dir, "bar_comparison.png")))
    paths.append(plot_cost_vs_var(results, "p", "Yield probability $p$",
                                  "{:.2f}",
                                  "Cost gap vs LIR across yield probabilities",
                                  os.path.join(result_dir, "cost_gap_vs_p.png")))
    paths.append(plot_cost_vs_var(results, "L", "Leadtime L", "{:d}",
                                  "Cost gap vs LIR across leadtimes",
                                  os.path.join(result_dir, "cost_gap_vs_L.png")))
    paths.append(plot_cost_vs_var(results, "T", "Horizon length T", "{:d}",
                                  "Cost gap vs LIR across horizon lengths",
                                  os.path.join(result_dir, "cost_gap_vs_T.png")))
    paths.append(plot_drl_training_curves(results, os.path.join(result_dir, "drl_training_curve.png")))

    # 选第一个实例做 ordering heatmap
    if results:
        paths.append(plot_ordering_heatmap(results[0]["inst"],
                                           os.path.join(result_dir, "ordering_heatmap.png")))

    print("Generated figures:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
