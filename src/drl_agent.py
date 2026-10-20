"""
drl_agent.py (clean v3)
=======================
论文 §4.4 的 DRL 实现: ψ_1 (FC+ReLU) → ψ_2 (行 softmax) → ψ_3 (映射到
[w^{WNH}-δ, w^{WNH}+δ]).

核心:
- rollout_torch(demands, yields)  -> C_T^{DRL}  (torch tensor, 可微)
- 训练: 同一组 (demands, yields) 喂 numpy 跑 WNH (baseline) 和
        torch 跑 DRL; 损失 L = exp((C_DRL - C_WNH)/C_WNH × 100).

要求: pip install torch
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore

from inventory_env import InventoryParams, simulate_episode, sample_episode_sequences, simulate_episode_sequences
from heuristics import WNHPolicy, enumerate_yield_paths


# ---------------------------------------------------------------------------
# PyTorch 神经网络
# ---------------------------------------------------------------------------
class _PolicyNet(nn.Module):
    """s_t ∈ R^{L+1}  ->  η ∈ [0,1]^{n_paths}.

    body: FC [state_dim → hidden → hidden → hidden → 2*n_paths]
    output reshape to (2, n_paths) then row-softmax -> η_j = softmax[..., 0]

    暖启动 (warm start): 最后一层 Linear 的 bias 初始化为 0, 这样初始 forward
    时 body 输出近 0, softmax(0, 0) = (0.5, 0.5), η=0.5. 由于 w_hat =
    η*(w^{WNH}-δ) + (1-η)*(w^{WNH}+δ), 所以 η=0.5 -> w_hat = w^{WNH} —
    也就是说, 训练起始, DRL 退化到 WNH 基准, 是论文的精神所在.
    """

    def __init__(self, state_dim: int, n_paths: int, hidden: Optional[list] = None):
        super().__init__()
        if hidden is None:
            hidden = [8, 16, 16]
        layers = []
        in_dim = state_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        last = nn.Linear(in_dim, 2 * n_paths)
        # 暖启动: bias = 0, weight 仍然小随机
        nn.init.zeros_(last.bias)
        nn.init.normal_(last.weight, mean=0.0, std=0.1)
        layers.append(last)
        self.body = nn.Sequential(*layers)
        self._n_paths = n_paths

    def forward(self, s):
        if s.dim() == 1:
            s = s.unsqueeze(0)
        x = self.body(s)                              # (B, 2*n_paths)
        x = x.view(-1, 2, self._n_paths)              # (B, 2, n_paths)
        eta = torch.softmax(x, dim=1)[:, 0, :]        # (B, n_paths)
        return eta


# ---------------------------------------------------------------------------
# DRL Agent 类
# ---------------------------------------------------------------------------
class DRLAgent:
    def __init__(
        self,
        params: InventoryParams,
        S_L: float,
        w_WNH: np.ndarray,
        delta: float = 0.25,
        hidden: Optional[list] = None,
        lr: float = 0.003,
        lr_decay: float = 0.98,
        lr_min: float = 0.0001,
    ):
        self.params = params
        self.S_L = S_L
        self.w_WNH = np.asarray(w_WNH, dtype=np.float64)
        self.delta = float(delta)
        self.lr = float(lr)
        self.lr_decay = float(lr_decay)
        self.lr_min = float(lr_min)

        self.paths, self.indices, _ = enumerate_yield_paths(params.L)
        self.n_paths = len(self.w_WNH)

        if HAS_TORCH:
            self.net = _PolicyNet(params.state_dim, self.n_paths, hidden)
            self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
            self.paths_t = torch.tensor(self.paths, dtype=torch.float32)
            self.w_lo_t = torch.tensor(self.w_WNH - self.delta, dtype=torch.float32)
            self.w_hi_t = torch.tensor(self.w_WNH + self.delta, dtype=torch.float32)
        else:
            self.net = None
            self.paths_t = None
            self.w_lo_t = None
            self.w_hi_t = None

    # ----------------------------------------------------------------------
    # Inference (numpy / python)
    # ----------------------------------------------------------------------
    def predict_weight_vector(self, state_np: np.ndarray) -> np.ndarray:
        if not HAS_TORCH:
            return self.w_WNH.copy()
        self.net.eval()
        with torch.no_grad():
            s = torch.as_tensor(state_np, dtype=torch.float32)
            if s.dim() == 1:
                s = s.unsqueeze(0)
            eta = self.net(s).cpu().numpy()[0]
        return eta * (self.w_WNH - self.delta) + (1.0 - eta) * (self.w_WNH + self.delta)

    def __call__(self, state: np.ndarray) -> float:
        w_hat = self.predict_weight_vector(state)
        return _policy_with_weight_vector(state, self.params, self.S_L, w_hat, self.paths)

    # ----------------------------------------------------------------------
    # PyTorch 可微 rollout (单 episode, 已抽好的 demands / yields)
    # ----------------------------------------------------------------------
    def rollout_torch(self, demands_seq: np.ndarray, yields_seq: np.ndarray):
        """返回 torch scalar C_T^{DRL}, 可反向传播到 net.parameters().

        outstanding 长度 = L, 每 t 周期:
          1) q_received = outstanding[0]; 左移; (此时 q_received = q_{t-L})
          2) I_pre = I + y * q_received
          3) 若 t <= T-L: 用网络 -> η -> w_hat -> q_t -> outstanding[-1] = q_t
          4) d_t -> C_t -> I = I_pre - d
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")
        T = self.params.T
        L = self.params.L
        device = next(self.net.parameters()).device

        demands = torch.as_tensor(demands_seq, dtype=torch.float32, device=device)
        yields_ = torch.as_tensor(yields_seq, dtype=torch.float32, device=device)

        # outstanding 长度 = L, 初始化全 0 (auto-grad 安全: 是一个 leaf tensor)
        outstanding = torch.zeros(L, dtype=torch.float32, device=device)

        I = torch.tensor(0.0, device=device)
        C = torch.tensor(0.0, device=device)
        S_L = torch.tensor(self.S_L, dtype=torch.float32, device=device)
        alpha = torch.tensor(self.params.alpha, dtype=torch.float32, device=device)
        c_o = torch.tensor(self.params.c_o, dtype=torch.float32, device=device)
        c_h = torch.tensor(self.params.c_h, dtype=torch.float32, device=device)
        c_b = torch.tensor(self.params.c_b, dtype=torch.float32, device=device)
        w_lo = self.w_lo_t.to(device)
        w_hi = self.w_hi_t.to(device)

        for t in range(1, T + 1):
            # 期间初: 接收 + 产出 (每期都取, 让 outstanding 移位准确)
            if L >= 1:
                q_received = outstanding[0]
                outstanding = torch.cat([outstanding[1:],
                                         torch.zeros(1, device=device)], dim=0)
            else:
                q_received = torch.tensor(0.0, device=device)
            y = yields_[t - 1]
            I_pre = I + y * q_received

            # 下单 (如果 t <= T - L)
            if t <= T - L and L >= 1:
                state_in = torch.cat([I_pre.unsqueeze(0), outstanding], dim=0)
                eta = self.net(state_in.unsqueeze(0))[0]    # (n_paths,)
                w_hat = eta * w_lo + (1.0 - eta) * w_hi
                q_tail = outstanding[1:]                    # (L-1,)
                if L >= 2:
                    gap = S_L - I_pre - self.paths_t.to(device) @ q_tail
                else:
                    gap = S_L - I_pre
                    gap = gap.unsqueeze(0)
                pos_gap = F.softplus(gap, beta=5.0)
                q_t = (w_hat * pos_gap).sum()
                # 用 cat 替换 in-place 写 (保持 autograd)
                outstanding = torch.cat([outstanding[:-1], q_t.unsqueeze(0)], dim=0)
            else:
                q_t = torch.tensor(0.0, device=device)

            # 单期成本
            d = demands[t - 1]
            purchase = c_o * q_received * y
            if I_pre >= d:
                holdback = c_h * (I_pre - d)
            else:
                holdback = c_b * (d - I_pre)
            c_t = purchase + holdback
            disc = alpha ** (t - 1)
            C = C + disc * c_t

            # 期末
            I = I_pre - d

        return C


# ---------------------------------------------------------------------------
# 给定权重向量 -> 订货量 (numpy, 辅助 GDSH 等)
# ---------------------------------------------------------------------------
def _policy_with_weight_vector(state, params, S_L, w_vec, paths):
    """q = Σ_j w_j · [S_L - I_t - Σ_i y_i q_{t+i-L}]^+"""
    I_pre = state[0]
    if params.L <= 1:
        return max(S_L - I_pre, 0.0)
    Q_tail = state[1:params.L]
    total = 0.0
    for k in range(paths.shape[0]):
        y_path = paths[k]
        gap = S_L - I_pre - np.dot(y_path, Q_tail)
        if gap > 0.0:
            total += w_vec[k] * gap
    return total


# ---------------------------------------------------------------------------
# 训练 (Algorithm 1)
# ---------------------------------------------------------------------------
def train_drl(
    agent: DRLAgent,
    wnh_policy,
    params: InventoryParams,
    demand_sampler,
    n_episodes: int = 100,
    n_epochs: int = 60,
    base_seed: int = 777,
    verbose: bool = False,
    log_every: int = 5,
) -> List[float]:
    """论文 Algorithm 1: 批量梯度下降, loss = exp((C_DRL - C_WNH)/C_WNH × 100).

    每个 epoch:
      1) 抽 n_episodes 个 (demands, yields) 对
      2) 用 torch.rollout 算 C_DRL, 反向
      3) 用 numpy 跑 WNH 算 C_WNH 作为 baseline
      4) loss = mean over batch of [exp((C_DRL - C_WNH) / C_WNH × 100)]
      5) Adam 步进
    """
    history: List[float] = []
    if not HAS_TORCH or agent.net is None:
        return history

    for epoch in range(n_epochs):
        lr_now = max(agent.lr_min, agent.lr * (agent.lr_decay ** epoch))
        for g in agent.optim.param_groups:
            g["lr"] = lr_now

        agent.net.train()
        agent.optim.zero_grad()

        rng = np.random.default_rng(base_seed + epoch * 9973)

        # 1) 抽 n_episodes 对 (demands, yields)
        seqs = [sample_episode_sequences(params, demand_sampler, rng)
                for _ in range(n_episodes)]

        # 2) torch 跑 DRL
        d_costs = []
        for d_seq, y_seq in seqs:
            C_drl = agent.rollout_torch(d_seq, y_seq)
            d_costs.append(C_drl)
        C_drl_stack = torch.stack(d_costs)         # (n_episodes,)

        # 3) numpy 跑 WNH
        C_wnh_mean = float(
            np.mean([
                simulate_episode_sequences(wnh_policy, params, d_seq, y_seq)[0]
                for d_seq, y_seq in seqs
            ])
        )

        # 4) Loss (数值稳定: 用 softplus 代替 exp, 防止初 epoch 数值爆炸)
        C_wnh_const = max(C_wnh_mean, 1e-6)
        ratio = (C_drl_stack - C_wnh_const) / C_wnh_const * 100.0
        # softplus(ratio) = log(1 + exp(ratio)), 与论文 exp(...) spirit 一致但数值稳定.
        loss = F.softplus(ratio).mean()

        loss.backward()
        agent.optim.step()

        history.append(float(C_drl_stack.mean().item()))

        if verbose and ((epoch + 1) % log_every == 0 or epoch == 0):
            print(f"[DRL] epoch {epoch+1:>4d}/{n_epochs}  "
                  f"C_DRL={float(C_drl_stack.mean()):9.2f}  "
                  f"C_WNH={C_wnh_mean:9.2f}  "
                  f"ratio={float(C_drl_stack.mean())/C_wnh_mean:.4f}  "
                  f"loss={float(loss):.4f}  lr={lr_now:.5f}")

    return history
