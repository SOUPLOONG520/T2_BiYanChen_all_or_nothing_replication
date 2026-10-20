"""Debug DRL rollout vs numpy rollout."""
import os
os.chdir('E:\\DeepSeek\\work\\T2BiYanChen' if os.path.exists('E:\\DeepSeek') else '.')
import sys
sys.path.insert(0, 'src')
import numpy as np

from inventory_env import (
    InventoryParams, make_demand_sampler, sample_episode_sequences,
    simulate_episode_with_sampler, simulate_episode_sequences,
)
from heuristics import WNHPolicy, target_inventory_level
from drl_agent import DRLAgent

params = InventoryParams(T=50, L=2, c_o=2.0, c_h=1.0, c_b=9.0, p=0.55, alpha=0.95)
S_L = target_inventory_level(params, mu_D=10, sigma_D=2)
sampler = make_demand_sampler('normal', mu=10, sigma=2)
wnh = WNHPolicy(params, S_L)
print('S_L =', S_L)

agent = DRLAgent(params, S_L, wnh.get_weight_vector(), delta=0.25, hidden=[8,16,16])

# warm start check
s0 = np.zeros(params.state_dim)
w0 = agent.predict_weight_vector(s0)
print('w_hat (zero state) =', w0)
print('w_WNH             =', wnh.get_weight_vector())
print('diff:', np.linalg.norm(w0 - wnh.get_weight_vector()))

# Sample one episode and compare numpy vs torch rollout
rng = np.random.default_rng(42)
d_seq, y_seq = sample_episode_sequences(params, sampler, rng)

# numpy evaluate via agent.__call__
def _policy(state):
    return agent(state)

C_numpy, _ = simulate_episode_sequences(_policy, params, d_seq, y_seq)
C_torch = agent.rollout_torch(d_seq, y_seq).item()
print(f"C (numpy via agent) = {C_numpy:.4f}")
print(f"C (torch rollout)   = {C_torch:.4f}")

# And WNH
C_wnh, _ = simulate_episode_sequences(wnh, params, d_seq, y_seq)
print(f"C (WNH)             = {C_wnh:.4f}")

# Compare a single state
state = np.array([10.0, 5.0, 2.0])
w_hat = agent.predict_weight_vector(state)
print(f"state = {state}")
print(f"w_hat = {w_hat}")
print(f"q (agent) = {agent(state):.4f}")
print(f"q (WNH)   = {wnh(state):.4f}")
