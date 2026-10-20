import re

with open('E:\\DeepSeek工作产出\\人工智能与经济管理研究课程\\T2_BiYanChen_all_or_nothing_replication\\src\\inventory_env.py', 'r', encoding='utf-8') as f:
    content = f.read()

old = '''def sample_episode_sequences(
    params: InventoryParams,
    demand_sampler: Callable[[np.random.Generator], float],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """\u4e3a\u5355\u4e2a episode \u62bd\u6837 demand / yield \u5e8f\u5217."""
    T = params.T
    yields_seq = np.array([int(rng.random() < params.p) for _ in range(T)],
                          dtype=np.float32)
    demands_seq = np.array([demand_sampler(rng) for _ in range(T)],
                           dtype=np.float32)
    return demands_seq, yields_seq'''

new = '''def sample_episode_sequences(
    params: InventoryParams,
    demand_sampler: Callable[[np.random.Generator], float],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample (demand, yield) sequences for one episode.

    Order aligned with simulate_episode_with_sampler:
    each period t consumes (rng.random() < p) for yield and
    demand_sampler(rng) for demand. Thus numpy and torch rollout see
    identical random sequences.
    """
    T = params.T
    yields_seq = np.zeros(T, dtype=np.float32)
    demands_seq = np.zeros(T, dtype=np.float32)
    for t in range(T):
        yields_seq[t] = int(rng.random() < params.p)
        demands_seq[t] = demand_sampler(rng)
    return demands_seq, yields_seq'''

if old in content:
    content = content.replace(old, new)
    with open('E:\\DeepSeek工作产出\\人工智能与经济管理研究课程\\T2_BiYanChen_all_or_nothing_replication\\src\\inventory_env.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('patched OK')
else:
    print('not found - failed to patch')
    print('--- search ---')
    # Try with file directly
    pass
