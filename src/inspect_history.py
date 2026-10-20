import json
import os
os.chdir('E:\\DeepSeek\\DeepSeek\\Work\\T2BiYanChen' if os.path.exists('E:\\DeepSeek') else '.')
# Try working dir
for cand in [
    'E:\\DeepSeek工作产出\\人工智能与经济管理研究课程\\T2_BiYanChen_all_or_nothing_replication',
    'E:\\DeepSeek\\人工智能与经济管理研究课程\\T2_BiYanChen_all_or_nothing_replication',
]:
    if os.path.exists(cand):
        os.chdir(cand)
        break
with open('results/experiment_results.json', 'r', encoding='utf-8') as f:
    res = json.load(f)
for i, r in enumerate(res):
    inst = r['inst']
    print(f"Instance {i+1}: T={inst['T']}, L={inst['L']}, p={inst['p']}, sigma={inst.get('demand_sigma')}")
    h = r.get('drl_history') or []
    if h:
        n = len(h)
        for k in [0, 1, 2, 3, 4]:
            if k < n:
                print(f"  epoch {k+1:>3d}: C_train = {h[k]:.2f}")
        if n > 12:
            mid = n // 2
            print('  ...')
            for k in [mid-1, mid, mid+1]:
                if 0 <= k < n:
                    print(f"  epoch {k+1:>3d}: C_train = {h[k]:.2f}")
            print('  ...')
        for k in range(max(0, n-3), n):
            print(f"  epoch {k+1:>3d}: C_train = {h[k]:.2f}")
    print(f"  Final: LIR={r['lir_cost']:.2f}, WNH={r['wnh_cost']:.2f}, GDSH={r['gdsh_cost']:.2f}, DRL={r['drl_cost']:.2f}")
