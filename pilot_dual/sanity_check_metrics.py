#!/usr/bin/env python3
"""
Sanity Check (metric-level): §5.3 补充验证
验证 logit 扰动的边界优势是否在最终评估指标上成立。

对 7x7 sweep 网格的每个相邻步骤计算：
    eta_BF1  = ΔE_BF1  / ΔC   (每多剪1%参数，BF1 绝对恶化量)
    eta_HD95 = ΔE_HD95 / ΔC   (每多剪1%参数，HD95 相对恶化量)

分别对 head 方向 (h↑, m固定) 和 MLP 方向 (m↑, h固定) 汇报中位数比值。

Usage:
    python -m pilot_dual.sanity_check_metrics \
        --sweep_dir results/pilot_cascade_v8_sweep \
        --output_dir results/boundary_leverage \
        --bf1_0 0.5321 --hd95_0 21.29
"""
import argparse, json, glob, os
import numpy as np

HEAD_SP = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MLP_SP  = [0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95]


def load_sweep(sweep_dir):
    data = {}
    for fname in sorted(glob.glob(os.path.join(sweep_dir, 'cascade_results_v8_w*.json'))):
        d = json.load(open(fname))
        for r in d.get('cascade_results', []):
            if r.get('phase') != 'cascade':
                continue
            h = round(r['head_sp_target'], 2)
            m = round(r['mlp_sp_target'],  2)
            data[(h, m)] = {
                'bf1':  r['macro']['mean_boundary_f1'],
                'hd95': r['macro']['mean_hd95'],
                'C':    r['param_reduction_pct'],
            }
    return data


def compute_eta(data, bf1_0, hd95_0):
    rows = []

    # Head direction: (h_i, m_j) → (h_{i+1}, m_j)
    for j, m in enumerate(MLP_SP):
        for i, h in enumerate(HEAD_SP[:-1]):
            h2 = HEAD_SP[i + 1]
            if (h, m) not in data or (h2, m) not in data:
                continue
            dC = data[(h2, m)]['C'] - data[(h, m)]['C']
            if dC <= 0:
                rows.append(dict(step_type='head', from_h=h, from_m=m, to_h=h2, to_m=m,
                                 dC=dC, eta_BF1=np.nan, eta_HD95=np.nan))
                continue
            dE_bf1  = (bf1_0  - data[(h2, m)]['bf1'])  - (bf1_0  - data[(h, m)]['bf1'])
            dE_hd95 = ((data[(h2, m)]['hd95'] - hd95_0) / hd95_0) \
                    - ((data[(h,  m)]['hd95'] - hd95_0) / hd95_0)
            rows.append(dict(step_type='head', from_h=h, from_m=m, to_h=h2, to_m=m,
                             dC=round(dC, 4),
                             eta_BF1=round(dE_bf1 / dC, 6),
                             eta_HD95=round(dE_hd95 / dC, 6)))

    # MLP direction: (h_i, m_j) → (h_i, m_{j+1})
    for i, h in enumerate(HEAD_SP):
        for j, m in enumerate(MLP_SP[:-1]):
            m2 = MLP_SP[j + 1]
            if (h, m) not in data or (h, m2) not in data:
                continue
            dC = data[(h, m2)]['C'] - data[(h, m)]['C']
            if dC <= 0:
                rows.append(dict(step_type='mlp', from_h=h, from_m=m, to_h=h, to_m=m2,
                                 dC=dC, eta_BF1=np.nan, eta_HD95=np.nan))
                continue
            dE_bf1  = (bf1_0  - data[(h, m2)]['bf1'])  - (bf1_0  - data[(h, m)]['bf1'])
            dE_hd95 = ((data[(h, m2)]['hd95'] - hd95_0) / hd95_0) \
                    - ((data[(h,  m)]['hd95'] - hd95_0) / hd95_0)
            rows.append(dict(step_type='mlp', from_h=h, from_m=m, to_h=h, to_m=m2,
                             dC=round(dC, 4),
                             eta_BF1=round(dE_bf1 / dC, 6),
                             eta_HD95=round(dE_hd95 / dC, 6)))

    return rows


def summarize(rows):
    head_rows = [r for r in rows if r['step_type'] == 'head' and not np.isnan(r['eta_BF1'])]
    mlp_rows  = [r for r in rows if r['step_type'] == 'mlp'  and not np.isnan(r['eta_BF1'])]

    h_bf1  = np.array([r['eta_BF1']  for r in head_rows])
    m_bf1  = np.array([r['eta_BF1']  for r in mlp_rows])
    h_hd95 = np.array([r['eta_HD95'] for r in head_rows])
    m_hd95 = np.array([r['eta_HD95'] for r in mlp_rows])

    eps = 1e-9
    ratio_bf1  = float(np.median(h_bf1)  / (np.median(m_bf1)  + eps))
    ratio_hd95 = float(np.median(h_hd95) / (np.median(m_hd95) + eps))

    return {
        'n_head': len(head_rows),
        'n_mlp':  len(mlp_rows),
        'head_median_eta_BF1':  round(float(np.median(h_bf1)),  6),
        'mlp_median_eta_BF1':   round(float(np.median(m_bf1)),  6),
        'head_median_eta_HD95': round(float(np.median(h_hd95)), 6),
        'mlp_median_eta_HD95':  round(float(np.median(m_hd95)), 6),
        'ratio_BF1':  round(ratio_bf1,  4),
        'ratio_HD95': round(ratio_hd95, 4),
        'check_BF1':  ratio_bf1  > 1.0,
        'check_HD95': ratio_hd95 > 1.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sweep_dir',   default='results/pilot_cascade_v8_sweep')
    p.add_argument('--output_dir',  default='results/boundary_leverage')
    p.add_argument('--bf1_0',  type=float, default=0.5321,
                   help='Baseline BF1 (unpruned MedSAM)')
    p.add_argument('--hd95_0', type=float, default=21.29,
                   help='Baseline HD95 (unpruned MedSAM)')
    args = p.parse_args()

    print(f'Baseline: BF1_0={args.bf1_0}  HD95_0={args.hd95_0}')
    data = load_sweep(args.sweep_dir)
    print(f'Loaded {len(data)}/49 configs')

    rows = compute_eta(data, args.bf1_0, args.hd95_0)

    # Print step table
    head_valid = [r for r in rows if r['step_type']=='head' and not np.isnan(r['eta_BF1'])]
    mlp_valid  = [r for r in rows if r['step_type']=='mlp'  and not np.isnan(r['eta_BF1'])]
    nan_head   = [r for r in rows if r['step_type']=='head' and     np.isnan(r['eta_BF1'])]
    nan_mlp    = [r for r in rows if r['step_type']=='mlp'  and     np.isnan(r['eta_BF1'])]

    print(f'\n{"step_type":<6} {"from_h":>6} {"from_m":>6} {"to_h":>6} {"to_m":>6} '
          f'{"dC%":>7} {"eta_BF1":>10} {"eta_HD95":>10}')
    print('-' * 70)
    for r in rows:
        e_bf1  = f'{r["eta_BF1"]:10.6f}'  if not np.isnan(r['eta_BF1'])  else '       NaN'
        e_hd95 = f'{r["eta_HD95"]:10.6f}' if not np.isnan(r['eta_HD95']) else '       NaN'
        print(f'{r["step_type"]:<6} {r["from_h"]:>6.2f} {r["from_m"]:>6.2f} '
              f'{r["to_h"]:>6.2f} {r["to_m"]:>6.2f} '
              f'{r["dC"]:>7.3f} {e_bf1} {e_hd95}')

    # Summary
    s = summarize(rows)
    print('\n' + '=' * 70)
    print('METRIC-LEVEL SANITY CHECK SUMMARY')
    print('=' * 70)
    print(f'  Head steps (valid): n={s["n_head"]}')
    print(f'  MLP  steps (valid): n={s["n_mlp"]}')
    print(f'\n  eta_BF1  median:  head={s["head_median_eta_BF1"]:+.6f}  '
          f'mlp={s["mlp_median_eta_BF1"]:+.6f}')
    print(f'  eta_HD95 median:  head={s["head_median_eta_HD95"]:+.6f}  '
          f'mlp={s["mlp_median_eta_HD95"]:+.6f}')
    print(f'\n  ratio_BF1  (head/mlp median) = {s["ratio_BF1"]:.4f}  '
          f'{"✓ (>1)" if s["check_BF1"]  else "✗ (≤1)"}')
    print(f'  ratio_HD95 (head/mlp median) = {s["ratio_HD95"]:.4f}  '
          f'{"✓ (>1)" if s["check_HD95"] else "✗ (≤1)"}')

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'sanity_check_metrics.json')
    with open(out_path, 'w') as f:
        json.dump({'baseline': {'bf1_0': args.bf1_0, 'hd95_0': args.hd95_0},
                   'summary': s,
                   'steps': rows}, f, indent=2)
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()
