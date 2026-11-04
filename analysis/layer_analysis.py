"""
Layer-wise analysis for the choice of intermediate blocks and the complementarity question
(reviewers R2-10, R3-5, R3-6).

For every backbone block 0..10 (+ the final global embedding) and every feature file:
  * identity retrieval Rank-1 / mAP of the block feature alone,
  * camera-probe balanced accuracy (identity-disjoint CV),
  * linear CKA between the block feature and the final global embedding.
Comparing the baseline with XCamFormer shows (a) where camera information lives in the network,
(b) how much the supervised blocks change, and (c) whether supervised blocks simply duplicate the
final embedding (CKA ~ 1) or carry complementary information (CKA < 1).

Usage
-----
python analysis/layer_analysis.py --feats feats/mars_base.pt feats/mars_full.pt \
       --names "Baseline (E2)" "XCamFormer" --out results/layers_mars
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.camera_invariance import camera_probe, stack  # noqa: E402
from analysis.common import ALL_BLOCKS, linear_cka, load_feats, retrieval  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--feats', nargs='+', required=True)
    p.add_argument('--names', nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--probe_max_n', type=int, default=6000)
    p.add_argument('--cka_max_n', type=int, default=5000)
    p.add_argument('--highlight', default='5,8')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    variants = [f'block{b}' for b in ALL_BLOCKS] + ['global_bn', 'full']

    res = {}
    for path, name in zip(a.feats, a.names):
        d = load_feats(path)
        res[name] = {}
        ref, _, _ = stack(d, 'global_bn')
        rng = np.random.RandomState(0)
        sub = rng.choice(len(ref), min(a.cka_max_n, len(ref)), replace=False)
        for v in variants:
            q = d['query']['blocks'][int(v[5:])] if v.startswith('block') else d['query'][v]
            g = d['gallery']['blocks'][int(v[5:])] if v.startswith('block') else d['gallery'][v]
            r = retrieval(q, g, d['query'], d['gallery'], metric='cosine')
            x, pids, cams = stack(d, v)
            pr = camera_probe(x, pids, cams, max_n=a.probe_max_n)
            cka = linear_cka(x[sub], ref[sub]) if v != 'full' else float('nan')
            res[name][v] = {'rank1': r['rank1'], 'mAP': r['mAP'], 'cam_probe_bacc': pr['balanced_acc'],
                            'chance': pr['chance_balanced'], 'cka_vs_final_global': cka}
            print(f"[{name}] {v:>9}: R1 {r['rank1']:.3f} mAP {r['mAP']:.3f} | cam-probe {pr['balanced_acc']:.3f} | CKA {cka:.3f}")

    with open(os.path.join(a.out, 'layer_analysis.json'), 'w') as f:
        json.dump(res, f, indent=2)

    names = list(res)
    hdr = '| Block | ' + ' | '.join(f'{n} mAP | {n} cam-probe | {n} CKA→final' for n in names) + ' |'
    lines = [hdr, '|' + '---|' * (1 + 3 * len(names))]
    for v in variants:
        row = [v]
        for n in names:
            r = res[n][v]
            row += [f"{r['mAP']*100:.1f}", f"{r['cam_probe_bacc']*100:.1f}", f"{r['cka_vs_final_global']:.2f}"]
        lines.append('| ' + ' | '.join(row) + ' |')
    open(os.path.join(a.out, 'layer_analysis.md'), 'w').write('\n'.join(lines) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    hl = [int(x) for x in a.highlight.split(',') if x]
    xs = list(ALL_BLOCKS)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for n in names:
        axes[0].plot(xs, [res[n][f'block{b}']['mAP'] * 100 for b in xs], marker='o', ms=3, label=n)
        axes[1].plot(xs, [res[n][f'block{b}']['cam_probe_bacc'] * 100 for b in xs], marker='o', ms=3, label=n)
        axes[2].plot(xs, [res[n][f'block{b}']['cka_vs_final_global'] for b in xs], marker='o', ms=3, label=n)
    chance = res[names[0]]['block0']['chance'] * 100
    axes[1].axhline(chance, ls='--', c='grey', lw=1, label='chance')
    for ax, t in zip(axes, ['Identity mAP of block feature (%)', 'Camera-probe balanced acc. (%)', 'Linear CKA to final embedding']):
        for b in hl:
            ax.axvline(b, c='k', alpha=0.15, lw=6)
        ax.set_title(t, fontsize=10); ax.set_xlabel('Transformer block (0-based)'); ax.set_xticks(xs)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, 'layer_analysis.png'), dpi=300)
    print(f'[saved] {a.out}')


if __name__ == '__main__':
    main()
