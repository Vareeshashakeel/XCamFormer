"""
Explicit camera-invariance diagnostics (reviewers R3-1, R2-9).

Compares two or more feature files (e.g. the camera-removed baseline E2 and full XCamFormer):
  1. Camera probe: logistic regression predicting the camera ID from frozen embeddings, evaluated
     with identity-disjoint 5-fold cross-validation (GroupKFold on person ID) so the probe cannot
     shortcut through identity. Lower balanced accuracy  ->  less camera information.
  2. Distance statistics (cosine distance, L2-normalised features) for four pair types:
       same-ID/same-cam, same-ID/cross-cam, diff-ID/same-cam, diff-ID/cross-cam,
     the cross-camera gap of positives, and AUC(P[cross-cam positive closer than same-cam negative]).
  3. Histograms + t-SNE coloured by camera.

Usage
-----
python analysis/camera_invariance.py --feats feats/mars_base.pt feats/mars_full.pt \
       --names "Baseline (E2)" "XCamFormer" --out results/invariance_mars
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import get_variant, load_feats  # noqa: E402


def stack(d, variant):
    """query + gallery tracklets; on MARS the distractor (pid 0) and junk (pid -1) tracklets are dropped."""
    q, g = d['query'], d['gallery']
    x = torch.cat([get_variant(q, variant), get_variant(g, variant)]).float()
    pids = np.concatenate([q['pids'], g['pids']])
    cams = np.concatenate([q['camids'], g['camids']])
    if d.get('meta', {}).get('Dataset_name') == 'Mars':
        keep = pids > 0
        x, pids, cams = x[torch.from_numpy(keep)], pids[keep], cams[keep]
    return x, pids, cams


def camera_probe(x, pids, cams, folds=5, seed=0, max_n=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = F.normalize(x, dim=1).numpy()
    if max_n is not None and len(x) > max_n:
        idx = np.random.RandomState(seed).choice(len(x), max_n, replace=False)
        x, pids, cams = x[idx], pids[idx], cams[idx]
    accs, baccs = [], []
    for tr, te in GroupKFold(n_splits=folds).split(x, cams, groups=pids):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
        clf.fit(x[tr], cams[tr])
        pred = clf.predict(x[te])
        accs.append(float((pred == cams[te]).mean()))
        baccs.append(float(balanced_accuracy_score(cams[te], pred)))
    n_cam = len(np.unique(cams))
    majority = float(np.bincount(cams).max() / len(cams))
    return {'acc': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
            'balanced_acc': float(np.mean(baccs)), 'balanced_acc_std': float(np.std(baccs)),
            'chance_balanced': 1.0 / n_cam, 'majority_baseline': majority, 'n_cameras': int(n_cam), 'n': int(len(x))}


def pair_stats(x, pids, cams, max_n=4000, seed=0):
    from sklearn.metrics import roc_auc_score
    if len(x) > max_n:
        idx = np.random.RandomState(seed).choice(len(x), max_n, replace=False)
        x, pids, cams = x[idx], pids[idx], cams[idx]
    x = F.normalize(x, dim=1)
    d = (1 - x @ x.t()).numpy()
    iu = np.triu_indices(len(x), 1)
    d = d[iu]
    same_id = (pids[:, None] == pids[None, :])[iu]
    same_cam = (cams[:, None] == cams[None, :])[iu]
    groups = {
        'sameID_sameCam': d[same_id & same_cam],
        'sameID_crossCam': d[same_id & ~same_cam],
        'diffID_sameCam': d[~same_id & same_cam],
        'diffID_crossCam': d[~same_id & ~same_cam],
    }
    stats = {k: {'mean': float(v.mean()) if len(v) else float('nan'), 'n': int(len(v))} for k, v in groups.items()}
    pos_x, neg_s = groups['sameID_crossCam'], groups['diffID_sameCam']
    stats['positive_cross_camera_gap'] = stats['sameID_crossCam']['mean'] - stats['sameID_sameCam']['mean']
    stats['negative_camera_clustering_gap'] = stats['diffID_crossCam']['mean'] - stats['diffID_sameCam']['mean']
    if len(pos_x) and len(neg_s):
        rng = np.random.RandomState(seed)
        neg = neg_s if len(neg_s) <= 200000 else rng.choice(neg_s, 200000, replace=False)
        y = np.r_[np.ones(len(pos_x)), np.zeros(len(neg))]
        stats['auc_crossCamPos_vs_sameCamNeg'] = float(roc_auc_score(y, -np.r_[pos_x, neg]))
    return stats, groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--feats', nargs='+', required=True)
    p.add_argument('--names', nargs='+', required=True)
    p.add_argument('--variant', default='full', help='full | global_bn | parts_bn | block5 ...')
    p.add_argument('--out', required=True)
    p.add_argument('--probe_max_n', type=int, default=8000)
    p.add_argument('--tsne_n', type=int, default=1500)
    p.add_argument('--no_plots', action='store_true')
    a = p.parse_args()
    assert len(a.feats) == len(a.names)
    os.makedirs(a.out, exist_ok=True)

    results, all_groups, tsne_in = {}, {}, {}
    for path, name in zip(a.feats, a.names):
        d = load_feats(path)
        x, pids, cams = stack(d, a.variant)
        print(f'[{name}] n={len(x)} dim={x.shape[1]} cameras={len(np.unique(cams))}')
        probe = camera_probe(x, pids, cams, max_n=a.probe_max_n)
        stats, groups = pair_stats(x, pids, cams)
        results[name] = {'camera_probe': probe, 'distances': stats}
        all_groups[name] = groups
        tsne_in[name] = (x, cams)
        print(f"  camera probe balanced acc {probe['balanced_acc']:.3f} (chance {probe['chance_balanced']:.3f})")
        print(f"  cross-camera positive gap {stats['positive_cross_camera_gap']:.4f} | "
              f"AUC {stats.get('auc_crossCamPos_vs_sameCamNeg', float('nan')):.4f}")

    with open(os.path.join(a.out, f'invariance_{a.variant}.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # markdown table for the paper / response letter
    lines = ['| Model | Camera probe bal. acc. (chance) | d(same-ID, same-cam) | d(same-ID, cross-cam) | Cross-cam gap | d(diff-ID, same-cam) | d(diff-ID, cross-cam) | AUC |',
             '|---|---|---|---|---|---|---|---|']
    for n, r in results.items():
        pr, ds = r['camera_probe'], r['distances']
        lines.append(f"| {n} | {pr['balanced_acc']*100:.1f} ± {pr['balanced_acc_std']*100:.1f} ({pr['chance_balanced']*100:.1f}) | "
                     f"{ds['sameID_sameCam']['mean']:.3f} | {ds['sameID_crossCam']['mean']:.3f} | {ds['positive_cross_camera_gap']:.3f} | "
                     f"{ds['diffID_sameCam']['mean']:.3f} | {ds['diffID_crossCam']['mean']:.3f} | {ds.get('auc_crossCamPos_vs_sameCamNeg', float('nan')):.3f} |")
    open(os.path.join(a.out, f'invariance_{a.variant}.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    if a.no_plots:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # histograms
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 3.6), sharey=True, squeeze=False)
    for ax, (n, groups) in zip(axes[0], all_groups.items()):
        for k, c in [('sameID_sameCam', '#1f77b4'), ('sameID_crossCam', '#2ca02c'),
                     ('diffID_sameCam', '#d62728'), ('diffID_crossCam', '#ff7f0e')]:
            if len(groups[k]):
                ax.hist(groups[k], bins=60, density=True, histtype='step', lw=1.6, color=c, label=k.replace('_', ' / '))
        ax.set_title(n); ax.set_xlabel('cosine distance')
    axes[0][0].set_ylabel('density'); axes[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, f'distance_hist_{a.variant}.png'), dpi=300)

    # t-SNE coloured by camera
    from sklearn.manifold import TSNE
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 4.2), squeeze=False)
    for ax, (n, (x, cams)) in zip(axes[0], tsne_in.items()):
        idx = np.random.RandomState(0).choice(len(x), min(a.tsne_n, len(x)), replace=False)
        z = TSNE(n_components=2, init='pca', perplexity=min(30.0, (len(idx) - 1) / 3.0), random_state=0).fit_transform(F.normalize(x[idx], dim=1).numpy())
        sc = ax.scatter(z[:, 0], z[:, 1], c=cams[idx], cmap='tab10', s=4, vmin=0, vmax=9)
        ax.set_title(n); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(a.out, f'tsne_by_camera_{a.variant}.png'), dpi=300)
    print(f'[saved] plots + tables in {a.out}')


if __name__ == '__main__':
    main()
