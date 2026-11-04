"""
Embedding size vs accuracy vs deployment cost (reviewer R2-13).

Reports, for the full 13,056-d embedding and compact alternatives:
  Rank-1 / mAP, bytes per tracklet (fp32 / fp16), gallery storage, storage for 1M tracklets,
  brute-force matching time per query against the gallery, and model forward time per tracklet.
PCA is fitted on TRAINING features (extract with --train_subset) so no test data is used.

Usage
-----
python analysis/embedding_cost.py --feats feats/mars_full.pt --out results/embedding_mars
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import distmat, load_feats, retrieval  # noqa: E402


def match_time(qf, gf, reps=20):
    qf, gf = qf.float(), gf.float()
    n = min(reps, len(qf))
    t0 = time.perf_counter()
    for i in range(n):
        d = (gf - qf[i:i + 1]).pow(2).sum(1)
        torch.topk(d, k=min(20, len(gf)), largest=False)
    return (time.perf_counter() - t0) / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--feats', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--pca_dims', default='128,256,512,1024,2048')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(1)  # single-core matching time = conservative, hardware-independent-ish

    d = load_feats(a.feats)
    q, g = d['query'], d['gallery']
    rows = []

    def add(name, qf, gf, metric='euclidean'):
        r = retrieval(qf, gf, q, g, metric=metric)
        dim = qf.shape[1]
        rows.append({
            'variant': name, 'dim': int(dim), 'rank1': r['rank1'], 'rank5': r['rank5'], 'mAP': r['mAP'],
            'bytes_fp32': 4 * dim, 'bytes_fp16': 2 * dim,
            'gallery_MB_fp32': 4 * dim * len(gf) / 2 ** 20,
            'GB_per_1M_tracklets_fp16': 2 * dim * 1e6 / 2 ** 30,
            'match_ms_per_query_1core': 1000 * match_time(qf, gf),
        })
        print(f"{name:>22} d={dim:>6}  R1 {r['rank1']*100:5.1f}  mAP {r['mAP']*100:5.1f}  "
              f"{rows[-1]['bytes_fp32']/1024:.1f} KB/tracklet  {rows[-1]['match_ms_per_query_1core']:.2f} ms/query")

    add('Full (submitted)', q['full'], g['full'])
    add('Global only', q['global_bn'], g['global_bn'])
    add('Parts only', q['parts_bn'], g['parts_bn'])
    add('Full, fp16', q['full'].half().float(), g['full'].half().float())

    if 'train' in d:
        from sklearn.decomposition import PCA
        tr = d['train']['full'].float().numpy()
        for k in [int(x) for x in a.pca_dims.split(',')]:
            if k >= min(tr.shape):
                print(f'[skip] PCA-{k}: needs > {k} training tracklets')
                continue
            pca = PCA(n_components=k, random_state=0).fit(tr)
            qf = torch.from_numpy(pca.transform(q['full'].float().numpy())).float()
            gf = torch.from_numpy(pca.transform(g['full'].float().numpy())).float()
            add(f'Full -> PCA-{k}', qf, gf, metric='cosine')
    else:
        print('[info] no training features in file; re-extract with --train_subset 2000 to get PCA rows')

    fwd = d['query']['forward_s']
    summary = {
        'rows': rows,
        'forward_ms_per_tracklet_mean': float(1000 * np.mean(fwd)),
        'forward_ms_per_tracklet_median': float(1000 * np.median(fwd)),
        'gallery_size': int(len(g['pids'])),
        'flip': bool(d['meta'].get('flip', False)),
    }
    with open(os.path.join(a.out, 'embedding_cost.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    lines = ['| Embedding | Dim | Rank-1 | mAP | KB / tracklet (fp32) | GB / 1M tracklets (fp16) | Match ms / query (1 core) |',
             '|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['variant']} | {r['dim']} | {r['rank1']*100:.1f} | {r['mAP']*100:.1f} | {r['bytes_fp32']/1024:.2f} | "
                     f"{r['GB_per_1M_tracklets_fp16']:.2f} | {r['match_ms_per_query_1core']:.2f} |")
    lines.append(f"\nModel forward time per tracklet (all dense clips): mean {summary['forward_ms_per_tracklet_mean']:.1f} ms, "
                 f"median {summary['forward_ms_per_tracklet_median']:.1f} ms; gallery size {summary['gallery_size']}; "
                 f"flip-test {'ON (two forward passes)' if summary['flip'] else 'off'}.")
    open(os.path.join(a.out, 'embedding_cost.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
