"""
Extract every feature variant of a trained checkpoint once, so that all revision analyses run
offline on the saved file (no further GPU time).

Examples
--------
# standard test features (MARS)
python analysis/extract_features.py --Dataset_name Mars --model_path ckpt/Mars_best.pth --out feats/mars_full.pt
# + a training subset (used to fit PCA without touching the test set)
python analysis/extract_features.py --Dataset_name Mars --model_path ckpt/Mars_best.pth --out feats/mars_full.pt --train_subset 2000
# occluded queries (gallery stays clean)
python analysis/extract_features.py --Dataset_name Mars --model_path ckpt/Mars_best.pth --out feats/mars_full_band30.pt --occ_mode band --occ_ratio 0.3
# cross-dataset: MARS-trained model evaluated on iLIDS-VID (split 0..9)
python analysis/extract_features.py --Dataset_name iLIDSVID --split_id 0 --model_path ckpt/Mars_best.pth --out feats/mars2ilids_s0.pt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import ALL_BLOCKS, extract, load_model, retrieval  # noqa: E402
from Dataloader import test_loaders  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--Dataset_name', required=True)
    p.add_argument('--model_path', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--split_id', type=int, default=0)
    p.add_argument('--seq_len', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--flip', action='store_true')
    p.add_argument('--train_subset', type=int, default=0, help='also extract N random training tracklets (0 = skip)')
    p.add_argument('--occ_mode', choices=['erase', 'band'], default=None)
    p.add_argument('--occ_ratio', type=float, default=0.0)
    p.add_argument('--occ_target', choices=['query', 'both'], default='query')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loaders, _ = test_loaders(a.Dataset_name, split_id=a.split_id, seq_len=a.seq_len,
                              num_workers=a.num_workers, which=('query', 'gallery'))
    if a.train_subset > 0:
        tl, _ = test_loaders(a.Dataset_name, split_id=a.split_id, seq_len=a.seq_len,
                             num_workers=a.num_workers, which=('train',), max_items=a.train_subset)
        loaders['train'] = tl['train']

    model = load_model(a.model_path, device)
    saved = {'meta': vars(a)}
    for name, loader in loaders.items():
        occ = a.occ_mode if (name == 'query' or (name == 'gallery' and a.occ_target == 'both')) else None
        print(f'[extract] {name}: {len(loader)} tracklets' + (f' (occlusion {occ} {a.occ_ratio})' if occ else ''))
        saved[name] = extract(model, loader, device, flip=a.flip, blocks=ALL_BLOCKS,
                              occ_mode=occ, occ_ratio=a.occ_ratio, seed=a.seed, desc=name)

    r = retrieval(saved['query']['full'], saved['gallery']['full'], saved['query'], saved['gallery'])
    print(f"[check] full embedding: Rank-1 {r['rank1']:.4f}  Rank-5 {r['rank5']:.4f}  mAP {r['mAP']:.4f}")
    saved['meta']['check'] = r
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(saved, a.out)
    print(f'[saved] {a.out}')


if __name__ == '__main__':
    main()
