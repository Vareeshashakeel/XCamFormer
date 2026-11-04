"""
Recover the iLIDS-VID train/test split that an old checkpoint was trained on.

The checkpoint's identity classifier has one output per TRAINING identity. Tracklets of
training identities get a confident prediction; identities never seen in training do not.
We rank all identities by the classifier confidence (averaged over both cameras) and take
the 150 most confident ones as the old training set.

Writes <ILIDS_ROOT>/splits.json with the recovered split as split 0 (the previous file is
kept as splits_backup.json).

Usage
-----
export ILIDS_ROOT=.../xcam/data/ILID
python3 analysis/recover_old_split.py --model_path ../weights/iLIDSVID_partxcam_best.pth
"""
import argparse
import json
import os
import shutil
import sys

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import load_model  # noqa: E402
from Dataloader import VideoDataset, val_collate_fn  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--num_train', type=int, default=150)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--dry_run', action='store_true', help='only print, do not write splits.json')
    a = p.parse_args()

    root = os.environ['ILIDS_ROOT']
    seq = os.path.join(root, 'i-LIDS-VID', 'sequences')
    cams = ['cam1', 'cam2']
    persons = sorted({d for c in cams for d in os.listdir(os.path.join(seq, c)) if d.startswith('person')})
    items, owners = [], []
    for ci, c in enumerate(cams):
        for pid, d in enumerate(persons):
            folder = os.path.join(seq, c, d)
            if not os.path.isdir(folder):
                continue
            imgs = tuple(sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')))
            if imgs:
                items.append((imgs, pid, ci))
                owners.append(d)
    print(f'{len(persons)} identities, {len(items)} tracklets')

    tf = T.Compose([T.Resize([256, 128]), T.ToTensor(), T.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
    loader = DataLoader(VideoDataset(items, seq_len=4, sample='dense', transform=tf), batch_size=1,
                        shuffle=False, num_workers=a.num_workers, collate_fn=val_collate_fn)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(a.model_path, device)
    probs = {d: [] for d in persons}
    with torch.no_grad():
        for i, (imgs, _, _, _) in enumerate(loader):
            out = model(imgs.squeeze(0).to(device), None, return_all=True)
            prob = torch.softmax(model.classifier(out['global_bn']).float(), dim=1).mean(0)
            probs[owners[i]].append(prob.cpu())
            if (i + 1) % 100 == 0:
                print(f'  {i + 1}/{len(items)} tracklets')

    # Each classifier output belongs to exactly ONE training identity: solve a one-to-one
    # assignment (Hungarian) between classes and identities on the mean class probability.
    from scipy.optimize import linear_sum_assignment
    have = [d for d in persons if probs[d]]
    P = torch.stack([torch.stack(probs[d]).mean(0) for d in have]).numpy()   # [ids, classes]
    rows, cols = linear_sum_assignment(-P)
    n_cls = P.shape[1]
    assigned = {have[r]: (int(c), float(P[r, c])) for r, c in zip(rows, cols)}
    train = sorted(assigned)
    test = sorted(d for d in have if d not in assigned)
    tr_p = sorted(v[1] for v in assigned.values())
    te_max = sorted((float(P[have.index(d)].max()) for d in test), reverse=True)
    print(f'classes: {n_cls}; assigned train ids: {len(train)}')
    print(f'assigned prob of train ids: min {tr_p[0]:.3f}, 5th lowest {tr_p[min(4, len(tr_p) - 1)]:.3f}, median {tr_p[len(tr_p) // 2]:.3f}')
    print(f'highest prob of any non-assigned id: {te_max[0]:.3f}, 5th highest {te_max[min(4, len(te_max) - 1)]:.3f}')
    # agreement with the simple top-150-by-confidence rule
    top = sorted(have, key=lambda d: -float(P[have.index(d)].max()))[:a.num_train]
    print(f'agreement with top-{a.num_train} confidence rule: {len(set(top) & set(train))}/{len(train)}')
    # identities that do not have a tracklet in BOTH cameras cannot be used by the loader
    complete = {d for d in persons if all(os.path.isdir(os.path.join(seq, c, d)) for c in cams)}
    dropped = sorted(set(persons) - complete)
    if dropped:
        print('identities missing in one camera (dropped from the written split):',
              [(d, 'train' if d in train else 'test') for d in dropped])
    train = [d for d in train if d in complete]
    test = [d for d in test if d in complete]

    sorted_split = None
    sp = os.path.join(root, 'splits.json')
    if os.path.exists(sp):
        sorted_split = json.load(open(sp))[0]
        overlap = len(set(train) & set(sorted_split['test']))
        print(f'recovered old-train ids that are in the current split-0 TEST set: {overlap}/{len(train)}')

    if a.dry_run:
        return
    if os.path.exists(sp):
        shutil.copy(sp, os.path.join(root, 'splits_backup.json'))
    json.dump([{'train': train, 'test': test}], open(sp, 'w'), indent=1)
    print(f'[saved] recovered split written to {sp} (old file kept as splits_backup.json)')


if __name__ == '__main__':
    main()
