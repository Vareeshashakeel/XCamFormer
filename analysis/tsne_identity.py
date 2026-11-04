"""
Clip-level features of PRID2011 test identities for an identity-coloured t-SNE (Fig. 7).

For every test tracklet, up to --clips evenly spaced 4-frame clips are passed through each
model and the global (BN) feature of every clip is saved. Output: <out>.npz
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import load_model  # noqa: E402
from Dataloader import test_loaders  # noqa: E402


@torch.no_grad()
def clip_feats(model, loaders, device, n_clips, blocks=(3, 4, 5, 8), frame_level=False):
    feats, pids, cams = [], [], []
    bfeats = {b: [] for b in blocks}
    for name in ('query', 'gallery'):
        for imgs, b_pids, b_cams, _ in loaders[name]:
            imgs = imgs.squeeze(0)                          # [chunks, T, 3, H, W]
            if frame_level:
                # single-frame features: take n_clips frames spread over the whole tracklet and
                # feed each frame as a static 4-frame clip (frame repeated), so every point is one frame
                fr = imgs.reshape(-1, *imgs.shape[2:])                     # [chunks*T, 3, H, W]
                idx = np.unique(np.linspace(0, fr.shape[0] - 1, n_clips).round().astype(int))
                x = fr[idx].unsqueeze(1).repeat(1, imgs.shape[1], 1, 1, 1)  # [n, T, 3, H, W]
            else:
                idx = np.unique(np.linspace(0, imgs.shape[0] - 1, n_clips).round().astype(int))
                x = imgs[idx]
            r = model(x.to(device), None, return_all=True, out_indices=blocks)
            f = r['global_bn'].float().cpu().numpy()        # [clips, 768]
            feats.append(f)
            for b in blocks:
                bfeats[b].append(r['blocks'][b].float().cpu().numpy())
            pids += [int(b_pids[0])] * len(f)
            cams += [int(b_cams.view(-1)[0].item())] * len(f)
    blk = {b: np.concatenate(v).astype(np.float16) for b, v in bfeats.items()}
    return np.concatenate(feats).astype(np.float16), np.array(pids), np.array(cams), blk


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', required=True, help='name=path pairs')
    p.add_argument('--clips', type=int, default=10)
    p.add_argument('--frames', action='store_true', help='single-frame features instead of 4-frame clips')
    p.add_argument('--out', default='results/tsne_identity_prid')
    a = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loaders, _ = test_loaders('PRID', split_id=0, seq_len=4, num_workers=2, which=('query', 'gallery'))
    save = {}
    for item in a.models:
        name, path = item.split('=', 1)
        if not os.path.exists(path):
            print(f'[skip] {name}: {path} not found')
            continue
        print(f'[run] {name}: {path}')
        f, pid, cam, blk = clip_feats(load_model(path, device), loaders, device, a.clips, frame_level=a.frames)
        save[f'{name}_feat'], save[f'{name}_pid'], save[f'{name}_cam'] = f, pid, cam
        for b, v in blk.items():
            save[f'{name}_block{b}'] = v
        print(f'      {len(f)} clips, {len(set(pid.tolist()))} identities')
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    np.savez_compressed(a.out + '.npz', **save)
    print(f'[saved] {a.out}.npz')


if __name__ == '__main__':
    main()
